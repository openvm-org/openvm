// Lean compiler output
// Module: Mathlib.Tactic.NormNum.Inv
// Imports: public import Init public meta import Init public import Mathlib.Data.Rat.Cast.CharZero public import Mathlib.Tactic.NormNum.Basic
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
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* lp_Qq_Qq_synthInstanceQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* l_Rat_ofInt(lean_object*);
lean_object* lp_Qq_Qq_trySynthInstanceQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LOption_toOption___redArg(lean_object*);
lean_object* l_Lean_Meta_whnfR(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_derive(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toInt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isRat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toNNRat_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferDivisionSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Rat_blt(lean_object*, lean_object*);
lean_object* l_Rat_inv(lean_object*);
lean_object* lp_batteries_Lean_Expr_natLit_x21(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_Lean_mkRawNatLit(lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_instDecidableEqRat_decEq(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferDivisionRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* lp_Qq_Qq_assertDefEqQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_appArg_x21(lean_object*);
lean_object* l_Rat_neg(lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Level_succ___override(lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_matchesInstance(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__4_value;
static const lean_array_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__7_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__7_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "withReducible"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__11_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__11_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(197, 44, 223, 192, 8, 197, 146, 83)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "with_reducible"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__12_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__13;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__14;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "assumption"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__16_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__16_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__16_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__16_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(240, 50, 167, 190, 65, 82, 149, 231)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__16_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__17;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__18;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__19;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__20;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__21;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__22;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__23;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__24;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__25;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__26;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__27;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__28;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__29;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__30;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__31;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__32_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__32;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__33_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__33;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "CharZero"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__0_value),LEAN_SCALAR_PTR_LITERAL(244, 4, 154, 112, 35, 213, 145, 119)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "AddGroupWithOne"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "toAddMonoidWithOne"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__2_value),LEAN_SCALAR_PTR_LITERAL(88, 61, 45, 121, 84, 135, 11, 188)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__3_value),LEAN_SCALAR_PTR_LITERAL(226, 82, 90, 134, 221, 253, 108, 55)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Ring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "toAddGroupWithOne"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__5_value),LEAN_SCALAR_PTR_LITERAL(151, 9, 120, 97, 235, 184, 251, 227)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__6_value),LEAN_SCALAR_PTR_LITERAL(99, 161, 243, 168, 232, 89, 236, 229)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "not a characteristic zero ring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__9;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing_x3f___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfAddMonoidWithOne___auto__1;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfAddMonoidWithOne___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 43, .m_capacity = 43, .m_length = 42, .m_data = "not a characteristic zero AddMonoidWithOne"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfAddMonoidWithOne___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfAddMonoidWithOne___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfAddMonoidWithOne___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfAddMonoidWithOne___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfAddMonoidWithOne(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfAddMonoidWithOne___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfAddMonoidWithOne_x3f___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfAddMonoidWithOne_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfAddMonoidWithOne_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing___auto__1;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "DivisionRing"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "toRing"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing___closed__0_value),LEAN_SCALAR_PTR_LITERAL(34, 214, 17, 155, 7, 71, 232, 190)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing___closed__1_value),LEAN_SCALAR_PTR_LITERAL(196, 15, 37, 9, 106, 139, 236, 93)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 40, .m_capacity = 40, .m_length = 39, .m_data = "not a characteristic zero division ring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___auto__1;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "AddCommMonoidWithOne"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(126, 216, 146, 120, 99, 62, 20, 70)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__3_value),LEAN_SCALAR_PTR_LITERAL(172, 33, 204, 185, 213, 137, 110, 97)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "NonAssocSemiring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "toAddCommMonoidWithOne"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(46, 119, 91, 198, 213, 11, 55, 139)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__3_value),LEAN_SCALAR_PTR_LITERAL(2, 121, 193, 151, 116, 56, 170, 8)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Semiring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "toNonAssocSemiring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__5_value),LEAN_SCALAR_PTR_LITERAL(37, 127, 172, 14, 25, 240, 239, 179)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__6_value),LEAN_SCALAR_PTR_LITERAL(146, 92, 66, 67, 127, 202, 60, 223)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "DivisionSemiring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "toSemiring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 208, 64, 71, 63, 26, 215, 130)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__9_value),LEAN_SCALAR_PTR_LITERAL(241, 35, 131, 210, 203, 216, 146, 177)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__10_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing_x3f___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "mkRat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Int"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__5;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instRing"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__6_value),LEAN_SCALAR_PTR_LITERAL(42, 135, 58, 37, 72, 75, 21, 158)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__8;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__9_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__10_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__11;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Rat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__13_value),LEAN_SCALAR_PTR_LITERAL(231, 55, 105, 214, 206, 30, 120, 51)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__14_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__15;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HDiv"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hDiv"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__16_value),LEAN_SCALAR_PTR_LITERAL(74, 223, 78, 88, 255, 236, 144, 164)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__18_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__17_value),LEAN_SCALAR_PTR_LITERAL(26, 183, 188, 240, 156, 118, 170, 84)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__19_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__20_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__21;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__22;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__23;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__24;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instHDiv"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__25_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__25_value),LEAN_SCALAR_PTR_LITERAL(34, 70, 113, 198, 157, 211, 131, 18)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__26_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__27;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__28;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "instDiv"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__29_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__30_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__13_value),LEAN_SCALAR_PTR_LITERAL(231, 55, 105, 214, 206, 30, 120, 51)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__30_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__29_value),LEAN_SCALAR_PTR_LITERAL(136, 163, 206, 229, 214, 76, 207, 233)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__30_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__31;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__32_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__32;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__33_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__33;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "cast"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__34_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__35_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__35_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__34_value),LEAN_SCALAR_PTR_LITERAL(181, 4, 252, 84, 28, 16, 24, 6)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__35_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__36_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__36;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__37_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__37;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "instIntCast"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__38_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__39_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__13_value),LEAN_SCALAR_PTR_LITERAL(231, 55, 105, 214, 206, 30, 120, 51)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__39_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__38_value),LEAN_SCALAR_PTR_LITERAL(137, 132, 79, 174, 153, 204, 15, 82)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__39 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__39_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__40_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__40;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__41_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__41;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__42_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__9_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__42_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__34_value),LEAN_SCALAR_PTR_LITERAL(19, 237, 167, 212, 100, 179, 19, 112)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__42 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__42_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__43_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__43;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__44_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__44;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "instNatCast"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__45 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__45_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__46_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__13_value),LEAN_SCALAR_PTR_LITERAL(231, 55, 105, 214, 206, 30, 120, 51)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__46_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__45_value),LEAN_SCALAR_PTR_LITERAL(212, 188, 9, 184, 182, 122, 248, 243)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__46 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__46_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__47_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__47;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__48_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__48;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "instDivisionRing"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__49 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__49_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__50_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__13_value),LEAN_SCALAR_PTR_LITERAL(231, 55, 105, 214, 206, 30, 120, 51)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__50_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__49_value),LEAN_SCALAR_PTR_LITERAL(50, 224, 126, 109, 69, 205, 182, 92)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__50 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__50_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__51_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__51;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__52 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__52_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__53 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__53_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "NormNum"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__54 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__54_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "isRat_mkRat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__55 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__55_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__56_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__52_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__56_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__56_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__53_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__56_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__56_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__54_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__56_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__56_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__55_value),LEAN_SCALAR_PTR_LITERAL(11, 76, 165, 152, 126, 130, 255, 148)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__56 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__56_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__57_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__57;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "evalMkRat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__52_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__53_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___closed__2_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__54_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 105, 221, 18, 0, 128, 244, 67)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___closed__2_value),LEAN_SCALAR_PTR_LITERAL(1, 1, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___closed__3_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "NNRat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "divNat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "AddMonoidWithOne"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(208, 217, 98, 171, 152, 255, 249, 89)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__6;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__7;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__8;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__9;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(208, 217, 98, 171, 152, 255, 249, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__29_value),LEAN_SCALAR_PTR_LITERAL(75, 159, 136, 66, 17, 69, 39, 86)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__10_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__11;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__12;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__13;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__14;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "toNatCast"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(113, 54, 100, 45, 135, 24, 207, 244)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__16_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__15_value),LEAN_SCALAR_PTR_LITERAL(83, 227, 187, 63, 172, 112, 247, 90)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__16_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__17;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__18;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__19;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__20;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__21;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__22;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__23;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "CommSemiring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__24_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__25_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__24_value),LEAN_SCALAR_PTR_LITERAL(22, 69, 197, 205, 197, 50, 81, 124)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__25_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__9_value),LEAN_SCALAR_PTR_LITERAL(1, 71, 172, 115, 76, 22, 6, 37)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__25_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__26;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__27;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "instCommSemiringNNRat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__28_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__28_value),LEAN_SCALAR_PTR_LITERAL(208, 209, 59, 60, 113, 177, 238, 123)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__29_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__30;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__31;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__32_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__32;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__33_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__33;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__34_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__34;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__35_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__35;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__36_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__36;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Semifield"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__37_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "toDivisionSemiring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__38_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__39_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__37_value),LEAN_SCALAR_PTR_LITERAL(205, 214, 159, 28, 31, 185, 116, 83)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__39_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__38_value),LEAN_SCALAR_PTR_LITERAL(121, 136, 96, 129, 245, 140, 119, 141)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__39 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__39_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__40_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__40;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__41_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__41;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "instSemifield"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__42 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__42_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__43_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(208, 217, 98, 171, 152, 255, 249, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__43_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__42_value),LEAN_SCALAR_PTR_LITERAL(205, 57, 222, 62, 67, 95, 206, 37)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__43 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__43_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__44_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__44;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__45_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__45;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "isNNRat_divNat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__46 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__46_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__47_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__52_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__47_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__47_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__53_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__47_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__47_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__54_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__47_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__46_value),LEAN_SCALAR_PTR_LITERAL(71, 128, 123, 35, 213, 153, 89, 224)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__47 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__47_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__48_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__48;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "evalNNRatDivNat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__52_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__53_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___closed__2_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__54_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___closed__1_value),LEAN_SCALAR_PTR_LITERAL(53, 214, 184, 59, 187, 7, 121, 175)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___closed__2_value),LEAN_SCALAR_PTR_LITERAL(1, 1, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___closed__3_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalRatCast_spec__0___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalRatCast_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalRatCast_spec__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalRatCast_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__13_value),LEAN_SCALAR_PTR_LITERAL(231, 55, 105, 214, 206, 30, 120, 51)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__0_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__34_value),LEAN_SCALAR_PTR_LITERAL(151, 226, 138, 206, 139, 44, 193, 88)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "toRatCast"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing___closed__0_value),LEAN_SCALAR_PTR_LITERAL(34, 214, 17, 155, 7, 71, 232, 190)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(10, 55, 59, 222, 234, 254, 235, 26)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "isNat_ratCast"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__52_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__53_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__54_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(105, 123, 69, 58, 198, 190, 228, 227)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "negOfNat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(100, 231, 152, 184, 84, 220, 144, 243)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__7;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "isInt_ratCast"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__52_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__9_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__53_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__9_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__9_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__54_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__9_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(11, 43, 163, 177, 243, 231, 177, 118)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 208, 64, 71, 63, 26, 215, 130)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "inferInstance"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(17, 162, 120, 176, 98, 85, 114, 76)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing___closed__0_value),LEAN_SCALAR_PTR_LITERAL(34, 214, 17, 155, 7, 71, 232, 190)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__13_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__38_value),LEAN_SCALAR_PTR_LITERAL(66, 176, 51, 228, 18, 108, 54, 75)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "isNNRat_ratCast"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__52_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__15_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__15_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__53_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__15_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__15_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__54_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__15_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(138, 80, 24, 84, 61, 227, 157, 80)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "isRat_ratCast"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__52_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__17_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__17_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__53_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__17_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__17_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__54_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__17_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__16_value),LEAN_SCALAR_PTR_LITERAL(82, 117, 197, 162, 236, 236, 50, 115)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__17_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "evalRatCast"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__52_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__53_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___closed__2_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__54_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___closed__1_value),LEAN_SCALAR_PTR_LITERAL(48, 1, 182, 80, 210, 177, 151, 33)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___closed__2_value),LEAN_SCALAR_PTR_LITERAL(1, 1, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___closed__3_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalRatCast = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_NormNum_evalNNRatCast_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_NormNum_evalNNRatCast_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_NormNum_evalNNRatCast_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_NormNum_evalNNRatCast_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__0___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(208, 217, 98, 171, 152, 255, 249, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__0___closed__0_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__34_value),LEAN_SCALAR_PTR_LITERAL(60, 54, 121, 2, 219, 220, 7, 222)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "NNRatCast"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(61, 217, 231, 3, 133, 178, 168, 33)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "toNNRatCast"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 208, 64, 71, 63, 26, 215, 130)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(91, 148, 22, 247, 181, 204, 251, 118)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "isNat_nnratCast"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__52_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__53_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__5_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__54_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__5_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(78, 158, 37, 178, 182, 96, 20, 248)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "isNNRat_nnratCast"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__52_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__53_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__7_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__54_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__7_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(106, 157, 152, 206, 208, 20, 23, 49)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___boxed, .m_arity = 9, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))} };
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "evalNNRatCast"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__52_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__53_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___closed__2_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__54_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___closed__1_value),LEAN_SCALAR_PTR_LITERAL(234, 255, 181, 5, 11, 242, 118, 148)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___closed__2_value),LEAN_SCALAR_PTR_LITERAL(1, 1, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___closed__3_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Mathlib_Meta_NormNum_Result_inv_spec__0(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "isNat_inv_one"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__52_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__53_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__54_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__0_value),LEAN_SCALAR_PTR_LITERAL(218, 149, 108, 194, 183, 185, 108, 38)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "isNat_inv_zero"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__52_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__53_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__54_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__3_value),LEAN_SCALAR_PTR_LITERAL(210, 6, 124, 250, 109, 48, 234, 23)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Inv"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "inv"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__5_value),LEAN_SCALAR_PTR_LITERAL(142, 68, 231, 210, 96, 163, 154, 19)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__6_value),LEAN_SCALAR_PTR_LITERAL(63, 31, 248, 222, 13, 64, 40, 141)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "InvOneClass"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toInv"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__8_value),LEAN_SCALAR_PTR_LITERAL(120, 190, 7, 179, 62, 236, 21, 116)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__9_value),LEAN_SCALAR_PTR_LITERAL(28, 25, 248, 9, 15, 85, 72, 194)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "DivInvOneMonoid"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "toInvOneClass"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__11_value),LEAN_SCALAR_PTR_LITERAL(162, 155, 123, 0, 237, 243, 28, 65)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__13_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__12_value),LEAN_SCALAR_PTR_LITERAL(181, 224, 200, 199, 184, 130, 54, 26)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "DivisionMonoid"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "toDivInvOneMonoid"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__14_value),LEAN_SCALAR_PTR_LITERAL(16, 242, 184, 157, 107, 26, 18, 78)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__16_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__15_value),LEAN_SCALAR_PTR_LITERAL(60, 63, 43, 77, 240, 6, 89, 70)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "GroupWithZero"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "toDivisionMonoid"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__17_value),LEAN_SCALAR_PTR_LITERAL(55, 132, 4, 209, 65, 207, 153, 53)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__19_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__18_value),LEAN_SCALAR_PTR_LITERAL(198, 76, 78, 187, 42, 89, 29, 20)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__19_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "toGroupWithZero"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__21_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 208, 64, 71, 63, 26, 215, 130)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__21_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__20_value),LEAN_SCALAR_PTR_LITERAL(164, 129, 71, 97, 30, 189, 214, 64)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__21_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "isNNRat_inv_pos"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__23_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__52_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__23_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__23_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__53_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__23_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__23_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__54_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__23_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__22_value),LEAN_SCALAR_PTR_LITERAL(10, 227, 249, 73, 61, 205, 23, 101)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__23_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__24;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "isInt_inv_neg_one"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__25_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__26_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__52_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__26_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__26_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__53_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__26_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__26_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__54_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__26_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__25_value),LEAN_SCALAR_PTR_LITERAL(181, 130, 57, 157, 231, 31, 4, 104)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__26_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "isRat_inv_neg"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__27_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__28_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__52_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__28_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__28_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__53_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__28_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__28_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__54_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__28_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__27_value),LEAN_SCALAR_PTR_LITERAL(133, 13, 69, 133, 148, 149, 113, 184)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__28_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__29;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Nat_cast___at___00Mathlib_Meta_NormNum_Result_inv_spec__0_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalInv___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalInv___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_NormNum_evalInv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_NormNum_evalInv___lam__1___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalInv___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalInv___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalInv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "evalInv"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalInv___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalInv___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalInv___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__52_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalInv___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalInv___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__53_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalInv___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalInv___closed__2_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__54_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalInv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalInv___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalInv___closed__1_value),LEAN_SCALAR_PTR_LITERAL(5, 30, 59, 71, 208, 158, 44, 163)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalInv___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalInv___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalInv___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalInv___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalInv___closed__2_value),LEAN_SCALAR_PTR_LITERAL(1, 1, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalInv___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalInv___closed__3_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalInv = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalInv___closed__3_value;
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__13(void){
_start:
{
lean_object* v___x_28_; lean_object* v___x_29_; 
v___x_28_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__12));
v___x_29_ = l_Lean_mkAtom(v___x_28_);
return v___x_29_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__14(void){
_start:
{
lean_object* v___x_30_; lean_object* v___x_31_; lean_object* v___x_32_; 
v___x_30_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__13, &lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__13_once, _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__13);
v___x_31_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__5));
v___x_32_ = lean_array_push(v___x_31_, v___x_30_);
return v___x_32_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__17(void){
_start:
{
lean_object* v___x_39_; lean_object* v___x_40_; 
v___x_39_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__15));
v___x_40_ = l_Lean_mkAtom(v___x_39_);
return v___x_40_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__18(void){
_start:
{
lean_object* v___x_41_; lean_object* v___x_42_; lean_object* v___x_43_; 
v___x_41_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__17, &lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__17_once, _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__17);
v___x_42_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__5));
v___x_43_ = lean_array_push(v___x_42_, v___x_41_);
return v___x_43_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__19(void){
_start:
{
lean_object* v___x_44_; lean_object* v___x_45_; lean_object* v___x_46_; lean_object* v___x_47_; 
v___x_44_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__18, &lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__18_once, _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__18);
v___x_45_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__16));
v___x_46_ = lean_box(2);
v___x_47_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_47_, 0, v___x_46_);
lean_ctor_set(v___x_47_, 1, v___x_45_);
lean_ctor_set(v___x_47_, 2, v___x_44_);
return v___x_47_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__20(void){
_start:
{
lean_object* v___x_48_; lean_object* v___x_49_; lean_object* v___x_50_; 
v___x_48_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__19, &lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__19_once, _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__19);
v___x_49_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__5));
v___x_50_ = lean_array_push(v___x_49_, v___x_48_);
return v___x_50_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__21(void){
_start:
{
lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; 
v___x_51_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__20, &lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__20_once, _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__20);
v___x_52_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__9));
v___x_53_ = lean_box(2);
v___x_54_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_54_, 0, v___x_53_);
lean_ctor_set(v___x_54_, 1, v___x_52_);
lean_ctor_set(v___x_54_, 2, v___x_51_);
return v___x_54_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__22(void){
_start:
{
lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; 
v___x_55_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__21, &lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__21_once, _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__21);
v___x_56_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__5));
v___x_57_ = lean_array_push(v___x_56_, v___x_55_);
return v___x_57_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__23(void){
_start:
{
lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; 
v___x_58_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__22, &lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__22_once, _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__22);
v___x_59_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__7));
v___x_60_ = lean_box(2);
v___x_61_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_61_, 0, v___x_60_);
lean_ctor_set(v___x_61_, 1, v___x_59_);
lean_ctor_set(v___x_61_, 2, v___x_58_);
return v___x_61_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__24(void){
_start:
{
lean_object* v___x_62_; lean_object* v___x_63_; lean_object* v___x_64_; 
v___x_62_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__23, &lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__23_once, _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__23);
v___x_63_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__5));
v___x_64_ = lean_array_push(v___x_63_, v___x_62_);
return v___x_64_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__25(void){
_start:
{
lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; lean_object* v___x_68_; 
v___x_65_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__24, &lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__24_once, _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__24);
v___x_66_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__4));
v___x_67_ = lean_box(2);
v___x_68_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_68_, 0, v___x_67_);
lean_ctor_set(v___x_68_, 1, v___x_66_);
lean_ctor_set(v___x_68_, 2, v___x_65_);
return v___x_68_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__26(void){
_start:
{
lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; 
v___x_69_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__25, &lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__25_once, _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__25);
v___x_70_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__14, &lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__14_once, _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__14);
v___x_71_ = lean_array_push(v___x_70_, v___x_69_);
return v___x_71_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__27(void){
_start:
{
lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; 
v___x_72_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__26, &lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__26_once, _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__26);
v___x_73_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__11));
v___x_74_ = lean_box(2);
v___x_75_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_75_, 0, v___x_74_);
lean_ctor_set(v___x_75_, 1, v___x_73_);
lean_ctor_set(v___x_75_, 2, v___x_72_);
return v___x_75_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__28(void){
_start:
{
lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; 
v___x_76_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__27, &lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__27_once, _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__27);
v___x_77_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__5));
v___x_78_ = lean_array_push(v___x_77_, v___x_76_);
return v___x_78_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__29(void){
_start:
{
lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; 
v___x_79_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__28, &lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__28_once, _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__28);
v___x_80_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__9));
v___x_81_ = lean_box(2);
v___x_82_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_82_, 0, v___x_81_);
lean_ctor_set(v___x_82_, 1, v___x_80_);
lean_ctor_set(v___x_82_, 2, v___x_79_);
return v___x_82_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__30(void){
_start:
{
lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; 
v___x_83_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__29, &lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__29_once, _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__29);
v___x_84_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__5));
v___x_85_ = lean_array_push(v___x_84_, v___x_83_);
return v___x_85_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__31(void){
_start:
{
lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; 
v___x_86_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__30, &lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__30_once, _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__30);
v___x_87_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__7));
v___x_88_ = lean_box(2);
v___x_89_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_89_, 0, v___x_88_);
lean_ctor_set(v___x_89_, 1, v___x_87_);
lean_ctor_set(v___x_89_, 2, v___x_86_);
return v___x_89_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__32(void){
_start:
{
lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; 
v___x_90_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__31, &lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__31_once, _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__31);
v___x_91_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__5));
v___x_92_ = lean_array_push(v___x_91_, v___x_90_);
return v___x_92_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__33(void){
_start:
{
lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; 
v___x_93_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__32, &lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__32_once, _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__32);
v___x_94_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__4));
v___x_95_ = lean_box(2);
v___x_96_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_96_, 0, v___x_95_);
lean_ctor_set(v___x_96_, 1, v___x_94_);
lean_ctor_set(v___x_96_, 2, v___x_93_);
return v___x_96_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1(void){
_start:
{
lean_object* v___x_97_; 
v___x_97_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__33, &lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__33_once, _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__33);
return v___x_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0_spec__0(lean_object* v_msgData_98_, lean_object* v___y_99_, lean_object* v___y_100_, lean_object* v___y_101_, lean_object* v___y_102_){
_start:
{
lean_object* v___x_104_; lean_object* v_env_105_; lean_object* v___x_106_; lean_object* v_mctx_107_; lean_object* v_lctx_108_; lean_object* v_options_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; 
v___x_104_ = lean_st_ref_get(v___y_102_);
v_env_105_ = lean_ctor_get(v___x_104_, 0);
lean_inc_ref(v_env_105_);
lean_dec(v___x_104_);
v___x_106_ = lean_st_ref_get(v___y_100_);
v_mctx_107_ = lean_ctor_get(v___x_106_, 0);
lean_inc_ref(v_mctx_107_);
lean_dec(v___x_106_);
v_lctx_108_ = lean_ctor_get(v___y_99_, 2);
v_options_109_ = lean_ctor_get(v___y_101_, 2);
lean_inc_ref(v_options_109_);
lean_inc_ref(v_lctx_108_);
v___x_110_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_110_, 0, v_env_105_);
lean_ctor_set(v___x_110_, 1, v_mctx_107_);
lean_ctor_set(v___x_110_, 2, v_lctx_108_);
lean_ctor_set(v___x_110_, 3, v_options_109_);
v___x_111_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_111_, 0, v___x_110_);
lean_ctor_set(v___x_111_, 1, v_msgData_98_);
v___x_112_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_112_, 0, v___x_111_);
return v___x_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0_spec__0___boxed(lean_object* v_msgData_113_, lean_object* v___y_114_, lean_object* v___y_115_, lean_object* v___y_116_, lean_object* v___y_117_, lean_object* v___y_118_){
_start:
{
lean_object* v_res_119_; 
v_res_119_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0_spec__0(v_msgData_113_, v___y_114_, v___y_115_, v___y_116_, v___y_117_);
lean_dec(v___y_117_);
lean_dec_ref(v___y_116_);
lean_dec(v___y_115_);
lean_dec_ref(v___y_114_);
return v_res_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0___redArg(lean_object* v_msg_120_, lean_object* v___y_121_, lean_object* v___y_122_, lean_object* v___y_123_, lean_object* v___y_124_){
_start:
{
lean_object* v_ref_126_; lean_object* v___x_127_; lean_object* v_a_128_; lean_object* v___x_130_; uint8_t v_isShared_131_; uint8_t v_isSharedCheck_136_; 
v_ref_126_ = lean_ctor_get(v___y_123_, 5);
v___x_127_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0_spec__0(v_msg_120_, v___y_121_, v___y_122_, v___y_123_, v___y_124_);
v_a_128_ = lean_ctor_get(v___x_127_, 0);
v_isSharedCheck_136_ = !lean_is_exclusive(v___x_127_);
if (v_isSharedCheck_136_ == 0)
{
v___x_130_ = v___x_127_;
v_isShared_131_ = v_isSharedCheck_136_;
goto v_resetjp_129_;
}
else
{
lean_inc(v_a_128_);
lean_dec(v___x_127_);
v___x_130_ = lean_box(0);
v_isShared_131_ = v_isSharedCheck_136_;
goto v_resetjp_129_;
}
v_resetjp_129_:
{
lean_object* v___x_132_; lean_object* v___x_134_; 
lean_inc(v_ref_126_);
v___x_132_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_132_, 0, v_ref_126_);
lean_ctor_set(v___x_132_, 1, v_a_128_);
if (v_isShared_131_ == 0)
{
lean_ctor_set_tag(v___x_130_, 1);
lean_ctor_set(v___x_130_, 0, v___x_132_);
v___x_134_ = v___x_130_;
goto v_reusejp_133_;
}
else
{
lean_object* v_reuseFailAlloc_135_; 
v_reuseFailAlloc_135_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_135_, 0, v___x_132_);
v___x_134_ = v_reuseFailAlloc_135_;
goto v_reusejp_133_;
}
v_reusejp_133_:
{
return v___x_134_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0___redArg___boxed(lean_object* v_msg_137_, lean_object* v___y_138_, lean_object* v___y_139_, lean_object* v___y_140_, lean_object* v___y_141_, lean_object* v___y_142_){
_start:
{
lean_object* v_res_143_; 
v_res_143_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0___redArg(v_msg_137_, v___y_138_, v___y_139_, v___y_140_, v___y_141_);
lean_dec(v___y_141_);
lean_dec_ref(v___y_140_);
lean_dec(v___y_139_);
lean_dec_ref(v___y_138_);
return v_res_143_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__9(void){
_start:
{
lean_object* v___x_158_; lean_object* v___x_159_; 
v___x_158_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__8));
v___x_159_ = l_Lean_stringToMessageData(v___x_158_);
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing(lean_object* v_u_160_, lean_object* v_00_u03b1_161_, lean_object* v___i_162_, lean_object* v_a_163_, lean_object* v_a_164_, lean_object* v_a_165_, lean_object* v_a_166_){
_start:
{
lean_object* v___x_168_; 
v___x_168_ = l_Lean_Meta_saveState___redArg(v_a_164_, v_a_166_);
if (lean_obj_tag(v___x_168_) == 0)
{
lean_object* v_a_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; 
v_a_169_ = lean_ctor_get(v___x_168_, 0);
lean_inc(v_a_169_);
lean_dec_ref_known(v___x_168_, 1);
v___x_170_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__1));
v___x_171_ = lean_box(0);
v___x_172_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_172_, 0, v_u_160_);
lean_ctor_set(v___x_172_, 1, v___x_171_);
lean_inc_ref_n(v___x_172_, 2);
v___x_173_ = l_Lean_Expr_const___override(v___x_170_, v___x_172_);
lean_inc_ref_n(v_00_u03b1_161_, 2);
v___x_174_ = l_Lean_Expr_app___override(v___x_173_, v_00_u03b1_161_);
v___x_175_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__4));
v___x_176_ = l_Lean_Expr_const___override(v___x_175_, v___x_172_);
v___x_177_ = l_Lean_Expr_app___override(v___x_176_, v_00_u03b1_161_);
v___x_178_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__7));
v___x_179_ = l_Lean_Expr_const___override(v___x_178_, v___x_172_);
v___x_180_ = l_Lean_Expr_app___override(v___x_179_, v_00_u03b1_161_);
v___x_181_ = l_Lean_Expr_app___override(v___x_180_, v___i_162_);
v___x_182_ = l_Lean_Expr_app___override(v___x_177_, v___x_181_);
v___x_183_ = l_Lean_Expr_app___override(v___x_174_, v___x_182_);
v___x_184_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_183_, v_a_163_, v_a_164_, v_a_165_, v_a_166_);
if (lean_obj_tag(v___x_184_) == 0)
{
lean_dec(v_a_169_);
return v___x_184_;
}
else
{
lean_object* v_a_185_; uint8_t v___y_187_; uint8_t v___x_199_; 
v_a_185_ = lean_ctor_get(v___x_184_, 0);
lean_inc(v_a_185_);
v___x_199_ = l_Lean_Exception_isInterrupt(v_a_185_);
if (v___x_199_ == 0)
{
uint8_t v___x_200_; 
v___x_200_ = l_Lean_Exception_isRuntime(v_a_185_);
v___y_187_ = v___x_200_;
goto v___jp_186_;
}
else
{
lean_dec(v_a_185_);
v___y_187_ = v___x_199_;
goto v___jp_186_;
}
v___jp_186_:
{
if (v___y_187_ == 0)
{
lean_object* v___x_188_; 
lean_dec_ref_known(v___x_184_, 1);
v___x_188_ = l_Lean_Meta_SavedState_restore___redArg(v_a_169_, v_a_164_, v_a_166_);
lean_dec(v_a_169_);
if (lean_obj_tag(v___x_188_) == 0)
{
lean_object* v___x_189_; lean_object* v___x_190_; 
lean_dec_ref_known(v___x_188_, 1);
v___x_189_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__9, &lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__9_once, _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__9);
v___x_190_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0___redArg(v___x_189_, v_a_163_, v_a_164_, v_a_165_, v_a_166_);
return v___x_190_;
}
else
{
lean_object* v_a_191_; lean_object* v___x_193_; uint8_t v_isShared_194_; uint8_t v_isSharedCheck_198_; 
v_a_191_ = lean_ctor_get(v___x_188_, 0);
v_isSharedCheck_198_ = !lean_is_exclusive(v___x_188_);
if (v_isSharedCheck_198_ == 0)
{
v___x_193_ = v___x_188_;
v_isShared_194_ = v_isSharedCheck_198_;
goto v_resetjp_192_;
}
else
{
lean_inc(v_a_191_);
lean_dec(v___x_188_);
v___x_193_ = lean_box(0);
v_isShared_194_ = v_isSharedCheck_198_;
goto v_resetjp_192_;
}
v_resetjp_192_:
{
lean_object* v___x_196_; 
if (v_isShared_194_ == 0)
{
v___x_196_ = v___x_193_;
goto v_reusejp_195_;
}
else
{
lean_object* v_reuseFailAlloc_197_; 
v_reuseFailAlloc_197_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_197_, 0, v_a_191_);
v___x_196_ = v_reuseFailAlloc_197_;
goto v_reusejp_195_;
}
v_reusejp_195_:
{
return v___x_196_;
}
}
}
}
else
{
lean_dec(v_a_169_);
return v___x_184_;
}
}
}
}
else
{
lean_object* v_a_201_; lean_object* v___x_203_; uint8_t v_isShared_204_; uint8_t v_isSharedCheck_208_; 
lean_dec_ref(v___i_162_);
lean_dec_ref(v_00_u03b1_161_);
lean_dec(v_u_160_);
v_a_201_ = lean_ctor_get(v___x_168_, 0);
v_isSharedCheck_208_ = !lean_is_exclusive(v___x_168_);
if (v_isSharedCheck_208_ == 0)
{
v___x_203_ = v___x_168_;
v_isShared_204_ = v_isSharedCheck_208_;
goto v_resetjp_202_;
}
else
{
lean_inc(v_a_201_);
lean_dec(v___x_168_);
v___x_203_ = lean_box(0);
v_isShared_204_ = v_isSharedCheck_208_;
goto v_resetjp_202_;
}
v_resetjp_202_:
{
lean_object* v___x_206_; 
if (v_isShared_204_ == 0)
{
v___x_206_ = v___x_203_;
goto v_reusejp_205_;
}
else
{
lean_object* v_reuseFailAlloc_207_; 
v_reuseFailAlloc_207_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_207_, 0, v_a_201_);
v___x_206_ = v_reuseFailAlloc_207_;
goto v_reusejp_205_;
}
v_reusejp_205_:
{
return v___x_206_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___boxed(lean_object* v_u_209_, lean_object* v_00_u03b1_210_, lean_object* v___i_211_, lean_object* v_a_212_, lean_object* v_a_213_, lean_object* v_a_214_, lean_object* v_a_215_, lean_object* v_a_216_){
_start:
{
lean_object* v_res_217_; 
v_res_217_ = lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing(v_u_209_, v_00_u03b1_210_, v___i_211_, v_a_212_, v_a_213_, v_a_214_, v_a_215_);
lean_dec(v_a_215_);
lean_dec_ref(v_a_214_);
lean_dec(v_a_213_);
lean_dec_ref(v_a_212_);
return v_res_217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0(lean_object* v_00_u03b1_218_, lean_object* v_msg_219_, lean_object* v___y_220_, lean_object* v___y_221_, lean_object* v___y_222_, lean_object* v___y_223_){
_start:
{
lean_object* v___x_225_; 
v___x_225_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0___redArg(v_msg_219_, v___y_220_, v___y_221_, v___y_222_, v___y_223_);
return v___x_225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0___boxed(lean_object* v_00_u03b1_226_, lean_object* v_msg_227_, lean_object* v___y_228_, lean_object* v___y_229_, lean_object* v___y_230_, lean_object* v___y_231_, lean_object* v___y_232_){
_start:
{
lean_object* v_res_233_; 
v_res_233_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0(v_00_u03b1_226_, v_msg_227_, v___y_228_, v___y_229_, v___y_230_, v___y_231_);
lean_dec(v___y_231_);
lean_dec_ref(v___y_230_);
lean_dec(v___y_229_);
lean_dec_ref(v___y_228_);
return v_res_233_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing_x3f___auto__1(void){
_start:
{
lean_object* v___x_234_; 
v___x_234_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__33, &lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__33_once, _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__33);
return v___x_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing_x3f(lean_object* v_u_235_, lean_object* v_00_u03b1_236_, lean_object* v___i_237_, lean_object* v_a_238_, lean_object* v_a_239_, lean_object* v_a_240_, lean_object* v_a_241_){
_start:
{
lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; 
v___x_243_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__1));
v___x_244_ = lean_box(0);
v___x_245_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_245_, 0, v_u_235_);
lean_ctor_set(v___x_245_, 1, v___x_244_);
lean_inc_ref_n(v___x_245_, 2);
v___x_246_ = l_Lean_Expr_const___override(v___x_243_, v___x_245_);
lean_inc_ref_n(v_00_u03b1_236_, 2);
v___x_247_ = l_Lean_Expr_app___override(v___x_246_, v_00_u03b1_236_);
v___x_248_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__4));
v___x_249_ = l_Lean_Expr_const___override(v___x_248_, v___x_245_);
v___x_250_ = l_Lean_Expr_app___override(v___x_249_, v_00_u03b1_236_);
v___x_251_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__7));
v___x_252_ = l_Lean_Expr_const___override(v___x_251_, v___x_245_);
v___x_253_ = l_Lean_Expr_app___override(v___x_252_, v_00_u03b1_236_);
v___x_254_ = l_Lean_Expr_app___override(v___x_253_, v___i_237_);
v___x_255_ = l_Lean_Expr_app___override(v___x_250_, v___x_254_);
v___x_256_ = l_Lean_Expr_app___override(v___x_247_, v___x_255_);
v___x_257_ = lp_Qq_Qq_trySynthInstanceQ___redArg(v___x_256_, v_a_238_, v_a_239_, v_a_240_, v_a_241_);
if (lean_obj_tag(v___x_257_) == 0)
{
lean_object* v_a_258_; lean_object* v___x_260_; uint8_t v_isShared_261_; uint8_t v_isSharedCheck_266_; 
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
v___x_262_ = l_Lean_LOption_toOption___redArg(v_a_258_);
if (v_isShared_261_ == 0)
{
lean_ctor_set(v___x_260_, 0, v___x_262_);
v___x_264_ = v___x_260_;
goto v_reusejp_263_;
}
else
{
lean_object* v_reuseFailAlloc_265_; 
v_reuseFailAlloc_265_ = lean_alloc_ctor(0, 1, 0);
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
else
{
lean_object* v_a_267_; lean_object* v___x_269_; uint8_t v_isShared_270_; uint8_t v_isSharedCheck_274_; 
v_a_267_ = lean_ctor_get(v___x_257_, 0);
v_isSharedCheck_274_ = !lean_is_exclusive(v___x_257_);
if (v_isSharedCheck_274_ == 0)
{
v___x_269_ = v___x_257_;
v_isShared_270_ = v_isSharedCheck_274_;
goto v_resetjp_268_;
}
else
{
lean_inc(v_a_267_);
lean_dec(v___x_257_);
v___x_269_ = lean_box(0);
v_isShared_270_ = v_isSharedCheck_274_;
goto v_resetjp_268_;
}
v_resetjp_268_:
{
lean_object* v___x_272_; 
if (v_isShared_270_ == 0)
{
v___x_272_ = v___x_269_;
goto v_reusejp_271_;
}
else
{
lean_object* v_reuseFailAlloc_273_; 
v_reuseFailAlloc_273_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_273_, 0, v_a_267_);
v___x_272_ = v_reuseFailAlloc_273_;
goto v_reusejp_271_;
}
v_reusejp_271_:
{
return v___x_272_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing_x3f___boxed(lean_object* v_u_275_, lean_object* v_00_u03b1_276_, lean_object* v___i_277_, lean_object* v_a_278_, lean_object* v_a_279_, lean_object* v_a_280_, lean_object* v_a_281_, lean_object* v_a_282_){
_start:
{
lean_object* v_res_283_; 
v_res_283_ = lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing_x3f(v_u_275_, v_00_u03b1_276_, v___i_277_, v_a_278_, v_a_279_, v_a_280_, v_a_281_);
lean_dec(v_a_281_);
lean_dec_ref(v_a_280_);
lean_dec(v_a_279_);
lean_dec_ref(v_a_278_);
return v_res_283_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfAddMonoidWithOne___auto__1(void){
_start:
{
lean_object* v___x_284_; 
v___x_284_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__33, &lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__33_once, _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__33);
return v___x_284_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfAddMonoidWithOne___closed__1(void){
_start:
{
lean_object* v___x_286_; lean_object* v___x_287_; 
v___x_286_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfAddMonoidWithOne___closed__0));
v___x_287_ = l_Lean_stringToMessageData(v___x_286_);
return v___x_287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfAddMonoidWithOne(lean_object* v_u_288_, lean_object* v_00_u03b1_289_, lean_object* v___i_290_, lean_object* v_a_291_, lean_object* v_a_292_, lean_object* v_a_293_, lean_object* v_a_294_){
_start:
{
lean_object* v___x_296_; 
v___x_296_ = l_Lean_Meta_saveState___redArg(v_a_292_, v_a_294_);
if (lean_obj_tag(v___x_296_) == 0)
{
lean_object* v_a_297_; lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v___x_304_; 
v_a_297_ = lean_ctor_get(v___x_296_, 0);
lean_inc(v_a_297_);
lean_dec_ref_known(v___x_296_, 1);
v___x_298_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__1));
v___x_299_ = lean_box(0);
v___x_300_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_300_, 0, v_u_288_);
lean_ctor_set(v___x_300_, 1, v___x_299_);
v___x_301_ = l_Lean_Expr_const___override(v___x_298_, v___x_300_);
v___x_302_ = l_Lean_Expr_app___override(v___x_301_, v_00_u03b1_289_);
v___x_303_ = l_Lean_Expr_app___override(v___x_302_, v___i_290_);
v___x_304_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_303_, v_a_291_, v_a_292_, v_a_293_, v_a_294_);
if (lean_obj_tag(v___x_304_) == 0)
{
lean_dec(v_a_297_);
return v___x_304_;
}
else
{
lean_object* v_a_305_; uint8_t v___y_307_; uint8_t v___x_319_; 
v_a_305_ = lean_ctor_get(v___x_304_, 0);
lean_inc(v_a_305_);
v___x_319_ = l_Lean_Exception_isInterrupt(v_a_305_);
if (v___x_319_ == 0)
{
uint8_t v___x_320_; 
v___x_320_ = l_Lean_Exception_isRuntime(v_a_305_);
v___y_307_ = v___x_320_;
goto v___jp_306_;
}
else
{
lean_dec(v_a_305_);
v___y_307_ = v___x_319_;
goto v___jp_306_;
}
v___jp_306_:
{
if (v___y_307_ == 0)
{
lean_object* v___x_308_; 
lean_dec_ref_known(v___x_304_, 1);
v___x_308_ = l_Lean_Meta_SavedState_restore___redArg(v_a_297_, v_a_292_, v_a_294_);
lean_dec(v_a_297_);
if (lean_obj_tag(v___x_308_) == 0)
{
lean_object* v___x_309_; lean_object* v___x_310_; 
lean_dec_ref_known(v___x_308_, 1);
v___x_309_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfAddMonoidWithOne___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfAddMonoidWithOne___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfAddMonoidWithOne___closed__1);
v___x_310_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0___redArg(v___x_309_, v_a_291_, v_a_292_, v_a_293_, v_a_294_);
return v___x_310_;
}
else
{
lean_object* v_a_311_; lean_object* v___x_313_; uint8_t v_isShared_314_; uint8_t v_isSharedCheck_318_; 
v_a_311_ = lean_ctor_get(v___x_308_, 0);
v_isSharedCheck_318_ = !lean_is_exclusive(v___x_308_);
if (v_isSharedCheck_318_ == 0)
{
v___x_313_ = v___x_308_;
v_isShared_314_ = v_isSharedCheck_318_;
goto v_resetjp_312_;
}
else
{
lean_inc(v_a_311_);
lean_dec(v___x_308_);
v___x_313_ = lean_box(0);
v_isShared_314_ = v_isSharedCheck_318_;
goto v_resetjp_312_;
}
v_resetjp_312_:
{
lean_object* v___x_316_; 
if (v_isShared_314_ == 0)
{
v___x_316_ = v___x_313_;
goto v_reusejp_315_;
}
else
{
lean_object* v_reuseFailAlloc_317_; 
v_reuseFailAlloc_317_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_317_, 0, v_a_311_);
v___x_316_ = v_reuseFailAlloc_317_;
goto v_reusejp_315_;
}
v_reusejp_315_:
{
return v___x_316_;
}
}
}
}
else
{
lean_dec(v_a_297_);
return v___x_304_;
}
}
}
}
else
{
lean_object* v_a_321_; lean_object* v___x_323_; uint8_t v_isShared_324_; uint8_t v_isSharedCheck_328_; 
lean_dec_ref(v___i_290_);
lean_dec_ref(v_00_u03b1_289_);
lean_dec(v_u_288_);
v_a_321_ = lean_ctor_get(v___x_296_, 0);
v_isSharedCheck_328_ = !lean_is_exclusive(v___x_296_);
if (v_isSharedCheck_328_ == 0)
{
v___x_323_ = v___x_296_;
v_isShared_324_ = v_isSharedCheck_328_;
goto v_resetjp_322_;
}
else
{
lean_inc(v_a_321_);
lean_dec(v___x_296_);
v___x_323_ = lean_box(0);
v_isShared_324_ = v_isSharedCheck_328_;
goto v_resetjp_322_;
}
v_resetjp_322_:
{
lean_object* v___x_326_; 
if (v_isShared_324_ == 0)
{
v___x_326_ = v___x_323_;
goto v_reusejp_325_;
}
else
{
lean_object* v_reuseFailAlloc_327_; 
v_reuseFailAlloc_327_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_327_, 0, v_a_321_);
v___x_326_ = v_reuseFailAlloc_327_;
goto v_reusejp_325_;
}
v_reusejp_325_:
{
return v___x_326_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfAddMonoidWithOne___boxed(lean_object* v_u_329_, lean_object* v_00_u03b1_330_, lean_object* v___i_331_, lean_object* v_a_332_, lean_object* v_a_333_, lean_object* v_a_334_, lean_object* v_a_335_, lean_object* v_a_336_){
_start:
{
lean_object* v_res_337_; 
v_res_337_ = lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfAddMonoidWithOne(v_u_329_, v_00_u03b1_330_, v___i_331_, v_a_332_, v_a_333_, v_a_334_, v_a_335_);
lean_dec(v_a_335_);
lean_dec_ref(v_a_334_);
lean_dec(v_a_333_);
lean_dec_ref(v_a_332_);
return v_res_337_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfAddMonoidWithOne_x3f___auto__1(void){
_start:
{
lean_object* v___x_338_; 
v___x_338_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__33, &lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__33_once, _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__33);
return v___x_338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfAddMonoidWithOne_x3f(lean_object* v_u_339_, lean_object* v_00_u03b1_340_, lean_object* v___i_341_, lean_object* v_a_342_, lean_object* v_a_343_, lean_object* v_a_344_, lean_object* v_a_345_){
_start:
{
lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___x_353_; 
v___x_347_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__1));
v___x_348_ = lean_box(0);
v___x_349_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_349_, 0, v_u_339_);
lean_ctor_set(v___x_349_, 1, v___x_348_);
v___x_350_ = l_Lean_Expr_const___override(v___x_347_, v___x_349_);
v___x_351_ = l_Lean_Expr_app___override(v___x_350_, v_00_u03b1_340_);
v___x_352_ = l_Lean_Expr_app___override(v___x_351_, v___i_341_);
v___x_353_ = lp_Qq_Qq_trySynthInstanceQ___redArg(v___x_352_, v_a_342_, v_a_343_, v_a_344_, v_a_345_);
if (lean_obj_tag(v___x_353_) == 0)
{
lean_object* v_a_354_; lean_object* v___x_356_; uint8_t v_isShared_357_; uint8_t v_isSharedCheck_362_; 
v_a_354_ = lean_ctor_get(v___x_353_, 0);
v_isSharedCheck_362_ = !lean_is_exclusive(v___x_353_);
if (v_isSharedCheck_362_ == 0)
{
v___x_356_ = v___x_353_;
v_isShared_357_ = v_isSharedCheck_362_;
goto v_resetjp_355_;
}
else
{
lean_inc(v_a_354_);
lean_dec(v___x_353_);
v___x_356_ = lean_box(0);
v_isShared_357_ = v_isSharedCheck_362_;
goto v_resetjp_355_;
}
v_resetjp_355_:
{
lean_object* v___x_358_; lean_object* v___x_360_; 
v___x_358_ = l_Lean_LOption_toOption___redArg(v_a_354_);
if (v_isShared_357_ == 0)
{
lean_ctor_set(v___x_356_, 0, v___x_358_);
v___x_360_ = v___x_356_;
goto v_reusejp_359_;
}
else
{
lean_object* v_reuseFailAlloc_361_; 
v_reuseFailAlloc_361_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_361_, 0, v___x_358_);
v___x_360_ = v_reuseFailAlloc_361_;
goto v_reusejp_359_;
}
v_reusejp_359_:
{
return v___x_360_;
}
}
}
else
{
lean_object* v_a_363_; lean_object* v___x_365_; uint8_t v_isShared_366_; uint8_t v_isSharedCheck_370_; 
v_a_363_ = lean_ctor_get(v___x_353_, 0);
v_isSharedCheck_370_ = !lean_is_exclusive(v___x_353_);
if (v_isSharedCheck_370_ == 0)
{
v___x_365_ = v___x_353_;
v_isShared_366_ = v_isSharedCheck_370_;
goto v_resetjp_364_;
}
else
{
lean_inc(v_a_363_);
lean_dec(v___x_353_);
v___x_365_ = lean_box(0);
v_isShared_366_ = v_isSharedCheck_370_;
goto v_resetjp_364_;
}
v_resetjp_364_:
{
lean_object* v___x_368_; 
if (v_isShared_366_ == 0)
{
v___x_368_ = v___x_365_;
goto v_reusejp_367_;
}
else
{
lean_object* v_reuseFailAlloc_369_; 
v_reuseFailAlloc_369_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_369_, 0, v_a_363_);
v___x_368_ = v_reuseFailAlloc_369_;
goto v_reusejp_367_;
}
v_reusejp_367_:
{
return v___x_368_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfAddMonoidWithOne_x3f___boxed(lean_object* v_u_371_, lean_object* v_00_u03b1_372_, lean_object* v___i_373_, lean_object* v_a_374_, lean_object* v_a_375_, lean_object* v_a_376_, lean_object* v_a_377_, lean_object* v_a_378_){
_start:
{
lean_object* v_res_379_; 
v_res_379_ = lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfAddMonoidWithOne_x3f(v_u_371_, v_00_u03b1_372_, v___i_373_, v_a_374_, v_a_375_, v_a_376_, v_a_377_);
lean_dec(v_a_377_);
lean_dec_ref(v_a_376_);
lean_dec(v_a_375_);
lean_dec_ref(v_a_374_);
return v_res_379_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing___auto__1(void){
_start:
{
lean_object* v___x_380_; 
v___x_380_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__33, &lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__33_once, _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__33);
return v___x_380_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing___closed__4(void){
_start:
{
lean_object* v___x_387_; lean_object* v___x_388_; 
v___x_387_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing___closed__3));
v___x_388_ = l_Lean_stringToMessageData(v___x_387_);
return v___x_388_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing(lean_object* v_u_389_, lean_object* v_00_u03b1_390_, lean_object* v___i_391_, lean_object* v_a_392_, lean_object* v_a_393_, lean_object* v_a_394_, lean_object* v_a_395_){
_start:
{
lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_399_; lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; 
v___x_397_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__1));
v___x_398_ = lean_box(0);
v___x_399_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_399_, 0, v_u_389_);
lean_ctor_set(v___x_399_, 1, v___x_398_);
lean_inc_ref(v___x_399_);
v___x_400_ = l_Lean_Expr_const___override(v___x_397_, v___x_399_);
lean_inc_ref(v_00_u03b1_390_);
v___x_401_ = l_Lean_Expr_app___override(v___x_400_, v_00_u03b1_390_);
v___x_402_ = l_Lean_Meta_saveState___redArg(v_a_393_, v_a_395_);
if (lean_obj_tag(v___x_402_) == 0)
{
lean_object* v_a_403_; lean_object* v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v___x_410_; lean_object* v___x_411_; lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v___x_415_; lean_object* v___x_416_; lean_object* v___x_417_; 
v_a_403_ = lean_ctor_get(v___x_402_, 0);
lean_inc(v_a_403_);
lean_dec_ref_known(v___x_402_, 1);
v___x_404_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__4));
lean_inc_ref_n(v___x_399_, 2);
v___x_405_ = l_Lean_Expr_const___override(v___x_404_, v___x_399_);
lean_inc_ref_n(v_00_u03b1_390_, 2);
v___x_406_ = l_Lean_Expr_app___override(v___x_405_, v_00_u03b1_390_);
v___x_407_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__7));
v___x_408_ = l_Lean_Expr_const___override(v___x_407_, v___x_399_);
v___x_409_ = l_Lean_Expr_app___override(v___x_408_, v_00_u03b1_390_);
v___x_410_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing___closed__2));
v___x_411_ = l_Lean_Expr_const___override(v___x_410_, v___x_399_);
v___x_412_ = l_Lean_Expr_app___override(v___x_411_, v_00_u03b1_390_);
v___x_413_ = l_Lean_Expr_app___override(v___x_412_, v___i_391_);
v___x_414_ = l_Lean_Expr_app___override(v___x_409_, v___x_413_);
v___x_415_ = l_Lean_Expr_app___override(v___x_406_, v___x_414_);
v___x_416_ = l_Lean_Expr_app___override(v___x_401_, v___x_415_);
v___x_417_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_416_, v_a_392_, v_a_393_, v_a_394_, v_a_395_);
if (lean_obj_tag(v___x_417_) == 0)
{
lean_dec(v_a_403_);
return v___x_417_;
}
else
{
lean_object* v_a_418_; uint8_t v___y_420_; uint8_t v___x_432_; 
v_a_418_ = lean_ctor_get(v___x_417_, 0);
lean_inc(v_a_418_);
v___x_432_ = l_Lean_Exception_isInterrupt(v_a_418_);
if (v___x_432_ == 0)
{
uint8_t v___x_433_; 
v___x_433_ = l_Lean_Exception_isRuntime(v_a_418_);
v___y_420_ = v___x_433_;
goto v___jp_419_;
}
else
{
lean_dec(v_a_418_);
v___y_420_ = v___x_432_;
goto v___jp_419_;
}
v___jp_419_:
{
if (v___y_420_ == 0)
{
lean_object* v___x_421_; 
lean_dec_ref_known(v___x_417_, 1);
v___x_421_ = l_Lean_Meta_SavedState_restore___redArg(v_a_403_, v_a_393_, v_a_395_);
lean_dec(v_a_403_);
if (lean_obj_tag(v___x_421_) == 0)
{
lean_object* v___x_422_; lean_object* v___x_423_; 
lean_dec_ref_known(v___x_421_, 1);
v___x_422_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing___closed__4, &lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing___closed__4_once, _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing___closed__4);
v___x_423_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0___redArg(v___x_422_, v_a_392_, v_a_393_, v_a_394_, v_a_395_);
return v___x_423_;
}
else
{
lean_object* v_a_424_; lean_object* v___x_426_; uint8_t v_isShared_427_; uint8_t v_isSharedCheck_431_; 
v_a_424_ = lean_ctor_get(v___x_421_, 0);
v_isSharedCheck_431_ = !lean_is_exclusive(v___x_421_);
if (v_isSharedCheck_431_ == 0)
{
v___x_426_ = v___x_421_;
v_isShared_427_ = v_isSharedCheck_431_;
goto v_resetjp_425_;
}
else
{
lean_inc(v_a_424_);
lean_dec(v___x_421_);
v___x_426_ = lean_box(0);
v_isShared_427_ = v_isSharedCheck_431_;
goto v_resetjp_425_;
}
v_resetjp_425_:
{
lean_object* v___x_429_; 
if (v_isShared_427_ == 0)
{
v___x_429_ = v___x_426_;
goto v_reusejp_428_;
}
else
{
lean_object* v_reuseFailAlloc_430_; 
v_reuseFailAlloc_430_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_430_, 0, v_a_424_);
v___x_429_ = v_reuseFailAlloc_430_;
goto v_reusejp_428_;
}
v_reusejp_428_:
{
return v___x_429_;
}
}
}
}
else
{
lean_dec(v_a_403_);
return v___x_417_;
}
}
}
}
else
{
lean_object* v_a_434_; lean_object* v___x_436_; uint8_t v_isShared_437_; uint8_t v_isSharedCheck_441_; 
lean_dec_ref(v___x_401_);
lean_dec_ref_known(v___x_399_, 2);
lean_dec_ref(v___i_391_);
lean_dec_ref(v_00_u03b1_390_);
v_a_434_ = lean_ctor_get(v___x_402_, 0);
v_isSharedCheck_441_ = !lean_is_exclusive(v___x_402_);
if (v_isSharedCheck_441_ == 0)
{
v___x_436_ = v___x_402_;
v_isShared_437_ = v_isSharedCheck_441_;
goto v_resetjp_435_;
}
else
{
lean_inc(v_a_434_);
lean_dec(v___x_402_);
v___x_436_ = lean_box(0);
v_isShared_437_ = v_isSharedCheck_441_;
goto v_resetjp_435_;
}
v_resetjp_435_:
{
lean_object* v___x_439_; 
if (v_isShared_437_ == 0)
{
v___x_439_ = v___x_436_;
goto v_reusejp_438_;
}
else
{
lean_object* v_reuseFailAlloc_440_; 
v_reuseFailAlloc_440_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_440_, 0, v_a_434_);
v___x_439_ = v_reuseFailAlloc_440_;
goto v_reusejp_438_;
}
v_reusejp_438_:
{
return v___x_439_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing___boxed(lean_object* v_u_442_, lean_object* v_00_u03b1_443_, lean_object* v___i_444_, lean_object* v_a_445_, lean_object* v_a_446_, lean_object* v_a_447_, lean_object* v_a_448_, lean_object* v_a_449_){
_start:
{
lean_object* v_res_450_; 
v_res_450_ = lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing(v_u_442_, v_00_u03b1_443_, v___i_444_, v_a_445_, v_a_446_, v_a_447_, v_a_448_);
lean_dec(v_a_448_);
lean_dec_ref(v_a_447_);
lean_dec(v_a_446_);
lean_dec_ref(v_a_445_);
return v_res_450_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___auto__1(void){
_start:
{
lean_object* v___x_451_; 
v___x_451_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__33, &lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__33_once, _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__33);
return v___x_451_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f(lean_object* v_u_471_, lean_object* v_00_u03b1_472_, lean_object* v___i_473_, lean_object* v_a_474_, lean_object* v_a_475_, lean_object* v_a_476_, lean_object* v_a_477_){
_start:
{
lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v___x_494_; lean_object* v___x_495_; lean_object* v___x_496_; lean_object* v___x_497_; lean_object* v___x_498_; lean_object* v___x_499_; lean_object* v___x_500_; lean_object* v___x_501_; 
v___x_479_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__1));
v___x_480_ = lean_box(0);
v___x_481_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_481_, 0, v_u_471_);
lean_ctor_set(v___x_481_, 1, v___x_480_);
lean_inc_ref_n(v___x_481_, 4);
v___x_482_ = l_Lean_Expr_const___override(v___x_479_, v___x_481_);
lean_inc_ref_n(v_00_u03b1_472_, 4);
v___x_483_ = l_Lean_Expr_app___override(v___x_482_, v_00_u03b1_472_);
v___x_484_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__1));
v___x_485_ = l_Lean_Expr_const___override(v___x_484_, v___x_481_);
v___x_486_ = l_Lean_Expr_app___override(v___x_485_, v_00_u03b1_472_);
v___x_487_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__4));
v___x_488_ = l_Lean_Expr_const___override(v___x_487_, v___x_481_);
v___x_489_ = l_Lean_Expr_app___override(v___x_488_, v_00_u03b1_472_);
v___x_490_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__7));
v___x_491_ = l_Lean_Expr_const___override(v___x_490_, v___x_481_);
v___x_492_ = l_Lean_Expr_app___override(v___x_491_, v_00_u03b1_472_);
v___x_493_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__10));
v___x_494_ = l_Lean_Expr_const___override(v___x_493_, v___x_481_);
v___x_495_ = l_Lean_Expr_app___override(v___x_494_, v_00_u03b1_472_);
v___x_496_ = l_Lean_Expr_app___override(v___x_495_, v___i_473_);
v___x_497_ = l_Lean_Expr_app___override(v___x_492_, v___x_496_);
v___x_498_ = l_Lean_Expr_app___override(v___x_489_, v___x_497_);
v___x_499_ = l_Lean_Expr_app___override(v___x_486_, v___x_498_);
v___x_500_ = l_Lean_Expr_app___override(v___x_483_, v___x_499_);
v___x_501_ = lp_Qq_Qq_trySynthInstanceQ___redArg(v___x_500_, v_a_474_, v_a_475_, v_a_476_, v_a_477_);
if (lean_obj_tag(v___x_501_) == 0)
{
lean_object* v_a_502_; lean_object* v___x_504_; uint8_t v_isShared_505_; uint8_t v_isSharedCheck_510_; 
v_a_502_ = lean_ctor_get(v___x_501_, 0);
v_isSharedCheck_510_ = !lean_is_exclusive(v___x_501_);
if (v_isSharedCheck_510_ == 0)
{
v___x_504_ = v___x_501_;
v_isShared_505_ = v_isSharedCheck_510_;
goto v_resetjp_503_;
}
else
{
lean_inc(v_a_502_);
lean_dec(v___x_501_);
v___x_504_ = lean_box(0);
v_isShared_505_ = v_isSharedCheck_510_;
goto v_resetjp_503_;
}
v_resetjp_503_:
{
lean_object* v___x_506_; lean_object* v___x_508_; 
v___x_506_ = l_Lean_LOption_toOption___redArg(v_a_502_);
if (v_isShared_505_ == 0)
{
lean_ctor_set(v___x_504_, 0, v___x_506_);
v___x_508_ = v___x_504_;
goto v_reusejp_507_;
}
else
{
lean_object* v_reuseFailAlloc_509_; 
v_reuseFailAlloc_509_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_509_, 0, v___x_506_);
v___x_508_ = v_reuseFailAlloc_509_;
goto v_reusejp_507_;
}
v_reusejp_507_:
{
return v___x_508_;
}
}
}
else
{
lean_object* v_a_511_; lean_object* v___x_513_; uint8_t v_isShared_514_; uint8_t v_isSharedCheck_518_; 
v_a_511_ = lean_ctor_get(v___x_501_, 0);
v_isSharedCheck_518_ = !lean_is_exclusive(v___x_501_);
if (v_isSharedCheck_518_ == 0)
{
v___x_513_ = v___x_501_;
v_isShared_514_ = v_isSharedCheck_518_;
goto v_resetjp_512_;
}
else
{
lean_inc(v_a_511_);
lean_dec(v___x_501_);
v___x_513_ = lean_box(0);
v_isShared_514_ = v_isSharedCheck_518_;
goto v_resetjp_512_;
}
v_resetjp_512_:
{
lean_object* v___x_516_; 
if (v_isShared_514_ == 0)
{
v___x_516_ = v___x_513_;
goto v_reusejp_515_;
}
else
{
lean_object* v_reuseFailAlloc_517_; 
v_reuseFailAlloc_517_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_517_, 0, v_a_511_);
v___x_516_ = v_reuseFailAlloc_517_;
goto v_reusejp_515_;
}
v_reusejp_515_:
{
return v___x_516_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___boxed(lean_object* v_u_519_, lean_object* v_00_u03b1_520_, lean_object* v___i_521_, lean_object* v_a_522_, lean_object* v_a_523_, lean_object* v_a_524_, lean_object* v_a_525_, lean_object* v_a_526_){
_start:
{
lean_object* v_res_527_; 
v_res_527_ = lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f(v_u_519_, v_00_u03b1_520_, v___i_521_, v_a_522_, v_a_523_, v_a_524_, v_a_525_);
lean_dec(v_a_525_);
lean_dec_ref(v_a_524_);
lean_dec(v_a_523_);
lean_dec_ref(v_a_522_);
return v_res_527_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing_x3f___auto__1(void){
_start:
{
lean_object* v___x_528_; 
v___x_528_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__33, &lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__33_once, _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1___closed__33);
return v___x_528_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing_x3f(lean_object* v_u_529_, lean_object* v_00_u03b1_530_, lean_object* v___i_531_, lean_object* v_a_532_, lean_object* v_a_533_, lean_object* v_a_534_, lean_object* v_a_535_){
_start:
{
lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_545_; lean_object* v___x_546_; lean_object* v___x_547_; lean_object* v___x_548_; lean_object* v___x_549_; lean_object* v___x_550_; lean_object* v___x_551_; lean_object* v___x_552_; lean_object* v___x_553_; lean_object* v___x_554_; lean_object* v___x_555_; 
v___x_537_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__1));
v___x_538_ = lean_box(0);
v___x_539_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_539_, 0, v_u_529_);
lean_ctor_set(v___x_539_, 1, v___x_538_);
lean_inc_ref_n(v___x_539_, 3);
v___x_540_ = l_Lean_Expr_const___override(v___x_537_, v___x_539_);
lean_inc_ref_n(v_00_u03b1_530_, 3);
v___x_541_ = l_Lean_Expr_app___override(v___x_540_, v_00_u03b1_530_);
v___x_542_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__4));
v___x_543_ = l_Lean_Expr_const___override(v___x_542_, v___x_539_);
v___x_544_ = l_Lean_Expr_app___override(v___x_543_, v_00_u03b1_530_);
v___x_545_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__7));
v___x_546_ = l_Lean_Expr_const___override(v___x_545_, v___x_539_);
v___x_547_ = l_Lean_Expr_app___override(v___x_546_, v_00_u03b1_530_);
v___x_548_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing___closed__2));
v___x_549_ = l_Lean_Expr_const___override(v___x_548_, v___x_539_);
v___x_550_ = l_Lean_Expr_app___override(v___x_549_, v_00_u03b1_530_);
v___x_551_ = l_Lean_Expr_app___override(v___x_550_, v___i_531_);
v___x_552_ = l_Lean_Expr_app___override(v___x_547_, v___x_551_);
v___x_553_ = l_Lean_Expr_app___override(v___x_544_, v___x_552_);
v___x_554_ = l_Lean_Expr_app___override(v___x_541_, v___x_553_);
v___x_555_ = lp_Qq_Qq_trySynthInstanceQ___redArg(v___x_554_, v_a_532_, v_a_533_, v_a_534_, v_a_535_);
if (lean_obj_tag(v___x_555_) == 0)
{
lean_object* v_a_556_; lean_object* v___x_558_; uint8_t v_isShared_559_; uint8_t v_isSharedCheck_564_; 
v_a_556_ = lean_ctor_get(v___x_555_, 0);
v_isSharedCheck_564_ = !lean_is_exclusive(v___x_555_);
if (v_isSharedCheck_564_ == 0)
{
v___x_558_ = v___x_555_;
v_isShared_559_ = v_isSharedCheck_564_;
goto v_resetjp_557_;
}
else
{
lean_inc(v_a_556_);
lean_dec(v___x_555_);
v___x_558_ = lean_box(0);
v_isShared_559_ = v_isSharedCheck_564_;
goto v_resetjp_557_;
}
v_resetjp_557_:
{
lean_object* v___x_560_; lean_object* v___x_562_; 
v___x_560_ = l_Lean_LOption_toOption___redArg(v_a_556_);
if (v_isShared_559_ == 0)
{
lean_ctor_set(v___x_558_, 0, v___x_560_);
v___x_562_ = v___x_558_;
goto v_reusejp_561_;
}
else
{
lean_object* v_reuseFailAlloc_563_; 
v_reuseFailAlloc_563_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_563_, 0, v___x_560_);
v___x_562_ = v_reuseFailAlloc_563_;
goto v_reusejp_561_;
}
v_reusejp_561_:
{
return v___x_562_;
}
}
}
else
{
lean_object* v_a_565_; lean_object* v___x_567_; uint8_t v_isShared_568_; uint8_t v_isSharedCheck_572_; 
v_a_565_ = lean_ctor_get(v___x_555_, 0);
v_isSharedCheck_572_ = !lean_is_exclusive(v___x_555_);
if (v_isSharedCheck_572_ == 0)
{
v___x_567_ = v___x_555_;
v_isShared_568_ = v_isSharedCheck_572_;
goto v_resetjp_566_;
}
else
{
lean_inc(v_a_565_);
lean_dec(v___x_555_);
v___x_567_ = lean_box(0);
v_isShared_568_ = v_isSharedCheck_572_;
goto v_resetjp_566_;
}
v_resetjp_566_:
{
lean_object* v___x_570_; 
if (v_isShared_568_ == 0)
{
v___x_570_ = v___x_567_;
goto v_reusejp_569_;
}
else
{
lean_object* v_reuseFailAlloc_571_; 
v_reuseFailAlloc_571_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_571_, 0, v_a_565_);
v___x_570_ = v_reuseFailAlloc_571_;
goto v_reusejp_569_;
}
v_reusejp_569_:
{
return v___x_570_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing_x3f___boxed(lean_object* v_u_573_, lean_object* v_00_u03b1_574_, lean_object* v___i_575_, lean_object* v_a_576_, lean_object* v_a_577_, lean_object* v_a_578_, lean_object* v_a_579_, lean_object* v_a_580_){
_start:
{
lean_object* v_res_581_; 
v_res_581_ = lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing_x3f(v_u_573_, v_00_u03b1_574_, v___i_575_, v_a_576_, v_a_577_, v_a_578_, v_a_579_);
lean_dec(v_a_579_);
lean_dec_ref(v_a_578_);
lean_dec(v_a_577_);
lean_dec_ref(v_a_576_);
return v_res_581_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1(void){
_start:
{
lean_object* v___x_583_; lean_object* v___x_584_; 
v___x_583_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__0));
v___x_584_ = l_Lean_stringToMessageData(v___x_583_);
return v___x_584_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__5(void){
_start:
{
lean_object* v___x_589_; lean_object* v___x_590_; lean_object* v___x_591_; 
v___x_589_ = lean_box(0);
v___x_590_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__4));
v___x_591_ = l_Lean_Expr_const___override(v___x_590_, v___x_589_);
return v___x_591_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__8(void){
_start:
{
lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v___x_598_; 
v___x_596_ = lean_box(0);
v___x_597_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__7));
v___x_598_ = l_Lean_Expr_const___override(v___x_597_, v___x_596_);
return v___x_598_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__11(void){
_start:
{
lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v___x_604_; 
v___x_602_ = lean_box(0);
v___x_603_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__10));
v___x_604_ = l_Lean_Expr_const___override(v___x_603_, v___x_602_);
return v___x_604_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__15(void){
_start:
{
lean_object* v___x_611_; lean_object* v___x_612_; lean_object* v___x_613_; 
v___x_611_ = lean_box(0);
v___x_612_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__14));
v___x_613_ = l_Lean_Expr_const___override(v___x_612_, v___x_611_);
return v___x_613_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__21(void){
_start:
{
lean_object* v___x_625_; lean_object* v___x_626_; lean_object* v___x_627_; 
v___x_625_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__20));
v___x_626_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__18));
v___x_627_ = l_Lean_Expr_const___override(v___x_626_, v___x_625_);
return v___x_627_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__22(void){
_start:
{
lean_object* v___x_628_; lean_object* v___x_629_; lean_object* v___x_630_; 
v___x_628_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__15, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__15_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__15);
v___x_629_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__21, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__21_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__21);
v___x_630_ = l_Lean_Expr_app___override(v___x_629_, v___x_628_);
return v___x_630_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__23(void){
_start:
{
lean_object* v___x_631_; lean_object* v___x_632_; lean_object* v___x_633_; 
v___x_631_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__15, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__15_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__15);
v___x_632_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__22, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__22_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__22);
v___x_633_ = l_Lean_Expr_app___override(v___x_632_, v___x_631_);
return v___x_633_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__24(void){
_start:
{
lean_object* v___x_634_; lean_object* v___x_635_; lean_object* v___x_636_; 
v___x_634_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__15, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__15_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__15);
v___x_635_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__23, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__23_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__23);
v___x_636_ = l_Lean_Expr_app___override(v___x_635_, v___x_634_);
return v___x_636_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__27(void){
_start:
{
lean_object* v___x_640_; lean_object* v___x_641_; lean_object* v___x_642_; 
v___x_640_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__12));
v___x_641_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__26));
v___x_642_ = l_Lean_Expr_const___override(v___x_641_, v___x_640_);
return v___x_642_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__28(void){
_start:
{
lean_object* v___x_643_; lean_object* v___x_644_; lean_object* v___x_645_; 
v___x_643_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__15, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__15_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__15);
v___x_644_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__27, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__27_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__27);
v___x_645_ = l_Lean_Expr_app___override(v___x_644_, v___x_643_);
return v___x_645_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__31(void){
_start:
{
lean_object* v___x_650_; lean_object* v___x_651_; lean_object* v___x_652_; 
v___x_650_ = lean_box(0);
v___x_651_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__30));
v___x_652_ = l_Lean_Expr_const___override(v___x_651_, v___x_650_);
return v___x_652_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__32(void){
_start:
{
lean_object* v___x_653_; lean_object* v___x_654_; lean_object* v___x_655_; 
v___x_653_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__31, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__31_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__31);
v___x_654_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__28, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__28_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__28);
v___x_655_ = l_Lean_Expr_app___override(v___x_654_, v___x_653_);
return v___x_655_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__33(void){
_start:
{
lean_object* v___x_656_; lean_object* v___x_657_; lean_object* v___x_658_; 
v___x_656_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__32, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__32_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__32);
v___x_657_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__24, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__24_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__24);
v___x_658_ = l_Lean_Expr_app___override(v___x_657_, v___x_656_);
return v___x_658_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__36(void){
_start:
{
lean_object* v___x_663_; lean_object* v___x_664_; lean_object* v___x_665_; 
v___x_663_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__12));
v___x_664_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__35));
v___x_665_ = l_Lean_Expr_const___override(v___x_664_, v___x_663_);
return v___x_665_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__37(void){
_start:
{
lean_object* v___x_666_; lean_object* v___x_667_; lean_object* v___x_668_; 
v___x_666_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__15, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__15_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__15);
v___x_667_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__36, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__36_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__36);
v___x_668_ = l_Lean_Expr_app___override(v___x_667_, v___x_666_);
return v___x_668_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__40(void){
_start:
{
lean_object* v___x_673_; lean_object* v___x_674_; lean_object* v___x_675_; 
v___x_673_ = lean_box(0);
v___x_674_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__39));
v___x_675_ = l_Lean_Expr_const___override(v___x_674_, v___x_673_);
return v___x_675_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__41(void){
_start:
{
lean_object* v___x_676_; lean_object* v___x_677_; lean_object* v___x_678_; 
v___x_676_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__40, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__40_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__40);
v___x_677_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__37, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__37_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__37);
v___x_678_ = l_Lean_Expr_app___override(v___x_677_, v___x_676_);
return v___x_678_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__43(void){
_start:
{
lean_object* v___x_682_; lean_object* v___x_683_; lean_object* v___x_684_; 
v___x_682_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__12));
v___x_683_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__42));
v___x_684_ = l_Lean_Expr_const___override(v___x_683_, v___x_682_);
return v___x_684_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__44(void){
_start:
{
lean_object* v___x_685_; lean_object* v___x_686_; lean_object* v___x_687_; 
v___x_685_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__15, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__15_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__15);
v___x_686_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__43, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__43_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__43);
v___x_687_ = l_Lean_Expr_app___override(v___x_686_, v___x_685_);
return v___x_687_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__47(void){
_start:
{
lean_object* v___x_692_; lean_object* v___x_693_; lean_object* v___x_694_; 
v___x_692_ = lean_box(0);
v___x_693_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__46));
v___x_694_ = l_Lean_Expr_const___override(v___x_693_, v___x_692_);
return v___x_694_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__48(void){
_start:
{
lean_object* v___x_695_; lean_object* v___x_696_; lean_object* v___x_697_; 
v___x_695_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__47, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__47_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__47);
v___x_696_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__44, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__44_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__44);
v___x_697_ = l_Lean_Expr_app___override(v___x_696_, v___x_695_);
return v___x_697_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__51(void){
_start:
{
lean_object* v___x_702_; lean_object* v___x_703_; lean_object* v___x_704_; 
v___x_702_ = lean_box(0);
v___x_703_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__50));
v___x_704_ = l_Lean_Expr_const___override(v___x_703_, v___x_702_);
return v___x_704_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__57(void){
_start:
{
lean_object* v___x_714_; lean_object* v___x_715_; lean_object* v___x_716_; 
v___x_714_ = lean_box(0);
v___x_715_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__56));
v___x_716_ = l_Lean_Expr_const___override(v___x_715_, v___x_714_);
return v___x_716_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0(lean_object* v_u_717_, lean_object* v_00_u03b1_718_, lean_object* v_e_719_, lean_object* v___y_720_, lean_object* v___y_721_, lean_object* v___y_722_, lean_object* v___y_723_){
_start:
{
lean_object* v___x_725_; 
lean_inc_ref(v_e_719_);
v___x_725_ = l_Lean_Meta_whnfR(v_e_719_, v___y_720_, v___y_721_, v___y_722_, v___y_723_);
if (lean_obj_tag(v___x_725_) == 0)
{
lean_object* v_a_726_; lean_object* v___y_728_; lean_object* v___y_729_; lean_object* v___y_730_; lean_object* v___y_731_; 
v_a_726_ = lean_ctor_get(v___x_725_, 0);
lean_inc(v_a_726_);
lean_dec_ref_known(v___x_725_, 1);
if (lean_obj_tag(v_a_726_) == 5)
{
lean_object* v_fn_734_; 
v_fn_734_ = lean_ctor_get(v_a_726_, 0);
lean_inc_ref(v_fn_734_);
if (lean_obj_tag(v_fn_734_) == 5)
{
lean_object* v_fn_735_; 
v_fn_735_ = lean_ctor_get(v_fn_734_, 0);
if (lean_obj_tag(v_fn_735_) == 4)
{
lean_object* v_declName_736_; 
v_declName_736_ = lean_ctor_get(v_fn_735_, 0);
lean_inc(v_declName_736_);
if (lean_obj_tag(v_declName_736_) == 1)
{
lean_object* v_pre_737_; 
v_pre_737_ = lean_ctor_get(v_declName_736_, 0);
if (lean_obj_tag(v_pre_737_) == 0)
{
lean_object* v_arg_738_; lean_object* v_arg_739_; lean_object* v_str_740_; lean_object* v___x_741_; uint8_t v___x_742_; 
v_arg_738_ = lean_ctor_get(v_a_726_, 1);
lean_inc_ref(v_arg_738_);
lean_dec_ref_known(v_a_726_, 2);
v_arg_739_ = lean_ctor_get(v_fn_734_, 1);
lean_inc_ref(v_arg_739_);
lean_dec_ref_known(v_fn_734_, 2);
v_str_740_ = lean_ctor_get(v_declName_736_, 1);
lean_inc_ref(v_str_740_);
lean_dec_ref_known(v_declName_736_, 2);
v___x_741_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__2));
v___x_742_ = lean_string_dec_eq(v_str_740_, v___x_741_);
lean_dec_ref(v_str_740_);
if (v___x_742_ == 0)
{
lean_dec_ref(v_arg_739_);
lean_dec_ref(v_arg_738_);
lean_dec_ref(v_e_719_);
v___y_728_ = v___y_720_;
v___y_729_ = v___y_721_;
v___y_730_ = v___y_722_;
v___y_731_ = v___y_723_;
goto v___jp_727_;
}
else
{
lean_object* v___x_743_; lean_object* v___x_744_; uint8_t v___x_745_; lean_object* v___x_746_; 
v___x_743_ = lean_box(0);
v___x_744_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__5, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__5_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__5);
v___x_745_ = 0;
lean_inc_ref(v_arg_739_);
v___x_746_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v___x_743_, v___x_744_, v_arg_739_, v___x_745_, v___y_720_, v___y_721_, v___y_722_, v___y_723_);
if (lean_obj_tag(v___x_746_) == 0)
{
lean_object* v_a_747_; lean_object* v___x_748_; lean_object* v___x_749_; 
v_a_747_ = lean_ctor_get(v___x_746_, 0);
lean_inc(v_a_747_);
lean_dec_ref_known(v___x_746_, 1);
v___x_748_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__8, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__8_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__8);
lean_inc_ref(v_arg_739_);
v___x_749_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toInt(v___x_743_, v___x_744_, v_arg_739_, v___x_748_, v_a_747_);
if (lean_obj_tag(v___x_749_) == 1)
{
lean_object* v_val_750_; lean_object* v_snd_751_; lean_object* v_fst_752_; lean_object* v_snd_753_; lean_object* v___x_754_; lean_object* v___x_755_; 
v_val_750_ = lean_ctor_get(v___x_749_, 0);
lean_inc(v_val_750_);
lean_dec_ref_known(v___x_749_, 1);
v_snd_751_ = lean_ctor_get(v_val_750_, 1);
lean_inc(v_snd_751_);
lean_dec(v_val_750_);
v_fst_752_ = lean_ctor_get(v_snd_751_, 0);
lean_inc(v_fst_752_);
v_snd_753_ = lean_ctor_get(v_snd_751_, 1);
lean_inc(v_snd_753_);
lean_dec(v_snd_751_);
v___x_754_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__11, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__11_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__11);
lean_inc_ref(v_arg_738_);
v___x_755_ = lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg(v___x_743_, v___x_754_, v_arg_738_, v___y_720_, v___y_721_, v___y_722_, v___y_723_);
if (lean_obj_tag(v___x_755_) == 0)
{
lean_object* v_a_756_; lean_object* v_fst_757_; lean_object* v_snd_758_; lean_object* v___x_759_; lean_object* v___x_760_; lean_object* v___x_761_; lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v___x_764_; lean_object* v___x_765_; lean_object* v___x_766_; lean_object* v___x_767_; 
v_a_756_ = lean_ctor_get(v___x_755_, 0);
lean_inc(v_a_756_);
lean_dec_ref_known(v___x_755_, 1);
v_fst_757_ = lean_ctor_get(v_a_756_, 0);
lean_inc_n(v_fst_757_, 2);
v_snd_758_ = lean_ctor_get(v_a_756_, 1);
lean_inc(v_snd_758_);
lean_dec(v_a_756_);
v___x_759_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__15, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__15_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__15);
v___x_760_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__33, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__33_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__33);
v___x_761_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__41, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__41_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__41);
lean_inc(v_fst_752_);
v___x_762_ = l_Lean_Expr_app___override(v___x_761_, v_fst_752_);
v___x_763_ = l_Lean_Expr_app___override(v___x_760_, v___x_762_);
v___x_764_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__48, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__48_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__48);
v___x_765_ = l_Lean_Expr_app___override(v___x_764_, v_fst_757_);
v___x_766_ = l_Lean_Expr_app___override(v___x_763_, v___x_765_);
lean_inc_ref(v___x_766_);
v___x_767_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v___x_743_, v___x_759_, v___x_766_, v___x_745_, v___y_720_, v___y_721_, v___y_722_, v___y_723_);
if (lean_obj_tag(v___x_767_) == 0)
{
lean_object* v_a_768_; lean_object* v___x_770_; uint8_t v_isShared_771_; uint8_t v_isSharedCheck_807_; 
v_a_768_ = lean_ctor_get(v___x_767_, 0);
v_isSharedCheck_807_ = !lean_is_exclusive(v___x_767_);
if (v_isSharedCheck_807_ == 0)
{
v___x_770_ = v___x_767_;
v_isShared_771_ = v_isSharedCheck_807_;
goto v_resetjp_769_;
}
else
{
lean_inc(v_a_768_);
lean_dec(v___x_767_);
v___x_770_ = lean_box(0);
v_isShared_771_ = v_isSharedCheck_807_;
goto v_resetjp_769_;
}
v_resetjp_769_:
{
lean_object* v___x_772_; lean_object* v_a_774_; lean_object* v___x_795_; 
v___x_772_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__51, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__51_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__51);
v___x_795_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27(v___x_743_, v___x_759_, v___x_766_, v___x_772_, v_a_768_);
if (lean_obj_tag(v___x_795_) == 0)
{
lean_object* v___x_796_; lean_object* v___x_797_; lean_object* v_a_798_; lean_object* v___x_800_; uint8_t v_isShared_801_; uint8_t v_isSharedCheck_805_; 
lean_del_object(v___x_770_);
lean_dec(v_snd_758_);
lean_dec(v_fst_757_);
lean_dec(v_snd_753_);
lean_dec(v_fst_752_);
lean_dec_ref(v_arg_739_);
lean_dec_ref(v_arg_738_);
lean_dec_ref(v_e_719_);
v___x_796_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1);
v___x_797_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0___redArg(v___x_796_, v___y_720_, v___y_721_, v___y_722_, v___y_723_);
v_a_798_ = lean_ctor_get(v___x_797_, 0);
v_isSharedCheck_805_ = !lean_is_exclusive(v___x_797_);
if (v_isSharedCheck_805_ == 0)
{
v___x_800_ = v___x_797_;
v_isShared_801_ = v_isSharedCheck_805_;
goto v_resetjp_799_;
}
else
{
lean_inc(v_a_798_);
lean_dec(v___x_797_);
v___x_800_ = lean_box(0);
v_isShared_801_ = v_isSharedCheck_805_;
goto v_resetjp_799_;
}
v_resetjp_799_:
{
lean_object* v___x_803_; 
if (v_isShared_801_ == 0)
{
v___x_803_ = v___x_800_;
goto v_reusejp_802_;
}
else
{
lean_object* v_reuseFailAlloc_804_; 
v_reuseFailAlloc_804_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_804_, 0, v_a_798_);
v___x_803_ = v_reuseFailAlloc_804_;
goto v_reusejp_802_;
}
v_reusejp_802_:
{
return v___x_803_;
}
}
}
else
{
lean_object* v_val_806_; 
v_val_806_ = lean_ctor_get(v___x_795_, 0);
lean_inc(v_val_806_);
lean_dec_ref_known(v___x_795_, 1);
v_a_774_ = v_val_806_;
goto v___jp_773_;
}
v___jp_773_:
{
lean_object* v_snd_775_; lean_object* v_snd_776_; lean_object* v_fst_777_; lean_object* v_fst_778_; lean_object* v_fst_779_; lean_object* v_snd_780_; lean_object* v___x_781_; lean_object* v___x_782_; lean_object* v___x_783_; lean_object* v___x_784_; lean_object* v___x_785_; lean_object* v___x_786_; lean_object* v___x_787_; lean_object* v___x_788_; lean_object* v___x_789_; lean_object* v___x_790_; lean_object* v___x_791_; lean_object* v___x_793_; 
v_snd_775_ = lean_ctor_get(v_a_774_, 1);
lean_inc(v_snd_775_);
v_snd_776_ = lean_ctor_get(v_snd_775_, 1);
lean_inc(v_snd_776_);
v_fst_777_ = lean_ctor_get(v_a_774_, 0);
lean_inc(v_fst_777_);
lean_dec_ref(v_a_774_);
v_fst_778_ = lean_ctor_get(v_snd_775_, 0);
lean_inc_n(v_fst_778_, 2);
lean_dec(v_snd_775_);
v_fst_779_ = lean_ctor_get(v_snd_776_, 0);
lean_inc_n(v_fst_779_, 2);
v_snd_780_ = lean_ctor_get(v_snd_776_, 1);
lean_inc(v_snd_780_);
lean_dec(v_snd_776_);
v___x_781_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__57, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__57_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__57);
v___x_782_ = l_Lean_Expr_app___override(v___x_781_, v_arg_739_);
v___x_783_ = l_Lean_Expr_app___override(v___x_782_, v_fst_752_);
v___x_784_ = l_Lean_Expr_app___override(v___x_783_, v_fst_778_);
v___x_785_ = l_Lean_Expr_app___override(v___x_784_, v_arg_738_);
v___x_786_ = l_Lean_Expr_app___override(v___x_785_, v_fst_757_);
v___x_787_ = l_Lean_Expr_app___override(v___x_786_, v_fst_779_);
v___x_788_ = l_Lean_Expr_app___override(v___x_787_, v_snd_753_);
v___x_789_ = l_Lean_Expr_app___override(v___x_788_, v_snd_758_);
v___x_790_ = l_Lean_Expr_app___override(v___x_789_, v_snd_780_);
v___x_791_ = lp_mathlib_Mathlib_Meta_NormNum_Result_isRat(v___x_743_, v___x_759_, v_e_719_, v___x_772_, v_fst_777_, v_fst_778_, v_fst_779_, v___x_790_);
if (v_isShared_771_ == 0)
{
lean_ctor_set(v___x_770_, 0, v___x_791_);
v___x_793_ = v___x_770_;
goto v_reusejp_792_;
}
else
{
lean_object* v_reuseFailAlloc_794_; 
v_reuseFailAlloc_794_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_794_, 0, v___x_791_);
v___x_793_ = v_reuseFailAlloc_794_;
goto v_reusejp_792_;
}
v_reusejp_792_:
{
return v___x_793_;
}
}
}
}
else
{
lean_dec_ref(v___x_766_);
lean_dec(v_snd_758_);
lean_dec(v_fst_757_);
lean_dec(v_snd_753_);
lean_dec(v_fst_752_);
lean_dec_ref(v_arg_739_);
lean_dec_ref(v_arg_738_);
lean_dec_ref(v_e_719_);
return v___x_767_;
}
}
else
{
lean_object* v_a_808_; lean_object* v___x_810_; uint8_t v_isShared_811_; uint8_t v_isSharedCheck_815_; 
lean_dec(v_snd_753_);
lean_dec(v_fst_752_);
lean_dec_ref(v_arg_739_);
lean_dec_ref(v_arg_738_);
lean_dec_ref(v_e_719_);
v_a_808_ = lean_ctor_get(v___x_755_, 0);
v_isSharedCheck_815_ = !lean_is_exclusive(v___x_755_);
if (v_isSharedCheck_815_ == 0)
{
v___x_810_ = v___x_755_;
v_isShared_811_ = v_isSharedCheck_815_;
goto v_resetjp_809_;
}
else
{
lean_inc(v_a_808_);
lean_dec(v___x_755_);
v___x_810_ = lean_box(0);
v_isShared_811_ = v_isSharedCheck_815_;
goto v_resetjp_809_;
}
v_resetjp_809_:
{
lean_object* v___x_813_; 
if (v_isShared_811_ == 0)
{
v___x_813_ = v___x_810_;
goto v_reusejp_812_;
}
else
{
lean_object* v_reuseFailAlloc_814_; 
v_reuseFailAlloc_814_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_814_, 0, v_a_808_);
v___x_813_ = v_reuseFailAlloc_814_;
goto v_reusejp_812_;
}
v_reusejp_812_:
{
return v___x_813_;
}
}
}
}
else
{
lean_object* v___x_816_; lean_object* v___x_817_; 
lean_dec(v___x_749_);
lean_dec_ref(v_arg_739_);
lean_dec_ref(v_arg_738_);
lean_dec_ref(v_e_719_);
v___x_816_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1);
v___x_817_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0___redArg(v___x_816_, v___y_720_, v___y_721_, v___y_722_, v___y_723_);
return v___x_817_;
}
}
else
{
lean_dec_ref(v_arg_739_);
lean_dec_ref(v_arg_738_);
lean_dec_ref(v_e_719_);
return v___x_746_;
}
}
}
else
{
lean_dec_ref_known(v_declName_736_, 2);
lean_dec_ref_known(v_fn_734_, 2);
lean_dec_ref_known(v_a_726_, 2);
lean_dec_ref(v_e_719_);
v___y_728_ = v___y_720_;
v___y_729_ = v___y_721_;
v___y_730_ = v___y_722_;
v___y_731_ = v___y_723_;
goto v___jp_727_;
}
}
else
{
lean_dec(v_declName_736_);
lean_dec_ref_known(v_fn_734_, 2);
lean_dec_ref_known(v_a_726_, 2);
lean_dec_ref(v_e_719_);
v___y_728_ = v___y_720_;
v___y_729_ = v___y_721_;
v___y_730_ = v___y_722_;
v___y_731_ = v___y_723_;
goto v___jp_727_;
}
}
else
{
lean_dec_ref_known(v_fn_734_, 2);
lean_dec_ref_known(v_a_726_, 2);
lean_dec_ref(v_e_719_);
v___y_728_ = v___y_720_;
v___y_729_ = v___y_721_;
v___y_730_ = v___y_722_;
v___y_731_ = v___y_723_;
goto v___jp_727_;
}
}
else
{
lean_dec_ref(v_fn_734_);
lean_dec_ref_known(v_a_726_, 2);
lean_dec_ref(v_e_719_);
v___y_728_ = v___y_720_;
v___y_729_ = v___y_721_;
v___y_730_ = v___y_722_;
v___y_731_ = v___y_723_;
goto v___jp_727_;
}
}
else
{
lean_dec(v_a_726_);
lean_dec_ref(v_e_719_);
v___y_728_ = v___y_720_;
v___y_729_ = v___y_721_;
v___y_730_ = v___y_722_;
v___y_731_ = v___y_723_;
goto v___jp_727_;
}
v___jp_727_:
{
lean_object* v___x_732_; lean_object* v___x_733_; 
v___x_732_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1);
v___x_733_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0___redArg(v___x_732_, v___y_728_, v___y_729_, v___y_730_, v___y_731_);
return v___x_733_;
}
}
else
{
lean_object* v_a_818_; lean_object* v___x_820_; uint8_t v_isShared_821_; uint8_t v_isSharedCheck_825_; 
lean_dec_ref(v_e_719_);
v_a_818_ = lean_ctor_get(v___x_725_, 0);
v_isSharedCheck_825_ = !lean_is_exclusive(v___x_725_);
if (v_isSharedCheck_825_ == 0)
{
v___x_820_ = v___x_725_;
v_isShared_821_ = v_isSharedCheck_825_;
goto v_resetjp_819_;
}
else
{
lean_inc(v_a_818_);
lean_dec(v___x_725_);
v___x_820_ = lean_box(0);
v_isShared_821_ = v_isSharedCheck_825_;
goto v_resetjp_819_;
}
v_resetjp_819_:
{
lean_object* v___x_823_; 
if (v_isShared_821_ == 0)
{
v___x_823_ = v___x_820_;
goto v_reusejp_822_;
}
else
{
lean_object* v_reuseFailAlloc_824_; 
v_reuseFailAlloc_824_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_824_, 0, v_a_818_);
v___x_823_ = v_reuseFailAlloc_824_;
goto v_reusejp_822_;
}
v_reusejp_822_:
{
return v___x_823_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___boxed(lean_object* v_u_826_, lean_object* v_00_u03b1_827_, lean_object* v_e_828_, lean_object* v___y_829_, lean_object* v___y_830_, lean_object* v___y_831_, lean_object* v___y_832_, lean_object* v___y_833_){
_start:
{
lean_object* v_res_834_; 
v_res_834_ = lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0(v_u_826_, v_00_u03b1_827_, v_e_828_, v___y_829_, v___y_830_, v___y_831_, v___y_832_);
lean_dec(v___y_832_);
lean_dec_ref(v___y_831_);
lean_dec(v___y_830_);
lean_dec_ref(v___y_829_);
lean_dec_ref(v_00_u03b1_827_);
lean_dec(v_u_826_);
return v_res_834_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__2(void){
_start:
{
lean_object* v___x_849_; lean_object* v___x_850_; lean_object* v___x_851_; 
v___x_849_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__12));
v___x_850_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__1));
v___x_851_ = l_Lean_Expr_const___override(v___x_850_, v___x_849_);
return v___x_851_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5(void){
_start:
{
lean_object* v___x_855_; lean_object* v___x_856_; lean_object* v___x_857_; 
v___x_855_ = lean_box(0);
v___x_856_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__4));
v___x_857_ = l_Lean_Expr_const___override(v___x_856_, v___x_855_);
return v___x_857_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__6(void){
_start:
{
lean_object* v___x_858_; lean_object* v___x_859_; lean_object* v___x_860_; 
v___x_858_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5);
v___x_859_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__21, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__21_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__21);
v___x_860_ = l_Lean_Expr_app___override(v___x_859_, v___x_858_);
return v___x_860_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__7(void){
_start:
{
lean_object* v___x_861_; lean_object* v___x_862_; lean_object* v___x_863_; 
v___x_861_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5);
v___x_862_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__6, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__6_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__6);
v___x_863_ = l_Lean_Expr_app___override(v___x_862_, v___x_861_);
return v___x_863_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__8(void){
_start:
{
lean_object* v___x_864_; lean_object* v___x_865_; lean_object* v___x_866_; 
v___x_864_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5);
v___x_865_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__7, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__7_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__7);
v___x_866_ = l_Lean_Expr_app___override(v___x_865_, v___x_864_);
return v___x_866_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__9(void){
_start:
{
lean_object* v___x_867_; lean_object* v___x_868_; lean_object* v___x_869_; 
v___x_867_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5);
v___x_868_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__27, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__27_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__27);
v___x_869_ = l_Lean_Expr_app___override(v___x_868_, v___x_867_);
return v___x_869_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__11(void){
_start:
{
lean_object* v___x_873_; lean_object* v___x_874_; lean_object* v___x_875_; 
v___x_873_ = lean_box(0);
v___x_874_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__10));
v___x_875_ = l_Lean_Expr_const___override(v___x_874_, v___x_873_);
return v___x_875_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__12(void){
_start:
{
lean_object* v___x_876_; lean_object* v___x_877_; lean_object* v___x_878_; 
v___x_876_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__11, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__11_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__11);
v___x_877_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__9, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__9_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__9);
v___x_878_ = l_Lean_Expr_app___override(v___x_877_, v___x_876_);
return v___x_878_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__13(void){
_start:
{
lean_object* v___x_879_; lean_object* v___x_880_; lean_object* v___x_881_; 
v___x_879_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__12, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__12_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__12);
v___x_880_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__8, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__8_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__8);
v___x_881_ = l_Lean_Expr_app___override(v___x_880_, v___x_879_);
return v___x_881_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__14(void){
_start:
{
lean_object* v___x_882_; lean_object* v___x_883_; lean_object* v___x_884_; 
v___x_882_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5);
v___x_883_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__43, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__43_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__43);
v___x_884_ = l_Lean_Expr_app___override(v___x_883_, v___x_882_);
return v___x_884_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__17(void){
_start:
{
lean_object* v___x_889_; lean_object* v___x_890_; lean_object* v___x_891_; 
v___x_889_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__12));
v___x_890_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__16));
v___x_891_ = l_Lean_Expr_const___override(v___x_890_, v___x_889_);
return v___x_891_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__18(void){
_start:
{
lean_object* v___x_892_; lean_object* v___x_893_; lean_object* v___x_894_; 
v___x_892_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5);
v___x_893_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__17, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__17_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__17);
v___x_894_ = l_Lean_Expr_app___override(v___x_893_, v___x_892_);
return v___x_894_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__19(void){
_start:
{
lean_object* v___x_895_; lean_object* v___x_896_; lean_object* v___x_897_; 
v___x_895_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5);
v___x_896_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__2, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__2_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__2);
v___x_897_ = l_Lean_Expr_app___override(v___x_896_, v___x_895_);
return v___x_897_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__20(void){
_start:
{
lean_object* v___x_898_; lean_object* v___x_899_; lean_object* v___x_900_; 
v___x_898_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__12));
v___x_899_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__4));
v___x_900_ = l_Lean_Expr_const___override(v___x_899_, v___x_898_);
return v___x_900_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__21(void){
_start:
{
lean_object* v___x_901_; lean_object* v___x_902_; lean_object* v___x_903_; 
v___x_901_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5);
v___x_902_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__20, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__20_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__20);
v___x_903_ = l_Lean_Expr_app___override(v___x_902_, v___x_901_);
return v___x_903_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__22(void){
_start:
{
lean_object* v___x_904_; lean_object* v___x_905_; lean_object* v___x_906_; 
v___x_904_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__12));
v___x_905_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__7));
v___x_906_ = l_Lean_Expr_const___override(v___x_905_, v___x_904_);
return v___x_906_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__23(void){
_start:
{
lean_object* v___x_907_; lean_object* v___x_908_; lean_object* v___x_909_; 
v___x_907_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5);
v___x_908_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__22, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__22_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__22);
v___x_909_ = l_Lean_Expr_app___override(v___x_908_, v___x_907_);
return v___x_909_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__26(void){
_start:
{
lean_object* v___x_914_; lean_object* v___x_915_; lean_object* v___x_916_; 
v___x_914_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__12));
v___x_915_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__25));
v___x_916_ = l_Lean_Expr_const___override(v___x_915_, v___x_914_);
return v___x_916_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__27(void){
_start:
{
lean_object* v___x_917_; lean_object* v___x_918_; lean_object* v___x_919_; 
v___x_917_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5);
v___x_918_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__26, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__26_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__26);
v___x_919_ = l_Lean_Expr_app___override(v___x_918_, v___x_917_);
return v___x_919_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__30(void){
_start:
{
lean_object* v___x_923_; lean_object* v___x_924_; lean_object* v___x_925_; 
v___x_923_ = lean_box(0);
v___x_924_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__29));
v___x_925_ = l_Lean_Expr_const___override(v___x_924_, v___x_923_);
return v___x_925_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__31(void){
_start:
{
lean_object* v___x_926_; lean_object* v___x_927_; lean_object* v___x_928_; 
v___x_926_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__30, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__30_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__30);
v___x_927_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__27, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__27_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__27);
v___x_928_ = l_Lean_Expr_app___override(v___x_927_, v___x_926_);
return v___x_928_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__32(void){
_start:
{
lean_object* v___x_929_; lean_object* v___x_930_; lean_object* v___x_931_; 
v___x_929_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__31, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__31_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__31);
v___x_930_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__23, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__23_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__23);
v___x_931_ = l_Lean_Expr_app___override(v___x_930_, v___x_929_);
return v___x_931_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__33(void){
_start:
{
lean_object* v___x_932_; lean_object* v___x_933_; lean_object* v___x_934_; 
v___x_932_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__32, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__32_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__32);
v___x_933_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__21, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__21_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__21);
v___x_934_ = l_Lean_Expr_app___override(v___x_933_, v___x_932_);
return v___x_934_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__34(void){
_start:
{
lean_object* v___x_935_; lean_object* v___x_936_; lean_object* v___x_937_; 
v___x_935_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__33, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__33_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__33);
v___x_936_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__19, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__19_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__19);
v___x_937_ = l_Lean_Expr_app___override(v___x_936_, v___x_935_);
return v___x_937_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__35(void){
_start:
{
lean_object* v___x_938_; lean_object* v___x_939_; lean_object* v___x_940_; 
v___x_938_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__34, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__34_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__34);
v___x_939_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__18, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__18_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__18);
v___x_940_ = l_Lean_Expr_app___override(v___x_939_, v___x_938_);
return v___x_940_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__36(void){
_start:
{
lean_object* v___x_941_; lean_object* v___x_942_; lean_object* v___x_943_; 
v___x_941_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__35, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__35_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__35);
v___x_942_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__14, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__14_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__14);
v___x_943_ = l_Lean_Expr_app___override(v___x_942_, v___x_941_);
return v___x_943_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__40(void){
_start:
{
lean_object* v___x_949_; lean_object* v___x_950_; lean_object* v___x_951_; 
v___x_949_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__12));
v___x_950_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__39));
v___x_951_ = l_Lean_Expr_const___override(v___x_950_, v___x_949_);
return v___x_951_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__41(void){
_start:
{
lean_object* v___x_952_; lean_object* v___x_953_; lean_object* v___x_954_; 
v___x_952_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5);
v___x_953_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__40, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__40_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__40);
v___x_954_ = l_Lean_Expr_app___override(v___x_953_, v___x_952_);
return v___x_954_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__44(void){
_start:
{
lean_object* v___x_959_; lean_object* v___x_960_; lean_object* v___x_961_; 
v___x_959_ = lean_box(0);
v___x_960_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__43));
v___x_961_ = l_Lean_Expr_const___override(v___x_960_, v___x_959_);
return v___x_961_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__45(void){
_start:
{
lean_object* v___x_962_; lean_object* v___x_963_; lean_object* v___x_964_; 
v___x_962_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__44, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__44_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__44);
v___x_963_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__41, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__41_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__41);
v___x_964_ = l_Lean_Expr_app___override(v___x_963_, v___x_962_);
return v___x_964_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__48(void){
_start:
{
lean_object* v___x_971_; lean_object* v___x_972_; lean_object* v___x_973_; 
v___x_971_ = lean_box(0);
v___x_972_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__47));
v___x_973_ = l_Lean_Expr_const___override(v___x_972_, v___x_971_);
return v___x_973_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0(lean_object* v_u_974_, lean_object* v_00_u03b1_975_, lean_object* v_e_976_, lean_object* v___y_977_, lean_object* v___y_978_, lean_object* v___y_979_, lean_object* v___y_980_){
_start:
{
lean_object* v___x_982_; 
v___x_982_ = l_Lean_Meta_whnfR(v_e_976_, v___y_977_, v___y_978_, v___y_979_, v___y_980_);
if (lean_obj_tag(v___x_982_) == 0)
{
lean_object* v_a_983_; lean_object* v___y_985_; lean_object* v___y_986_; lean_object* v___y_987_; lean_object* v___y_988_; 
v_a_983_ = lean_ctor_get(v___x_982_, 0);
lean_inc(v_a_983_);
lean_dec_ref_known(v___x_982_, 1);
if (lean_obj_tag(v_a_983_) == 5)
{
lean_object* v_fn_991_; 
v_fn_991_ = lean_ctor_get(v_a_983_, 0);
lean_inc_ref(v_fn_991_);
if (lean_obj_tag(v_fn_991_) == 5)
{
lean_object* v_fn_992_; 
v_fn_992_ = lean_ctor_get(v_fn_991_, 0);
if (lean_obj_tag(v_fn_992_) == 4)
{
lean_object* v_declName_993_; 
v_declName_993_ = lean_ctor_get(v_fn_992_, 0);
lean_inc(v_declName_993_);
if (lean_obj_tag(v_declName_993_) == 1)
{
lean_object* v_pre_994_; 
v_pre_994_ = lean_ctor_get(v_declName_993_, 0);
lean_inc(v_pre_994_);
if (lean_obj_tag(v_pre_994_) == 1)
{
lean_object* v_pre_995_; 
v_pre_995_ = lean_ctor_get(v_pre_994_, 0);
if (lean_obj_tag(v_pre_995_) == 0)
{
lean_object* v_arg_996_; lean_object* v_arg_997_; lean_object* v_str_998_; lean_object* v_str_999_; lean_object* v___x_1000_; uint8_t v___x_1001_; 
v_arg_996_ = lean_ctor_get(v_a_983_, 1);
lean_inc_ref(v_arg_996_);
lean_dec_ref_known(v_a_983_, 2);
v_arg_997_ = lean_ctor_get(v_fn_991_, 1);
lean_inc_ref(v_arg_997_);
lean_dec_ref_known(v_fn_991_, 2);
v_str_998_ = lean_ctor_get(v_declName_993_, 1);
lean_inc_ref(v_str_998_);
lean_dec_ref_known(v_declName_993_, 2);
v_str_999_ = lean_ctor_get(v_pre_994_, 1);
lean_inc_ref(v_str_999_);
lean_dec_ref_known(v_pre_994_, 2);
v___x_1000_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__0));
v___x_1001_ = lean_string_dec_eq(v_str_999_, v___x_1000_);
lean_dec_ref(v_str_999_);
if (v___x_1001_ == 0)
{
lean_dec_ref(v_str_998_);
lean_dec_ref(v_arg_997_);
lean_dec_ref(v_arg_996_);
v___y_985_ = v___y_977_;
v___y_986_ = v___y_978_;
v___y_987_ = v___y_979_;
v___y_988_ = v___y_980_;
goto v___jp_984_;
}
else
{
lean_object* v___x_1002_; uint8_t v___x_1003_; 
v___x_1002_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__1));
v___x_1003_ = lean_string_dec_eq(v_str_998_, v___x_1002_);
lean_dec_ref(v_str_998_);
if (v___x_1003_ == 0)
{
lean_dec_ref(v_arg_997_);
lean_dec_ref(v_arg_996_);
v___y_985_ = v___y_977_;
v___y_986_ = v___y_978_;
v___y_987_ = v___y_979_;
v___y_988_ = v___y_980_;
goto v___jp_984_;
}
else
{
lean_object* v___x_1004_; lean_object* v___x_1005_; uint8_t v___x_1006_; lean_object* v___x_1007_; 
v___x_1004_ = lean_box(0);
v___x_1005_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__11, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__11_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__11);
v___x_1006_ = 0;
lean_inc_ref(v_arg_997_);
v___x_1007_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v___x_1004_, v___x_1005_, v_arg_997_, v___x_1006_, v___y_977_, v___y_978_, v___y_979_, v___y_980_);
if (lean_obj_tag(v___x_1007_) == 0)
{
lean_object* v___x_1008_; 
lean_dec_ref_known(v___x_1007_, 1);
lean_inc_ref(v_arg_997_);
v___x_1008_ = lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg(v___x_1004_, v___x_1005_, v_arg_997_, v___y_977_, v___y_978_, v___y_979_, v___y_980_);
if (lean_obj_tag(v___x_1008_) == 0)
{
lean_object* v_a_1009_; lean_object* v_fst_1010_; lean_object* v_snd_1011_; lean_object* v___x_1012_; 
v_a_1009_ = lean_ctor_get(v___x_1008_, 0);
lean_inc(v_a_1009_);
lean_dec_ref_known(v___x_1008_, 1);
v_fst_1010_ = lean_ctor_get(v_a_1009_, 0);
lean_inc(v_fst_1010_);
v_snd_1011_ = lean_ctor_get(v_a_1009_, 1);
lean_inc(v_snd_1011_);
lean_dec(v_a_1009_);
lean_inc_ref(v_arg_996_);
v___x_1012_ = lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg(v___x_1004_, v___x_1005_, v_arg_996_, v___y_977_, v___y_978_, v___y_979_, v___y_980_);
if (lean_obj_tag(v___x_1012_) == 0)
{
lean_object* v_a_1013_; lean_object* v_fst_1014_; lean_object* v_snd_1015_; lean_object* v___x_1016_; lean_object* v___x_1017_; lean_object* v___x_1018_; lean_object* v___x_1019_; lean_object* v___x_1020_; lean_object* v___x_1021_; lean_object* v___x_1022_; lean_object* v___x_1023_; 
v_a_1013_ = lean_ctor_get(v___x_1012_, 0);
lean_inc(v_a_1013_);
lean_dec_ref_known(v___x_1012_, 1);
v_fst_1014_ = lean_ctor_get(v_a_1013_, 0);
lean_inc_n(v_fst_1014_, 2);
v_snd_1015_ = lean_ctor_get(v_a_1013_, 1);
lean_inc(v_snd_1015_);
lean_dec(v_a_1013_);
v___x_1016_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5);
v___x_1017_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__13, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__13_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__13);
v___x_1018_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__36, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__36_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__36);
lean_inc(v_fst_1010_);
v___x_1019_ = l_Lean_Expr_app___override(v___x_1018_, v_fst_1010_);
v___x_1020_ = l_Lean_Expr_app___override(v___x_1017_, v___x_1019_);
v___x_1021_ = l_Lean_Expr_app___override(v___x_1018_, v_fst_1014_);
v___x_1022_ = l_Lean_Expr_app___override(v___x_1020_, v___x_1021_);
lean_inc_ref(v___x_1022_);
v___x_1023_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v___x_1004_, v___x_1016_, v___x_1022_, v___x_1006_, v___y_977_, v___y_978_, v___y_979_, v___y_980_);
if (lean_obj_tag(v___x_1023_) == 0)
{
lean_object* v_a_1024_; lean_object* v___x_1026_; uint8_t v_isShared_1027_; uint8_t v_isSharedCheck_1053_; 
v_a_1024_ = lean_ctor_get(v___x_1023_, 0);
v_isSharedCheck_1053_ = !lean_is_exclusive(v___x_1023_);
if (v_isSharedCheck_1053_ == 0)
{
v___x_1026_ = v___x_1023_;
v_isShared_1027_ = v_isSharedCheck_1053_;
goto v_resetjp_1025_;
}
else
{
lean_inc(v_a_1024_);
lean_dec(v___x_1023_);
v___x_1026_ = lean_box(0);
v_isShared_1027_ = v_isSharedCheck_1053_;
goto v_resetjp_1025_;
}
v_resetjp_1025_:
{
lean_object* v___x_1028_; lean_object* v___x_1029_; 
v___x_1028_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__45, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__45_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__45);
v___x_1029_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toNNRat_x27(v___x_1004_, v___x_1016_, v___x_1022_, v___x_1028_, v_a_1024_);
if (lean_obj_tag(v___x_1029_) == 1)
{
lean_object* v_val_1030_; lean_object* v_snd_1031_; lean_object* v_snd_1032_; lean_object* v_fst_1033_; lean_object* v_fst_1034_; lean_object* v_fst_1035_; lean_object* v_snd_1036_; lean_object* v___x_1037_; lean_object* v___x_1038_; lean_object* v___x_1039_; lean_object* v___x_1040_; lean_object* v___x_1041_; lean_object* v___x_1042_; lean_object* v___x_1043_; lean_object* v___x_1044_; lean_object* v___x_1045_; lean_object* v___x_1046_; lean_object* v___x_1047_; lean_object* v___x_1049_; 
v_val_1030_ = lean_ctor_get(v___x_1029_, 0);
lean_inc(v_val_1030_);
lean_dec_ref_known(v___x_1029_, 1);
v_snd_1031_ = lean_ctor_get(v_val_1030_, 1);
lean_inc(v_snd_1031_);
v_snd_1032_ = lean_ctor_get(v_snd_1031_, 1);
lean_inc(v_snd_1032_);
v_fst_1033_ = lean_ctor_get(v_val_1030_, 0);
lean_inc(v_fst_1033_);
lean_dec(v_val_1030_);
v_fst_1034_ = lean_ctor_get(v_snd_1031_, 0);
lean_inc_n(v_fst_1034_, 2);
lean_dec(v_snd_1031_);
v_fst_1035_ = lean_ctor_get(v_snd_1032_, 0);
lean_inc_n(v_fst_1035_, 2);
v_snd_1036_ = lean_ctor_get(v_snd_1032_, 1);
lean_inc(v_snd_1036_);
lean_dec(v_snd_1032_);
v___x_1037_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__48, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__48_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__48);
v___x_1038_ = l_Lean_Expr_app___override(v___x_1037_, v_arg_997_);
v___x_1039_ = l_Lean_Expr_app___override(v___x_1038_, v_fst_1010_);
v___x_1040_ = l_Lean_Expr_app___override(v___x_1039_, v_fst_1034_);
v___x_1041_ = l_Lean_Expr_app___override(v___x_1040_, v_arg_996_);
v___x_1042_ = l_Lean_Expr_app___override(v___x_1041_, v_fst_1014_);
v___x_1043_ = l_Lean_Expr_app___override(v___x_1042_, v_fst_1035_);
v___x_1044_ = l_Lean_Expr_app___override(v___x_1043_, v_snd_1011_);
v___x_1045_ = l_Lean_Expr_app___override(v___x_1044_, v_snd_1015_);
v___x_1046_ = l_Lean_Expr_app___override(v___x_1045_, v_snd_1036_);
v___x_1047_ = lean_alloc_ctor(3, 5, 0);
lean_ctor_set(v___x_1047_, 0, v___x_1028_);
lean_ctor_set(v___x_1047_, 1, v_fst_1033_);
lean_ctor_set(v___x_1047_, 2, v_fst_1034_);
lean_ctor_set(v___x_1047_, 3, v_fst_1035_);
lean_ctor_set(v___x_1047_, 4, v___x_1046_);
if (v_isShared_1027_ == 0)
{
lean_ctor_set(v___x_1026_, 0, v___x_1047_);
v___x_1049_ = v___x_1026_;
goto v_reusejp_1048_;
}
else
{
lean_object* v_reuseFailAlloc_1050_; 
v_reuseFailAlloc_1050_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1050_, 0, v___x_1047_);
v___x_1049_ = v_reuseFailAlloc_1050_;
goto v_reusejp_1048_;
}
v_reusejp_1048_:
{
return v___x_1049_;
}
}
else
{
lean_object* v___x_1051_; lean_object* v___x_1052_; 
lean_dec(v___x_1029_);
lean_del_object(v___x_1026_);
lean_dec(v_snd_1015_);
lean_dec(v_fst_1014_);
lean_dec(v_snd_1011_);
lean_dec(v_fst_1010_);
lean_dec_ref(v_arg_997_);
lean_dec_ref(v_arg_996_);
v___x_1051_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1);
v___x_1052_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0___redArg(v___x_1051_, v___y_977_, v___y_978_, v___y_979_, v___y_980_);
return v___x_1052_;
}
}
}
else
{
lean_dec_ref(v___x_1022_);
lean_dec(v_snd_1015_);
lean_dec(v_fst_1014_);
lean_dec(v_snd_1011_);
lean_dec(v_fst_1010_);
lean_dec_ref(v_arg_997_);
lean_dec_ref(v_arg_996_);
return v___x_1023_;
}
}
else
{
lean_object* v_a_1054_; lean_object* v___x_1056_; uint8_t v_isShared_1057_; uint8_t v_isSharedCheck_1061_; 
lean_dec(v_snd_1011_);
lean_dec(v_fst_1010_);
lean_dec_ref(v_arg_997_);
lean_dec_ref(v_arg_996_);
v_a_1054_ = lean_ctor_get(v___x_1012_, 0);
v_isSharedCheck_1061_ = !lean_is_exclusive(v___x_1012_);
if (v_isSharedCheck_1061_ == 0)
{
v___x_1056_ = v___x_1012_;
v_isShared_1057_ = v_isSharedCheck_1061_;
goto v_resetjp_1055_;
}
else
{
lean_inc(v_a_1054_);
lean_dec(v___x_1012_);
v___x_1056_ = lean_box(0);
v_isShared_1057_ = v_isSharedCheck_1061_;
goto v_resetjp_1055_;
}
v_resetjp_1055_:
{
lean_object* v___x_1059_; 
if (v_isShared_1057_ == 0)
{
v___x_1059_ = v___x_1056_;
goto v_reusejp_1058_;
}
else
{
lean_object* v_reuseFailAlloc_1060_; 
v_reuseFailAlloc_1060_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1060_, 0, v_a_1054_);
v___x_1059_ = v_reuseFailAlloc_1060_;
goto v_reusejp_1058_;
}
v_reusejp_1058_:
{
return v___x_1059_;
}
}
}
}
else
{
lean_object* v_a_1062_; lean_object* v___x_1064_; uint8_t v_isShared_1065_; uint8_t v_isSharedCheck_1069_; 
lean_dec_ref(v_arg_997_);
lean_dec_ref(v_arg_996_);
v_a_1062_ = lean_ctor_get(v___x_1008_, 0);
v_isSharedCheck_1069_ = !lean_is_exclusive(v___x_1008_);
if (v_isSharedCheck_1069_ == 0)
{
v___x_1064_ = v___x_1008_;
v_isShared_1065_ = v_isSharedCheck_1069_;
goto v_resetjp_1063_;
}
else
{
lean_inc(v_a_1062_);
lean_dec(v___x_1008_);
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
lean_dec_ref(v_arg_997_);
lean_dec_ref(v_arg_996_);
return v___x_1007_;
}
}
}
}
else
{
lean_dec_ref_known(v_pre_994_, 2);
lean_dec_ref_known(v_declName_993_, 2);
lean_dec_ref_known(v_fn_991_, 2);
lean_dec_ref_known(v_a_983_, 2);
v___y_985_ = v___y_977_;
v___y_986_ = v___y_978_;
v___y_987_ = v___y_979_;
v___y_988_ = v___y_980_;
goto v___jp_984_;
}
}
else
{
lean_dec_ref_known(v_declName_993_, 2);
lean_dec(v_pre_994_);
lean_dec_ref_known(v_fn_991_, 2);
lean_dec_ref_known(v_a_983_, 2);
v___y_985_ = v___y_977_;
v___y_986_ = v___y_978_;
v___y_987_ = v___y_979_;
v___y_988_ = v___y_980_;
goto v___jp_984_;
}
}
else
{
lean_dec(v_declName_993_);
lean_dec_ref_known(v_fn_991_, 2);
lean_dec_ref_known(v_a_983_, 2);
v___y_985_ = v___y_977_;
v___y_986_ = v___y_978_;
v___y_987_ = v___y_979_;
v___y_988_ = v___y_980_;
goto v___jp_984_;
}
}
else
{
lean_dec_ref_known(v_fn_991_, 2);
lean_dec_ref_known(v_a_983_, 2);
v___y_985_ = v___y_977_;
v___y_986_ = v___y_978_;
v___y_987_ = v___y_979_;
v___y_988_ = v___y_980_;
goto v___jp_984_;
}
}
else
{
lean_dec_ref(v_fn_991_);
lean_dec_ref_known(v_a_983_, 2);
v___y_985_ = v___y_977_;
v___y_986_ = v___y_978_;
v___y_987_ = v___y_979_;
v___y_988_ = v___y_980_;
goto v___jp_984_;
}
}
else
{
lean_dec(v_a_983_);
v___y_985_ = v___y_977_;
v___y_986_ = v___y_978_;
v___y_987_ = v___y_979_;
v___y_988_ = v___y_980_;
goto v___jp_984_;
}
v___jp_984_:
{
lean_object* v___x_989_; lean_object* v___x_990_; 
v___x_989_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1);
v___x_990_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0___redArg(v___x_989_, v___y_985_, v___y_986_, v___y_987_, v___y_988_);
return v___x_990_;
}
}
else
{
lean_object* v_a_1070_; lean_object* v___x_1072_; uint8_t v_isShared_1073_; uint8_t v_isSharedCheck_1077_; 
v_a_1070_ = lean_ctor_get(v___x_982_, 0);
v_isSharedCheck_1077_ = !lean_is_exclusive(v___x_982_);
if (v_isSharedCheck_1077_ == 0)
{
v___x_1072_ = v___x_982_;
v_isShared_1073_ = v_isSharedCheck_1077_;
goto v_resetjp_1071_;
}
else
{
lean_inc(v_a_1070_);
lean_dec(v___x_982_);
v___x_1072_ = lean_box(0);
v_isShared_1073_ = v_isSharedCheck_1077_;
goto v_resetjp_1071_;
}
v_resetjp_1071_:
{
lean_object* v___x_1075_; 
if (v_isShared_1073_ == 0)
{
v___x_1075_ = v___x_1072_;
goto v_reusejp_1074_;
}
else
{
lean_object* v_reuseFailAlloc_1076_; 
v_reuseFailAlloc_1076_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1076_, 0, v_a_1070_);
v___x_1075_ = v_reuseFailAlloc_1076_;
goto v_reusejp_1074_;
}
v_reusejp_1074_:
{
return v___x_1075_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___boxed(lean_object* v_u_1078_, lean_object* v_00_u03b1_1079_, lean_object* v_e_1080_, lean_object* v___y_1081_, lean_object* v___y_1082_, lean_object* v___y_1083_, lean_object* v___y_1084_, lean_object* v___y_1085_){
_start:
{
lean_object* v_res_1086_; 
v_res_1086_ = lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0(v_u_1078_, v_00_u03b1_1079_, v_e_1080_, v___y_1081_, v___y_1082_, v___y_1083_, v___y_1084_);
lean_dec(v___y_1084_);
lean_dec_ref(v___y_1083_);
lean_dec(v___y_1082_);
lean_dec_ref(v___y_1081_);
lean_dec_ref(v_00_u03b1_1079_);
lean_dec(v_u_1078_);
return v_res_1086_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalRatCast_spec__0___redArg(lean_object* v_k_1099_, uint8_t v_allowLevelAssignments_1100_, lean_object* v___y_1101_, lean_object* v___y_1102_, lean_object* v___y_1103_, lean_object* v___y_1104_){
_start:
{
lean_object* v___x_1106_; 
v___x_1106_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), v_allowLevelAssignments_1100_, v_k_1099_, v___y_1101_, v___y_1102_, v___y_1103_, v___y_1104_);
if (lean_obj_tag(v___x_1106_) == 0)
{
lean_object* v_a_1107_; lean_object* v___x_1109_; uint8_t v_isShared_1110_; uint8_t v_isSharedCheck_1114_; 
v_a_1107_ = lean_ctor_get(v___x_1106_, 0);
v_isSharedCheck_1114_ = !lean_is_exclusive(v___x_1106_);
if (v_isSharedCheck_1114_ == 0)
{
v___x_1109_ = v___x_1106_;
v_isShared_1110_ = v_isSharedCheck_1114_;
goto v_resetjp_1108_;
}
else
{
lean_inc(v_a_1107_);
lean_dec(v___x_1106_);
v___x_1109_ = lean_box(0);
v_isShared_1110_ = v_isSharedCheck_1114_;
goto v_resetjp_1108_;
}
v_resetjp_1108_:
{
lean_object* v___x_1112_; 
if (v_isShared_1110_ == 0)
{
v___x_1112_ = v___x_1109_;
goto v_reusejp_1111_;
}
else
{
lean_object* v_reuseFailAlloc_1113_; 
v_reuseFailAlloc_1113_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1113_, 0, v_a_1107_);
v___x_1112_ = v_reuseFailAlloc_1113_;
goto v_reusejp_1111_;
}
v_reusejp_1111_:
{
return v___x_1112_;
}
}
}
else
{
lean_object* v_a_1115_; lean_object* v___x_1117_; uint8_t v_isShared_1118_; uint8_t v_isSharedCheck_1122_; 
v_a_1115_ = lean_ctor_get(v___x_1106_, 0);
v_isSharedCheck_1122_ = !lean_is_exclusive(v___x_1106_);
if (v_isSharedCheck_1122_ == 0)
{
v___x_1117_ = v___x_1106_;
v_isShared_1118_ = v_isSharedCheck_1122_;
goto v_resetjp_1116_;
}
else
{
lean_inc(v_a_1115_);
lean_dec(v___x_1106_);
v___x_1117_ = lean_box(0);
v_isShared_1118_ = v_isSharedCheck_1122_;
goto v_resetjp_1116_;
}
v_resetjp_1116_:
{
lean_object* v___x_1120_; 
if (v_isShared_1118_ == 0)
{
v___x_1120_ = v___x_1117_;
goto v_reusejp_1119_;
}
else
{
lean_object* v_reuseFailAlloc_1121_; 
v_reuseFailAlloc_1121_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1121_, 0, v_a_1115_);
v___x_1120_ = v_reuseFailAlloc_1121_;
goto v_reusejp_1119_;
}
v_reusejp_1119_:
{
return v___x_1120_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalRatCast_spec__0___redArg___boxed(lean_object* v_k_1123_, lean_object* v_allowLevelAssignments_1124_, lean_object* v___y_1125_, lean_object* v___y_1126_, lean_object* v___y_1127_, lean_object* v___y_1128_, lean_object* v___y_1129_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_1130_; lean_object* v_res_1131_; 
v_allowLevelAssignments_boxed_1130_ = lean_unbox(v_allowLevelAssignments_1124_);
v_res_1131_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalRatCast_spec__0___redArg(v_k_1123_, v_allowLevelAssignments_boxed_1130_, v___y_1125_, v___y_1126_, v___y_1127_, v___y_1128_);
lean_dec(v___y_1128_);
lean_dec_ref(v___y_1127_);
lean_dec(v___y_1126_);
lean_dec_ref(v___y_1125_);
return v_res_1131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalRatCast_spec__0(lean_object* v_00_u03b1_1132_, lean_object* v_k_1133_, uint8_t v_allowLevelAssignments_1134_, lean_object* v___y_1135_, lean_object* v___y_1136_, lean_object* v___y_1137_, lean_object* v___y_1138_){
_start:
{
lean_object* v___x_1140_; 
v___x_1140_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalRatCast_spec__0___redArg(v_k_1133_, v_allowLevelAssignments_1134_, v___y_1135_, v___y_1136_, v___y_1137_, v___y_1138_);
return v___x_1140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalRatCast_spec__0___boxed(lean_object* v_00_u03b1_1141_, lean_object* v_k_1142_, lean_object* v_allowLevelAssignments_1143_, lean_object* v___y_1144_, lean_object* v___y_1145_, lean_object* v___y_1146_, lean_object* v___y_1147_, lean_object* v___y_1148_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_1149_; lean_object* v_res_1150_; 
v_allowLevelAssignments_boxed_1149_ = lean_unbox(v_allowLevelAssignments_1143_);
v_res_1150_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalRatCast_spec__0(v_00_u03b1_1141_, v_k_1142_, v_allowLevelAssignments_boxed_1149_, v___y_1144_, v___y_1145_, v___y_1146_, v___y_1147_);
lean_dec(v___y_1147_);
lean_dec_ref(v___y_1146_);
lean_dec(v___y_1145_);
lean_dec_ref(v___y_1144_);
return v_res_1150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__0(lean_object* v_fn_1151_, lean_object* v___x_1152_, lean_object* v___y_1153_, lean_object* v___y_1154_, lean_object* v___y_1155_, lean_object* v___y_1156_){
_start:
{
lean_object* v___x_1158_; 
v___x_1158_ = l_Lean_Meta_isExprDefEq(v_fn_1151_, v___x_1152_, v___y_1153_, v___y_1154_, v___y_1155_, v___y_1156_);
return v___x_1158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__0___boxed(lean_object* v_fn_1159_, lean_object* v___x_1160_, lean_object* v___y_1161_, lean_object* v___y_1162_, lean_object* v___y_1163_, lean_object* v___y_1164_, lean_object* v___y_1165_){
_start:
{
lean_object* v_res_1166_; 
v_res_1166_ = lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__0(v_fn_1159_, v___x_1160_, v___y_1161_, v___y_1162_, v___y_1163_, v___y_1164_);
lean_dec(v___y_1164_);
lean_dec_ref(v___y_1163_);
lean_dec(v___y_1162_);
lean_dec_ref(v___y_1161_);
return v_res_1166_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__7(void){
_start:
{
lean_object* v___x_1184_; lean_object* v___x_1185_; lean_object* v___x_1186_; 
v___x_1184_ = lean_box(0);
v___x_1185_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__6));
v___x_1186_ = l_Lean_Expr_const___override(v___x_1185_, v___x_1184_);
return v___x_1186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1(lean_object* v_u_1213_, lean_object* v_00_u03b1_1214_, lean_object* v_e_1215_, lean_object* v___y_1216_, lean_object* v___y_1217_, lean_object* v___y_1218_, lean_object* v___y_1219_){
_start:
{
lean_object* v___x_1221_; 
lean_inc_ref(v_00_u03b1_1214_);
lean_inc(v_u_1213_);
v___x_1221_ = lp_mathlib_Mathlib_Meta_NormNum_inferDivisionRing(v_u_1213_, v_00_u03b1_1214_, v___y_1216_, v___y_1217_, v___y_1218_, v___y_1219_);
if (lean_obj_tag(v___x_1221_) == 0)
{
lean_object* v_a_1222_; lean_object* v___x_1223_; 
v_a_1222_ = lean_ctor_get(v___x_1221_, 0);
lean_inc(v_a_1222_);
lean_dec_ref_known(v___x_1221_, 1);
v___x_1223_ = l_Lean_Meta_whnfR(v_e_1215_, v___y_1216_, v___y_1217_, v___y_1218_, v___y_1219_);
if (lean_obj_tag(v___x_1223_) == 0)
{
lean_object* v_a_1224_; 
v_a_1224_ = lean_ctor_get(v___x_1223_, 0);
lean_inc(v_a_1224_);
lean_dec_ref_known(v___x_1223_, 1);
if (lean_obj_tag(v_a_1224_) == 5)
{
lean_object* v_fn_1225_; lean_object* v_arg_1226_; lean_object* v___x_1227_; lean_object* v___x_1228_; lean_object* v___x_1229_; lean_object* v___x_1230_; lean_object* v___x_1231_; lean_object* v___x_1232_; lean_object* v___x_1233_; lean_object* v___x_1234_; lean_object* v___x_1235_; lean_object* v___x_1236_; lean_object* v___x_1237_; lean_object* v___x_1238_; lean_object* v___f_1239_; uint8_t v___x_1240_; lean_object* v___x_1399_; 
v_fn_1225_ = lean_ctor_get(v_a_1224_, 0);
lean_inc_ref(v_fn_1225_);
v_arg_1226_ = lean_ctor_get(v_a_1224_, 1);
lean_inc_ref(v_arg_1226_);
lean_dec_ref_known(v_a_1224_, 2);
lean_inc_n(v_u_1213_, 2);
v___x_1227_ = l_Lean_Level_succ___override(v_u_1213_);
v___x_1228_ = lean_box(0);
v___x_1229_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__15, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__15_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__15);
v___x_1230_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__0));
v___x_1231_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1231_, 0, v_u_1213_);
lean_ctor_set(v___x_1231_, 1, v___x_1228_);
lean_inc_ref_n(v___x_1231_, 2);
v___x_1232_ = l_Lean_Expr_const___override(v___x_1230_, v___x_1231_);
lean_inc_ref_n(v_00_u03b1_1214_, 2);
v___x_1233_ = l_Lean_Expr_app___override(v___x_1232_, v_00_u03b1_1214_);
v___x_1234_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__2));
v___x_1235_ = l_Lean_Expr_const___override(v___x_1234_, v___x_1231_);
v___x_1236_ = l_Lean_Expr_app___override(v___x_1235_, v_00_u03b1_1214_);
lean_inc(v_a_1222_);
v___x_1237_ = l_Lean_Expr_app___override(v___x_1236_, v_a_1222_);
v___x_1238_ = l_Lean_Expr_app___override(v___x_1233_, v___x_1237_);
v___f_1239_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__0___boxed), 7, 2);
lean_closure_set(v___f_1239_, 0, v_fn_1225_);
lean_closure_set(v___f_1239_, 1, v___x_1238_);
v___x_1240_ = 0;
v___x_1399_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalRatCast_spec__0___redArg(v___f_1239_, v___x_1240_, v___y_1216_, v___y_1217_, v___y_1218_, v___y_1219_);
if (lean_obj_tag(v___x_1399_) == 0)
{
lean_object* v_a_1400_; uint8_t v___x_1401_; 
v_a_1400_ = lean_ctor_get(v___x_1399_, 0);
lean_inc(v_a_1400_);
lean_dec_ref_known(v___x_1399_, 1);
v___x_1401_ = lean_unbox(v_a_1400_);
lean_dec(v_a_1400_);
if (v___x_1401_ == 0)
{
lean_object* v___x_1402_; lean_object* v___x_1403_; lean_object* v_a_1404_; lean_object* v___x_1406_; uint8_t v_isShared_1407_; uint8_t v_isSharedCheck_1411_; 
lean_dec_ref_known(v___x_1231_, 2);
lean_dec(v___x_1227_);
lean_dec_ref(v_arg_1226_);
lean_dec(v_a_1222_);
lean_dec_ref(v_00_u03b1_1214_);
lean_dec(v_u_1213_);
v___x_1402_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1);
v___x_1403_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0___redArg(v___x_1402_, v___y_1216_, v___y_1217_, v___y_1218_, v___y_1219_);
v_a_1404_ = lean_ctor_get(v___x_1403_, 0);
v_isSharedCheck_1411_ = !lean_is_exclusive(v___x_1403_);
if (v_isSharedCheck_1411_ == 0)
{
v___x_1406_ = v___x_1403_;
v_isShared_1407_ = v_isSharedCheck_1411_;
goto v_resetjp_1405_;
}
else
{
lean_inc(v_a_1404_);
lean_dec(v___x_1403_);
v___x_1406_ = lean_box(0);
v_isShared_1407_ = v_isSharedCheck_1411_;
goto v_resetjp_1405_;
}
v_resetjp_1405_:
{
lean_object* v___x_1409_; 
if (v_isShared_1407_ == 0)
{
v___x_1409_ = v___x_1406_;
goto v_reusejp_1408_;
}
else
{
lean_object* v_reuseFailAlloc_1410_; 
v_reuseFailAlloc_1410_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1410_, 0, v_a_1404_);
v___x_1409_ = v_reuseFailAlloc_1410_;
goto v_reusejp_1408_;
}
v_reusejp_1408_:
{
return v___x_1409_;
}
}
}
else
{
goto v___jp_1241_;
}
}
else
{
lean_object* v_a_1412_; lean_object* v___x_1414_; uint8_t v_isShared_1415_; uint8_t v_isSharedCheck_1419_; 
lean_dec_ref_known(v___x_1231_, 2);
lean_dec(v___x_1227_);
lean_dec_ref(v_arg_1226_);
lean_dec(v_a_1222_);
lean_dec_ref(v_00_u03b1_1214_);
lean_dec(v_u_1213_);
v_a_1412_ = lean_ctor_get(v___x_1399_, 0);
v_isSharedCheck_1419_ = !lean_is_exclusive(v___x_1399_);
if (v_isSharedCheck_1419_ == 0)
{
v___x_1414_ = v___x_1399_;
v_isShared_1415_ = v_isSharedCheck_1419_;
goto v_resetjp_1413_;
}
else
{
lean_inc(v_a_1412_);
lean_dec(v___x_1399_);
v___x_1414_ = lean_box(0);
v_isShared_1415_ = v_isSharedCheck_1419_;
goto v_resetjp_1413_;
}
v_resetjp_1413_:
{
lean_object* v___x_1417_; 
if (v_isShared_1415_ == 0)
{
v___x_1417_ = v___x_1414_;
goto v_reusejp_1416_;
}
else
{
lean_object* v_reuseFailAlloc_1418_; 
v_reuseFailAlloc_1418_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1418_, 0, v_a_1412_);
v___x_1417_ = v_reuseFailAlloc_1418_;
goto v_reusejp_1416_;
}
v_reusejp_1416_:
{
return v___x_1417_;
}
}
}
v___jp_1241_:
{
lean_object* v___x_1242_; lean_object* v___x_1243_; 
v___x_1242_ = lean_box(0);
lean_inc_ref(v_arg_1226_);
v___x_1243_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v___x_1242_, v___x_1229_, v_arg_1226_, v___x_1240_, v___y_1216_, v___y_1217_, v___y_1218_, v___y_1219_);
if (lean_obj_tag(v___x_1243_) == 0)
{
lean_object* v_a_1244_; lean_object* v___x_1246_; uint8_t v_isShared_1247_; uint8_t v_isSharedCheck_1398_; 
v_a_1244_ = lean_ctor_get(v___x_1243_, 0);
v_isSharedCheck_1398_ = !lean_is_exclusive(v___x_1243_);
if (v_isSharedCheck_1398_ == 0)
{
v___x_1246_ = v___x_1243_;
v_isShared_1247_ = v_isSharedCheck_1398_;
goto v_resetjp_1245_;
}
else
{
lean_inc(v_a_1244_);
lean_dec(v___x_1243_);
v___x_1246_ = lean_box(0);
v_isShared_1247_ = v_isSharedCheck_1398_;
goto v_resetjp_1245_;
}
v_resetjp_1245_:
{
switch(lean_obj_tag(v_a_1244_))
{
case 1:
{
lean_object* v_lit_1248_; lean_object* v_proof_1249_; lean_object* v___x_1251_; uint8_t v_isShared_1252_; uint8_t v_isSharedCheck_1278_; 
lean_dec(v___x_1227_);
lean_dec(v_u_1213_);
v_lit_1248_ = lean_ctor_get(v_a_1244_, 1);
v_proof_1249_ = lean_ctor_get(v_a_1244_, 2);
v_isSharedCheck_1278_ = !lean_is_exclusive(v_a_1244_);
if (v_isSharedCheck_1278_ == 0)
{
lean_object* v_unused_1279_; 
v_unused_1279_ = lean_ctor_get(v_a_1244_, 0);
lean_dec(v_unused_1279_);
v___x_1251_ = v_a_1244_;
v_isShared_1252_ = v_isSharedCheck_1278_;
goto v_resetjp_1250_;
}
else
{
lean_inc(v_proof_1249_);
lean_inc(v_lit_1248_);
lean_dec(v_a_1244_);
v___x_1251_ = lean_box(0);
v_isShared_1252_ = v_isSharedCheck_1278_;
goto v_resetjp_1250_;
}
v_resetjp_1250_:
{
lean_object* v___x_1253_; lean_object* v___x_1254_; lean_object* v___x_1255_; lean_object* v___x_1256_; lean_object* v___x_1257_; lean_object* v___x_1258_; lean_object* v___x_1259_; lean_object* v___x_1260_; lean_object* v___x_1261_; lean_object* v___x_1262_; lean_object* v___x_1263_; lean_object* v___x_1264_; lean_object* v___x_1265_; lean_object* v___x_1266_; lean_object* v___x_1267_; lean_object* v___x_1268_; lean_object* v___x_1269_; lean_object* v___x_1270_; lean_object* v___x_1271_; lean_object* v___x_1273_; 
v___x_1253_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__4));
lean_inc_ref_n(v___x_1231_, 3);
v___x_1254_ = l_Lean_Expr_const___override(v___x_1253_, v___x_1231_);
lean_inc_ref_n(v_00_u03b1_1214_, 3);
v___x_1255_ = l_Lean_Expr_app___override(v___x_1254_, v_00_u03b1_1214_);
v___x_1256_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___closed__7));
v___x_1257_ = l_Lean_Expr_const___override(v___x_1256_, v___x_1231_);
v___x_1258_ = l_Lean_Expr_app___override(v___x_1257_, v_00_u03b1_1214_);
v___x_1259_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing___closed__2));
v___x_1260_ = l_Lean_Expr_const___override(v___x_1259_, v___x_1231_);
v___x_1261_ = l_Lean_Expr_app___override(v___x_1260_, v_00_u03b1_1214_);
lean_inc(v_a_1222_);
v___x_1262_ = l_Lean_Expr_app___override(v___x_1261_, v_a_1222_);
v___x_1263_ = l_Lean_Expr_app___override(v___x_1258_, v___x_1262_);
v___x_1264_ = l_Lean_Expr_app___override(v___x_1255_, v___x_1263_);
v___x_1265_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__4));
v___x_1266_ = l_Lean_Expr_const___override(v___x_1265_, v___x_1231_);
v___x_1267_ = l_Lean_Expr_app___override(v___x_1266_, v_00_u03b1_1214_);
v___x_1268_ = l_Lean_Expr_app___override(v___x_1267_, v_a_1222_);
v___x_1269_ = l_Lean_Expr_app___override(v___x_1268_, v_arg_1226_);
lean_inc_ref(v_lit_1248_);
v___x_1270_ = l_Lean_Expr_app___override(v___x_1269_, v_lit_1248_);
v___x_1271_ = l_Lean_Expr_app___override(v___x_1270_, v_proof_1249_);
if (v_isShared_1252_ == 0)
{
lean_ctor_set(v___x_1251_, 2, v___x_1271_);
lean_ctor_set(v___x_1251_, 0, v___x_1264_);
v___x_1273_ = v___x_1251_;
goto v_reusejp_1272_;
}
else
{
lean_object* v_reuseFailAlloc_1277_; 
v_reuseFailAlloc_1277_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1277_, 0, v___x_1264_);
lean_ctor_set(v_reuseFailAlloc_1277_, 1, v_lit_1248_);
lean_ctor_set(v_reuseFailAlloc_1277_, 2, v___x_1271_);
v___x_1273_ = v_reuseFailAlloc_1277_;
goto v_reusejp_1272_;
}
v_reusejp_1272_:
{
lean_object* v___x_1275_; 
if (v_isShared_1247_ == 0)
{
lean_ctor_set(v___x_1246_, 0, v___x_1273_);
v___x_1275_ = v___x_1246_;
goto v_reusejp_1274_;
}
else
{
lean_object* v_reuseFailAlloc_1276_; 
v_reuseFailAlloc_1276_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1276_, 0, v___x_1273_);
v___x_1275_ = v_reuseFailAlloc_1276_;
goto v_reusejp_1274_;
}
v_reusejp_1274_:
{
return v___x_1275_;
}
}
}
}
case 2:
{
lean_object* v_lit_1280_; lean_object* v_proof_1281_; lean_object* v___x_1283_; uint8_t v_isShared_1284_; uint8_t v_isSharedCheck_1304_; 
lean_dec(v___x_1227_);
lean_dec(v_u_1213_);
v_lit_1280_ = lean_ctor_get(v_a_1244_, 1);
v_proof_1281_ = lean_ctor_get(v_a_1244_, 2);
v_isSharedCheck_1304_ = !lean_is_exclusive(v_a_1244_);
if (v_isSharedCheck_1304_ == 0)
{
lean_object* v_unused_1305_; 
v_unused_1305_ = lean_ctor_get(v_a_1244_, 0);
lean_dec(v_unused_1305_);
v___x_1283_ = v_a_1244_;
v_isShared_1284_ = v_isSharedCheck_1304_;
goto v_resetjp_1282_;
}
else
{
lean_inc(v_proof_1281_);
lean_inc(v_lit_1280_);
lean_dec(v_a_1244_);
v___x_1283_ = lean_box(0);
v_isShared_1284_ = v_isSharedCheck_1304_;
goto v_resetjp_1282_;
}
v_resetjp_1282_:
{
lean_object* v___x_1285_; lean_object* v___x_1286_; lean_object* v___x_1287_; lean_object* v___x_1288_; lean_object* v___x_1289_; lean_object* v___x_1290_; lean_object* v___x_1291_; lean_object* v___x_1292_; lean_object* v___x_1293_; lean_object* v___x_1294_; lean_object* v___x_1295_; lean_object* v___x_1296_; lean_object* v___x_1297_; lean_object* v___x_1299_; 
v___x_1285_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing___closed__2));
lean_inc_ref(v___x_1231_);
v___x_1286_ = l_Lean_Expr_const___override(v___x_1285_, v___x_1231_);
lean_inc_ref(v_00_u03b1_1214_);
v___x_1287_ = l_Lean_Expr_app___override(v___x_1286_, v_00_u03b1_1214_);
lean_inc(v_a_1222_);
v___x_1288_ = l_Lean_Expr_app___override(v___x_1287_, v_a_1222_);
v___x_1289_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__7, &lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__7_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__7);
lean_inc_ref(v_lit_1280_);
v___x_1290_ = l_Lean_Expr_app___override(v___x_1289_, v_lit_1280_);
v___x_1291_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__9));
v___x_1292_ = l_Lean_Expr_const___override(v___x_1291_, v___x_1231_);
v___x_1293_ = l_Lean_Expr_app___override(v___x_1292_, v_00_u03b1_1214_);
v___x_1294_ = l_Lean_Expr_app___override(v___x_1293_, v_a_1222_);
v___x_1295_ = l_Lean_Expr_app___override(v___x_1294_, v_arg_1226_);
v___x_1296_ = l_Lean_Expr_app___override(v___x_1295_, v___x_1290_);
v___x_1297_ = l_Lean_Expr_app___override(v___x_1296_, v_proof_1281_);
if (v_isShared_1284_ == 0)
{
lean_ctor_set(v___x_1283_, 2, v___x_1297_);
lean_ctor_set(v___x_1283_, 0, v___x_1288_);
v___x_1299_ = v___x_1283_;
goto v_reusejp_1298_;
}
else
{
lean_object* v_reuseFailAlloc_1303_; 
v_reuseFailAlloc_1303_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1303_, 0, v___x_1288_);
lean_ctor_set(v_reuseFailAlloc_1303_, 1, v_lit_1280_);
lean_ctor_set(v_reuseFailAlloc_1303_, 2, v___x_1297_);
v___x_1299_ = v_reuseFailAlloc_1303_;
goto v_reusejp_1298_;
}
v_reusejp_1298_:
{
lean_object* v___x_1301_; 
if (v_isShared_1247_ == 0)
{
lean_ctor_set(v___x_1246_, 0, v___x_1299_);
v___x_1301_ = v___x_1246_;
goto v_reusejp_1300_;
}
else
{
lean_object* v_reuseFailAlloc_1302_; 
v_reuseFailAlloc_1302_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1302_, 0, v___x_1299_);
v___x_1301_ = v_reuseFailAlloc_1302_;
goto v_reusejp_1300_;
}
v_reusejp_1300_:
{
return v___x_1301_;
}
}
}
}
case 3:
{
lean_object* v_q_1306_; lean_object* v_n_1307_; lean_object* v_d_1308_; lean_object* v_proof_1309_; lean_object* v___x_1311_; uint8_t v_isShared_1312_; uint8_t v_isSharedCheck_1354_; 
lean_del_object(v___x_1246_);
v_q_1306_ = lean_ctor_get(v_a_1244_, 1);
v_n_1307_ = lean_ctor_get(v_a_1244_, 2);
v_d_1308_ = lean_ctor_get(v_a_1244_, 3);
v_proof_1309_ = lean_ctor_get(v_a_1244_, 4);
v_isSharedCheck_1354_ = !lean_is_exclusive(v_a_1244_);
if (v_isSharedCheck_1354_ == 0)
{
lean_object* v_unused_1355_; 
v_unused_1355_ = lean_ctor_get(v_a_1244_, 0);
lean_dec(v_unused_1355_);
v___x_1311_ = v_a_1244_;
v_isShared_1312_ = v_isSharedCheck_1354_;
goto v_resetjp_1310_;
}
else
{
lean_inc(v_proof_1309_);
lean_inc(v_d_1308_);
lean_inc(v_n_1307_);
lean_inc(v_q_1306_);
lean_dec(v_a_1244_);
v___x_1311_ = lean_box(0);
v_isShared_1312_ = v_isSharedCheck_1354_;
goto v_resetjp_1310_;
}
v_resetjp_1310_:
{
lean_object* v___x_1313_; 
lean_inc(v_a_1222_);
lean_inc_ref(v_00_u03b1_1214_);
v___x_1313_ = lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing(v_u_1213_, v_00_u03b1_1214_, v_a_1222_, v___y_1216_, v___y_1217_, v___y_1218_, v___y_1219_);
if (lean_obj_tag(v___x_1313_) == 0)
{
lean_object* v_a_1314_; lean_object* v___x_1316_; uint8_t v_isShared_1317_; uint8_t v_isSharedCheck_1345_; 
v_a_1314_ = lean_ctor_get(v___x_1313_, 0);
v_isSharedCheck_1345_ = !lean_is_exclusive(v___x_1313_);
if (v_isSharedCheck_1345_ == 0)
{
v___x_1316_ = v___x_1313_;
v_isShared_1317_ = v_isSharedCheck_1345_;
goto v_resetjp_1315_;
}
else
{
lean_inc(v_a_1314_);
lean_dec(v___x_1313_);
v___x_1316_ = lean_box(0);
v_isShared_1317_ = v_isSharedCheck_1345_;
goto v_resetjp_1315_;
}
v_resetjp_1315_:
{
lean_object* v___x_1318_; lean_object* v___x_1319_; lean_object* v___x_1320_; lean_object* v___x_1321_; lean_object* v___x_1322_; lean_object* v___x_1323_; lean_object* v___x_1324_; lean_object* v___x_1325_; lean_object* v___x_1326_; lean_object* v___x_1327_; lean_object* v___x_1328_; lean_object* v___x_1329_; lean_object* v___x_1330_; lean_object* v___x_1331_; lean_object* v___x_1332_; lean_object* v___x_1333_; lean_object* v___x_1334_; lean_object* v___x_1335_; lean_object* v___x_1336_; lean_object* v___x_1337_; lean_object* v___x_1338_; lean_object* v___x_1340_; 
v___x_1318_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__10));
lean_inc_ref_n(v___x_1231_, 2);
v___x_1319_ = l_Lean_Expr_const___override(v___x_1318_, v___x_1231_);
lean_inc_ref_n(v_00_u03b1_1214_, 2);
v___x_1320_ = l_Lean_Expr_app___override(v___x_1319_, v_00_u03b1_1214_);
v___x_1321_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__12));
v___x_1322_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1322_, 0, v___x_1227_);
lean_ctor_set(v___x_1322_, 1, v___x_1228_);
v___x_1323_ = l_Lean_Expr_const___override(v___x_1321_, v___x_1322_);
v___x_1324_ = l_Lean_Expr_app___override(v___x_1323_, v___x_1320_);
v___x_1325_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__13));
v___x_1326_ = l_Lean_Expr_const___override(v___x_1325_, v___x_1231_);
v___x_1327_ = l_Lean_Expr_app___override(v___x_1326_, v_00_u03b1_1214_);
lean_inc(v_a_1222_);
v___x_1328_ = l_Lean_Expr_app___override(v___x_1327_, v_a_1222_);
v___x_1329_ = l_Lean_Expr_app___override(v___x_1324_, v___x_1328_);
v___x_1330_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__15));
v___x_1331_ = l_Lean_Expr_const___override(v___x_1330_, v___x_1231_);
v___x_1332_ = l_Lean_Expr_app___override(v___x_1331_, v_00_u03b1_1214_);
v___x_1333_ = l_Lean_Expr_app___override(v___x_1332_, v_a_1222_);
v___x_1334_ = l_Lean_Expr_app___override(v___x_1333_, v_a_1314_);
v___x_1335_ = l_Lean_Expr_app___override(v___x_1334_, v_arg_1226_);
lean_inc_ref(v_n_1307_);
v___x_1336_ = l_Lean_Expr_app___override(v___x_1335_, v_n_1307_);
lean_inc_ref(v_d_1308_);
v___x_1337_ = l_Lean_Expr_app___override(v___x_1336_, v_d_1308_);
v___x_1338_ = l_Lean_Expr_app___override(v___x_1337_, v_proof_1309_);
if (v_isShared_1312_ == 0)
{
lean_ctor_set(v___x_1311_, 4, v___x_1338_);
lean_ctor_set(v___x_1311_, 0, v___x_1329_);
v___x_1340_ = v___x_1311_;
goto v_reusejp_1339_;
}
else
{
lean_object* v_reuseFailAlloc_1344_; 
v_reuseFailAlloc_1344_ = lean_alloc_ctor(3, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1344_, 0, v___x_1329_);
lean_ctor_set(v_reuseFailAlloc_1344_, 1, v_q_1306_);
lean_ctor_set(v_reuseFailAlloc_1344_, 2, v_n_1307_);
lean_ctor_set(v_reuseFailAlloc_1344_, 3, v_d_1308_);
lean_ctor_set(v_reuseFailAlloc_1344_, 4, v___x_1338_);
v___x_1340_ = v_reuseFailAlloc_1344_;
goto v_reusejp_1339_;
}
v_reusejp_1339_:
{
lean_object* v___x_1342_; 
if (v_isShared_1317_ == 0)
{
lean_ctor_set(v___x_1316_, 0, v___x_1340_);
v___x_1342_ = v___x_1316_;
goto v_reusejp_1341_;
}
else
{
lean_object* v_reuseFailAlloc_1343_; 
v_reuseFailAlloc_1343_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1343_, 0, v___x_1340_);
v___x_1342_ = v_reuseFailAlloc_1343_;
goto v_reusejp_1341_;
}
v_reusejp_1341_:
{
return v___x_1342_;
}
}
}
}
else
{
lean_object* v_a_1346_; lean_object* v___x_1348_; uint8_t v_isShared_1349_; uint8_t v_isSharedCheck_1353_; 
lean_del_object(v___x_1311_);
lean_dec_ref(v_proof_1309_);
lean_dec_ref(v_d_1308_);
lean_dec_ref(v_n_1307_);
lean_dec_ref(v_q_1306_);
lean_dec_ref_known(v___x_1231_, 2);
lean_dec(v___x_1227_);
lean_dec_ref(v_arg_1226_);
lean_dec(v_a_1222_);
lean_dec_ref(v_00_u03b1_1214_);
v_a_1346_ = lean_ctor_get(v___x_1313_, 0);
v_isSharedCheck_1353_ = !lean_is_exclusive(v___x_1313_);
if (v_isSharedCheck_1353_ == 0)
{
v___x_1348_ = v___x_1313_;
v_isShared_1349_ = v_isSharedCheck_1353_;
goto v_resetjp_1347_;
}
else
{
lean_inc(v_a_1346_);
lean_dec(v___x_1313_);
v___x_1348_ = lean_box(0);
v_isShared_1349_ = v_isSharedCheck_1353_;
goto v_resetjp_1347_;
}
v_resetjp_1347_:
{
lean_object* v___x_1351_; 
if (v_isShared_1349_ == 0)
{
v___x_1351_ = v___x_1348_;
goto v_reusejp_1350_;
}
else
{
lean_object* v_reuseFailAlloc_1352_; 
v_reuseFailAlloc_1352_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1352_, 0, v_a_1346_);
v___x_1351_ = v_reuseFailAlloc_1352_;
goto v_reusejp_1350_;
}
v_reusejp_1350_:
{
return v___x_1351_;
}
}
}
}
}
case 4:
{
lean_object* v_q_1356_; lean_object* v_n_1357_; lean_object* v_d_1358_; lean_object* v_proof_1359_; lean_object* v___x_1361_; uint8_t v_isShared_1362_; uint8_t v_isSharedCheck_1394_; 
lean_del_object(v___x_1246_);
lean_dec(v___x_1227_);
v_q_1356_ = lean_ctor_get(v_a_1244_, 1);
v_n_1357_ = lean_ctor_get(v_a_1244_, 2);
v_d_1358_ = lean_ctor_get(v_a_1244_, 3);
v_proof_1359_ = lean_ctor_get(v_a_1244_, 4);
v_isSharedCheck_1394_ = !lean_is_exclusive(v_a_1244_);
if (v_isSharedCheck_1394_ == 0)
{
lean_object* v_unused_1395_; 
v_unused_1395_ = lean_ctor_get(v_a_1244_, 0);
lean_dec(v_unused_1395_);
v___x_1361_ = v_a_1244_;
v_isShared_1362_ = v_isSharedCheck_1394_;
goto v_resetjp_1360_;
}
else
{
lean_inc(v_proof_1359_);
lean_inc(v_d_1358_);
lean_inc(v_n_1357_);
lean_inc(v_q_1356_);
lean_dec(v_a_1244_);
v___x_1361_ = lean_box(0);
v_isShared_1362_ = v_isSharedCheck_1394_;
goto v_resetjp_1360_;
}
v_resetjp_1360_:
{
lean_object* v___x_1363_; 
lean_inc(v_a_1222_);
lean_inc_ref(v_00_u03b1_1214_);
v___x_1363_ = lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing(v_u_1213_, v_00_u03b1_1214_, v_a_1222_, v___y_1216_, v___y_1217_, v___y_1218_, v___y_1219_);
if (lean_obj_tag(v___x_1363_) == 0)
{
lean_object* v_a_1364_; lean_object* v___x_1366_; uint8_t v_isShared_1367_; uint8_t v_isSharedCheck_1385_; 
v_a_1364_ = lean_ctor_get(v___x_1363_, 0);
v_isSharedCheck_1385_ = !lean_is_exclusive(v___x_1363_);
if (v_isSharedCheck_1385_ == 0)
{
v___x_1366_ = v___x_1363_;
v_isShared_1367_ = v_isSharedCheck_1385_;
goto v_resetjp_1365_;
}
else
{
lean_inc(v_a_1364_);
lean_dec(v___x_1363_);
v___x_1366_ = lean_box(0);
v_isShared_1367_ = v_isSharedCheck_1385_;
goto v_resetjp_1365_;
}
v_resetjp_1365_:
{
lean_object* v___x_1368_; lean_object* v___x_1369_; lean_object* v___x_1370_; lean_object* v___x_1371_; lean_object* v___x_1372_; lean_object* v___x_1373_; lean_object* v___x_1374_; lean_object* v___x_1375_; lean_object* v___x_1376_; lean_object* v___x_1377_; lean_object* v___x_1378_; lean_object* v___x_1380_; 
v___x_1368_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__7, &lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__7_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__7);
lean_inc_ref(v_n_1357_);
v___x_1369_ = l_Lean_Expr_app___override(v___x_1368_, v_n_1357_);
v___x_1370_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__17));
v___x_1371_ = l_Lean_Expr_const___override(v___x_1370_, v___x_1231_);
v___x_1372_ = l_Lean_Expr_app___override(v___x_1371_, v_00_u03b1_1214_);
lean_inc(v_a_1222_);
v___x_1373_ = l_Lean_Expr_app___override(v___x_1372_, v_a_1222_);
v___x_1374_ = l_Lean_Expr_app___override(v___x_1373_, v_a_1364_);
v___x_1375_ = l_Lean_Expr_app___override(v___x_1374_, v_arg_1226_);
v___x_1376_ = l_Lean_Expr_app___override(v___x_1375_, v___x_1369_);
lean_inc_ref(v_d_1358_);
v___x_1377_ = l_Lean_Expr_app___override(v___x_1376_, v_d_1358_);
v___x_1378_ = l_Lean_Expr_app___override(v___x_1377_, v_proof_1359_);
if (v_isShared_1362_ == 0)
{
lean_ctor_set(v___x_1361_, 4, v___x_1378_);
lean_ctor_set(v___x_1361_, 0, v_a_1222_);
v___x_1380_ = v___x_1361_;
goto v_reusejp_1379_;
}
else
{
lean_object* v_reuseFailAlloc_1384_; 
v_reuseFailAlloc_1384_ = lean_alloc_ctor(4, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1384_, 0, v_a_1222_);
lean_ctor_set(v_reuseFailAlloc_1384_, 1, v_q_1356_);
lean_ctor_set(v_reuseFailAlloc_1384_, 2, v_n_1357_);
lean_ctor_set(v_reuseFailAlloc_1384_, 3, v_d_1358_);
lean_ctor_set(v_reuseFailAlloc_1384_, 4, v___x_1378_);
v___x_1380_ = v_reuseFailAlloc_1384_;
goto v_reusejp_1379_;
}
v_reusejp_1379_:
{
lean_object* v___x_1382_; 
if (v_isShared_1367_ == 0)
{
lean_ctor_set(v___x_1366_, 0, v___x_1380_);
v___x_1382_ = v___x_1366_;
goto v_reusejp_1381_;
}
else
{
lean_object* v_reuseFailAlloc_1383_; 
v_reuseFailAlloc_1383_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1383_, 0, v___x_1380_);
v___x_1382_ = v_reuseFailAlloc_1383_;
goto v_reusejp_1381_;
}
v_reusejp_1381_:
{
return v___x_1382_;
}
}
}
}
else
{
lean_object* v_a_1386_; lean_object* v___x_1388_; uint8_t v_isShared_1389_; uint8_t v_isSharedCheck_1393_; 
lean_del_object(v___x_1361_);
lean_dec_ref(v_proof_1359_);
lean_dec_ref(v_d_1358_);
lean_dec_ref(v_n_1357_);
lean_dec_ref(v_q_1356_);
lean_dec_ref_known(v___x_1231_, 2);
lean_dec_ref(v_arg_1226_);
lean_dec(v_a_1222_);
lean_dec_ref(v_00_u03b1_1214_);
v_a_1386_ = lean_ctor_get(v___x_1363_, 0);
v_isSharedCheck_1393_ = !lean_is_exclusive(v___x_1363_);
if (v_isSharedCheck_1393_ == 0)
{
v___x_1388_ = v___x_1363_;
v_isShared_1389_ = v_isSharedCheck_1393_;
goto v_resetjp_1387_;
}
else
{
lean_inc(v_a_1386_);
lean_dec(v___x_1363_);
v___x_1388_ = lean_box(0);
v_isShared_1389_ = v_isSharedCheck_1393_;
goto v_resetjp_1387_;
}
v_resetjp_1387_:
{
lean_object* v___x_1391_; 
if (v_isShared_1389_ == 0)
{
v___x_1391_ = v___x_1388_;
goto v_reusejp_1390_;
}
else
{
lean_object* v_reuseFailAlloc_1392_; 
v_reuseFailAlloc_1392_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1392_, 0, v_a_1386_);
v___x_1391_ = v_reuseFailAlloc_1392_;
goto v_reusejp_1390_;
}
v_reusejp_1390_:
{
return v___x_1391_;
}
}
}
}
}
default: 
{
lean_object* v___x_1396_; lean_object* v___x_1397_; 
lean_del_object(v___x_1246_);
lean_dec(v_a_1244_);
lean_dec_ref_known(v___x_1231_, 2);
lean_dec(v___x_1227_);
lean_dec_ref(v_arg_1226_);
lean_dec(v_a_1222_);
lean_dec_ref(v_00_u03b1_1214_);
lean_dec(v_u_1213_);
v___x_1396_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1);
v___x_1397_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0___redArg(v___x_1396_, v___y_1216_, v___y_1217_, v___y_1218_, v___y_1219_);
return v___x_1397_;
}
}
}
}
else
{
lean_dec_ref_known(v___x_1231_, 2);
lean_dec(v___x_1227_);
lean_dec_ref(v_arg_1226_);
lean_dec(v_a_1222_);
lean_dec_ref(v_00_u03b1_1214_);
lean_dec(v_u_1213_);
return v___x_1243_;
}
}
}
else
{
lean_object* v___x_1420_; lean_object* v___x_1421_; 
lean_dec(v_a_1224_);
lean_dec(v_a_1222_);
lean_dec_ref(v_00_u03b1_1214_);
lean_dec(v_u_1213_);
v___x_1420_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1);
v___x_1421_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0___redArg(v___x_1420_, v___y_1216_, v___y_1217_, v___y_1218_, v___y_1219_);
return v___x_1421_;
}
}
else
{
lean_object* v_a_1422_; lean_object* v___x_1424_; uint8_t v_isShared_1425_; uint8_t v_isSharedCheck_1429_; 
lean_dec(v_a_1222_);
lean_dec_ref(v_00_u03b1_1214_);
lean_dec(v_u_1213_);
v_a_1422_ = lean_ctor_get(v___x_1223_, 0);
v_isSharedCheck_1429_ = !lean_is_exclusive(v___x_1223_);
if (v_isSharedCheck_1429_ == 0)
{
v___x_1424_ = v___x_1223_;
v_isShared_1425_ = v_isSharedCheck_1429_;
goto v_resetjp_1423_;
}
else
{
lean_inc(v_a_1422_);
lean_dec(v___x_1223_);
v___x_1424_ = lean_box(0);
v_isShared_1425_ = v_isSharedCheck_1429_;
goto v_resetjp_1423_;
}
v_resetjp_1423_:
{
lean_object* v___x_1427_; 
if (v_isShared_1425_ == 0)
{
v___x_1427_ = v___x_1424_;
goto v_reusejp_1426_;
}
else
{
lean_object* v_reuseFailAlloc_1428_; 
v_reuseFailAlloc_1428_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1428_, 0, v_a_1422_);
v___x_1427_ = v_reuseFailAlloc_1428_;
goto v_reusejp_1426_;
}
v_reusejp_1426_:
{
return v___x_1427_;
}
}
}
}
else
{
lean_object* v_a_1430_; lean_object* v___x_1432_; uint8_t v_isShared_1433_; uint8_t v_isSharedCheck_1437_; 
lean_dec_ref(v_e_1215_);
lean_dec_ref(v_00_u03b1_1214_);
lean_dec(v_u_1213_);
v_a_1430_ = lean_ctor_get(v___x_1221_, 0);
v_isSharedCheck_1437_ = !lean_is_exclusive(v___x_1221_);
if (v_isSharedCheck_1437_ == 0)
{
v___x_1432_ = v___x_1221_;
v_isShared_1433_ = v_isSharedCheck_1437_;
goto v_resetjp_1431_;
}
else
{
lean_inc(v_a_1430_);
lean_dec(v___x_1221_);
v___x_1432_ = lean_box(0);
v_isShared_1433_ = v_isSharedCheck_1437_;
goto v_resetjp_1431_;
}
v_resetjp_1431_:
{
lean_object* v___x_1435_; 
if (v_isShared_1433_ == 0)
{
v___x_1435_ = v___x_1432_;
goto v_reusejp_1434_;
}
else
{
lean_object* v_reuseFailAlloc_1436_; 
v_reuseFailAlloc_1436_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1436_, 0, v_a_1430_);
v___x_1435_ = v_reuseFailAlloc_1436_;
goto v_reusejp_1434_;
}
v_reusejp_1434_:
{
return v___x_1435_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___boxed(lean_object* v_u_1438_, lean_object* v_00_u03b1_1439_, lean_object* v_e_1440_, lean_object* v___y_1441_, lean_object* v___y_1442_, lean_object* v___y_1443_, lean_object* v___y_1444_, lean_object* v___y_1445_){
_start:
{
lean_object* v_res_1446_; 
v_res_1446_ = lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1(v_u_1438_, v_00_u03b1_1439_, v_e_1440_, v___y_1441_, v___y_1442_, v___y_1443_, v___y_1444_);
lean_dec(v___y_1444_);
lean_dec_ref(v___y_1443_);
lean_dec(v___y_1442_);
lean_dec_ref(v___y_1441_);
return v_res_1446_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_NormNum_evalNNRatCast_spec__0___redArg(lean_object* v_e_1459_, lean_object* v___y_1460_){
_start:
{
uint8_t v___x_1462_; 
v___x_1462_ = l_Lean_Expr_hasMVar(v_e_1459_);
if (v___x_1462_ == 0)
{
lean_object* v___x_1463_; 
v___x_1463_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1463_, 0, v_e_1459_);
return v___x_1463_;
}
else
{
lean_object* v___x_1464_; lean_object* v_mctx_1465_; lean_object* v___x_1466_; lean_object* v_fst_1467_; lean_object* v_snd_1468_; lean_object* v___x_1469_; lean_object* v_cache_1470_; lean_object* v_zetaDeltaFVarIds_1471_; lean_object* v_postponed_1472_; lean_object* v_diag_1473_; lean_object* v___x_1475_; uint8_t v_isShared_1476_; uint8_t v_isSharedCheck_1482_; 
v___x_1464_ = lean_st_ref_get(v___y_1460_);
v_mctx_1465_ = lean_ctor_get(v___x_1464_, 0);
lean_inc_ref(v_mctx_1465_);
lean_dec(v___x_1464_);
v___x_1466_ = l_Lean_instantiateMVarsCore(v_mctx_1465_, v_e_1459_);
v_fst_1467_ = lean_ctor_get(v___x_1466_, 0);
lean_inc(v_fst_1467_);
v_snd_1468_ = lean_ctor_get(v___x_1466_, 1);
lean_inc(v_snd_1468_);
lean_dec_ref(v___x_1466_);
v___x_1469_ = lean_st_ref_take(v___y_1460_);
v_cache_1470_ = lean_ctor_get(v___x_1469_, 1);
v_zetaDeltaFVarIds_1471_ = lean_ctor_get(v___x_1469_, 2);
v_postponed_1472_ = lean_ctor_get(v___x_1469_, 3);
v_diag_1473_ = lean_ctor_get(v___x_1469_, 4);
v_isSharedCheck_1482_ = !lean_is_exclusive(v___x_1469_);
if (v_isSharedCheck_1482_ == 0)
{
lean_object* v_unused_1483_; 
v_unused_1483_ = lean_ctor_get(v___x_1469_, 0);
lean_dec(v_unused_1483_);
v___x_1475_ = v___x_1469_;
v_isShared_1476_ = v_isSharedCheck_1482_;
goto v_resetjp_1474_;
}
else
{
lean_inc(v_diag_1473_);
lean_inc(v_postponed_1472_);
lean_inc(v_zetaDeltaFVarIds_1471_);
lean_inc(v_cache_1470_);
lean_dec(v___x_1469_);
v___x_1475_ = lean_box(0);
v_isShared_1476_ = v_isSharedCheck_1482_;
goto v_resetjp_1474_;
}
v_resetjp_1474_:
{
lean_object* v___x_1478_; 
if (v_isShared_1476_ == 0)
{
lean_ctor_set(v___x_1475_, 0, v_snd_1468_);
v___x_1478_ = v___x_1475_;
goto v_reusejp_1477_;
}
else
{
lean_object* v_reuseFailAlloc_1481_; 
v_reuseFailAlloc_1481_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1481_, 0, v_snd_1468_);
lean_ctor_set(v_reuseFailAlloc_1481_, 1, v_cache_1470_);
lean_ctor_set(v_reuseFailAlloc_1481_, 2, v_zetaDeltaFVarIds_1471_);
lean_ctor_set(v_reuseFailAlloc_1481_, 3, v_postponed_1472_);
lean_ctor_set(v_reuseFailAlloc_1481_, 4, v_diag_1473_);
v___x_1478_ = v_reuseFailAlloc_1481_;
goto v_reusejp_1477_;
}
v_reusejp_1477_:
{
lean_object* v___x_1479_; lean_object* v___x_1480_; 
v___x_1479_ = lean_st_ref_set(v___y_1460_, v___x_1478_);
v___x_1480_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1480_, 0, v_fst_1467_);
return v___x_1480_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_NormNum_evalNNRatCast_spec__0___redArg___boxed(lean_object* v_e_1484_, lean_object* v___y_1485_, lean_object* v___y_1486_){
_start:
{
lean_object* v_res_1487_; 
v_res_1487_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_NormNum_evalNNRatCast_spec__0___redArg(v_e_1484_, v___y_1485_);
lean_dec(v___y_1485_);
return v_res_1487_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_NormNum_evalNNRatCast_spec__0(lean_object* v_e_1488_, lean_object* v___y_1489_, lean_object* v___y_1490_, lean_object* v___y_1491_, lean_object* v___y_1492_){
_start:
{
lean_object* v___x_1494_; 
v___x_1494_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_NormNum_evalNNRatCast_spec__0___redArg(v_e_1488_, v___y_1490_);
return v___x_1494_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_NormNum_evalNNRatCast_spec__0___boxed(lean_object* v_e_1495_, lean_object* v___y_1496_, lean_object* v___y_1497_, lean_object* v___y_1498_, lean_object* v___y_1499_, lean_object* v___y_1500_){
_start:
{
lean_object* v_res_1501_; 
v_res_1501_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_NormNum_evalNNRatCast_spec__0(v_e_1495_, v___y_1496_, v___y_1497_, v___y_1498_, v___y_1499_);
lean_dec(v___y_1499_);
lean_dec_ref(v___y_1498_);
lean_dec(v___y_1497_);
lean_dec_ref(v___y_1496_);
return v_res_1501_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__0(lean_object* v___x_1505_, uint8_t v___x_1506_, lean_object* v___x_1507_, lean_object* v___x_1508_, lean_object* v___x_1509_, lean_object* v_00_u03b1_1510_, lean_object* v_e_1511_, uint8_t v___x_1512_, lean_object* v___y_1513_, lean_object* v___y_1514_, lean_object* v___y_1515_, lean_object* v___y_1516_){
_start:
{
lean_object* v___x_1518_; 
lean_inc(v___x_1507_);
v___x_1518_ = l_Lean_Meta_mkFreshExprMVar(v___x_1505_, v___x_1506_, v___x_1507_, v___y_1513_, v___y_1514_, v___y_1515_, v___y_1516_);
if (lean_obj_tag(v___x_1518_) == 0)
{
lean_object* v_a_1519_; lean_object* v___x_1520_; lean_object* v___x_1521_; lean_object* v___x_1522_; lean_object* v___x_1523_; 
v_a_1519_ = lean_ctor_get(v___x_1518_, 0);
lean_inc(v_a_1519_);
lean_dec_ref_known(v___x_1518_, 1);
v___x_1520_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__4));
v___x_1521_ = l_Lean_Expr_const___override(v___x_1520_, v___x_1508_);
v___x_1522_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1522_, 0, v___x_1521_);
v___x_1523_ = l_Lean_Meta_mkFreshExprMVar(v___x_1522_, v___x_1506_, v___x_1507_, v___y_1513_, v___y_1514_, v___y_1515_, v___y_1516_);
if (lean_obj_tag(v___x_1523_) == 0)
{
lean_object* v_a_1524_; lean_object* v_keyedConfig_1525_; uint8_t v_trackZetaDelta_1526_; lean_object* v_zetaDeltaSet_1527_; lean_object* v_lctx_1528_; lean_object* v_localInstances_1529_; lean_object* v_defEqCtx_x3f_1530_; lean_object* v_synthPendingDepth_1531_; lean_object* v_customCanUnfoldPredicate_x3f_1532_; uint8_t v_univApprox_1533_; uint8_t v_inTypeClassResolution_1534_; uint8_t v_cacheInferType_1535_; lean_object* v___x_1537_; uint8_t v_isShared_1538_; uint8_t v_isSharedCheck_1583_; 
v_a_1524_ = lean_ctor_get(v___x_1523_, 0);
lean_inc(v_a_1524_);
lean_dec_ref_known(v___x_1523_, 1);
v_keyedConfig_1525_ = lean_ctor_get(v___y_1513_, 0);
v_trackZetaDelta_1526_ = lean_ctor_get_uint8(v___y_1513_, sizeof(void*)*7);
v_zetaDeltaSet_1527_ = lean_ctor_get(v___y_1513_, 1);
v_lctx_1528_ = lean_ctor_get(v___y_1513_, 2);
v_localInstances_1529_ = lean_ctor_get(v___y_1513_, 3);
v_defEqCtx_x3f_1530_ = lean_ctor_get(v___y_1513_, 4);
v_synthPendingDepth_1531_ = lean_ctor_get(v___y_1513_, 5);
v_customCanUnfoldPredicate_x3f_1532_ = lean_ctor_get(v___y_1513_, 6);
v_univApprox_1533_ = lean_ctor_get_uint8(v___y_1513_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1534_ = lean_ctor_get_uint8(v___y_1513_, sizeof(void*)*7 + 2);
v_cacheInferType_1535_ = lean_ctor_get_uint8(v___y_1513_, sizeof(void*)*7 + 3);
v_isSharedCheck_1583_ = !lean_is_exclusive(v___y_1513_);
if (v_isSharedCheck_1583_ == 0)
{
v___x_1537_ = v___y_1513_;
v_isShared_1538_ = v_isSharedCheck_1583_;
goto v_resetjp_1536_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_1532_);
lean_inc(v_synthPendingDepth_1531_);
lean_inc(v_defEqCtx_x3f_1530_);
lean_inc(v_localInstances_1529_);
lean_inc(v_lctx_1528_);
lean_inc(v_zetaDeltaSet_1527_);
lean_inc(v_keyedConfig_1525_);
lean_dec(v___y_1513_);
v___x_1537_ = lean_box(0);
v_isShared_1538_ = v_isSharedCheck_1583_;
goto v_resetjp_1536_;
}
v_resetjp_1536_:
{
lean_object* v___x_1539_; lean_object* v___x_1540_; lean_object* v___x_1541_; lean_object* v___x_1542_; lean_object* v___x_1543_; uint8_t v___x_1544_; lean_object* v___x_1545_; lean_object* v___x_1547_; 
v___x_1539_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__0___closed__0));
v___x_1540_ = l_Lean_Expr_const___override(v___x_1539_, v___x_1509_);
v___x_1541_ = l_Lean_Expr_app___override(v___x_1540_, v_00_u03b1_1510_);
lean_inc(v_a_1519_);
v___x_1542_ = l_Lean_Expr_app___override(v___x_1541_, v_a_1519_);
lean_inc(v_a_1524_);
v___x_1543_ = l_Lean_Expr_app___override(v___x_1542_, v_a_1524_);
v___x_1544_ = 2;
v___x_1545_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1544_, v_keyedConfig_1525_);
if (v_isShared_1538_ == 0)
{
lean_ctor_set(v___x_1537_, 0, v___x_1545_);
v___x_1547_ = v___x_1537_;
goto v_reusejp_1546_;
}
else
{
lean_object* v_reuseFailAlloc_1582_; 
v_reuseFailAlloc_1582_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_1582_, 0, v___x_1545_);
lean_ctor_set(v_reuseFailAlloc_1582_, 1, v_zetaDeltaSet_1527_);
lean_ctor_set(v_reuseFailAlloc_1582_, 2, v_lctx_1528_);
lean_ctor_set(v_reuseFailAlloc_1582_, 3, v_localInstances_1529_);
lean_ctor_set(v_reuseFailAlloc_1582_, 4, v_defEqCtx_x3f_1530_);
lean_ctor_set(v_reuseFailAlloc_1582_, 5, v_synthPendingDepth_1531_);
lean_ctor_set(v_reuseFailAlloc_1582_, 6, v_customCanUnfoldPredicate_x3f_1532_);
lean_ctor_set_uint8(v_reuseFailAlloc_1582_, sizeof(void*)*7, v_trackZetaDelta_1526_);
lean_ctor_set_uint8(v_reuseFailAlloc_1582_, sizeof(void*)*7 + 1, v_univApprox_1533_);
lean_ctor_set_uint8(v_reuseFailAlloc_1582_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1534_);
lean_ctor_set_uint8(v_reuseFailAlloc_1582_, sizeof(void*)*7 + 3, v_cacheInferType_1535_);
v___x_1547_ = v_reuseFailAlloc_1582_;
goto v_reusejp_1546_;
}
v_reusejp_1546_:
{
lean_object* v___x_1548_; 
v___x_1548_ = l_Lean_Meta_isExprDefEq(v___x_1543_, v_e_1511_, v___x_1547_, v___y_1514_, v___y_1515_, v___y_1516_);
lean_dec_ref(v___x_1547_);
if (lean_obj_tag(v___x_1548_) == 0)
{
lean_object* v_a_1549_; lean_object* v___x_1551_; uint8_t v_isShared_1552_; uint8_t v_isSharedCheck_1573_; 
v_a_1549_ = lean_ctor_get(v___x_1548_, 0);
v_isSharedCheck_1573_ = !lean_is_exclusive(v___x_1548_);
if (v_isSharedCheck_1573_ == 0)
{
v___x_1551_ = v___x_1548_;
v_isShared_1552_ = v_isSharedCheck_1573_;
goto v_resetjp_1550_;
}
else
{
lean_inc(v_a_1549_);
lean_dec(v___x_1548_);
v___x_1551_ = lean_box(0);
v_isShared_1552_ = v_isSharedCheck_1573_;
goto v_resetjp_1550_;
}
v_resetjp_1550_:
{
uint8_t v___x_1553_; 
v___x_1553_ = lean_unbox(v_a_1549_);
if (v___x_1553_ == 0)
{
lean_object* v___x_1554_; lean_object* v___x_1555_; lean_object* v___x_1557_; 
v___x_1554_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1554_, 0, v_a_1524_);
lean_ctor_set(v___x_1554_, 1, v_a_1549_);
v___x_1555_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1555_, 0, v_a_1519_);
lean_ctor_set(v___x_1555_, 1, v___x_1554_);
if (v_isShared_1552_ == 0)
{
lean_ctor_set(v___x_1551_, 0, v___x_1555_);
v___x_1557_ = v___x_1551_;
goto v_reusejp_1556_;
}
else
{
lean_object* v_reuseFailAlloc_1558_; 
v_reuseFailAlloc_1558_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1558_, 0, v___x_1555_);
v___x_1557_ = v_reuseFailAlloc_1558_;
goto v_reusejp_1556_;
}
v_reusejp_1556_:
{
return v___x_1557_;
}
}
else
{
lean_object* v___x_1559_; lean_object* v_a_1560_; lean_object* v___x_1561_; lean_object* v_a_1562_; lean_object* v___x_1564_; uint8_t v_isShared_1565_; uint8_t v_isSharedCheck_1572_; 
lean_del_object(v___x_1551_);
lean_dec(v_a_1549_);
v___x_1559_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_NormNum_evalNNRatCast_spec__0___redArg(v_a_1519_, v___y_1514_);
v_a_1560_ = lean_ctor_get(v___x_1559_, 0);
lean_inc(v_a_1560_);
lean_dec_ref(v___x_1559_);
v___x_1561_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_NormNum_evalNNRatCast_spec__0___redArg(v_a_1524_, v___y_1514_);
v_a_1562_ = lean_ctor_get(v___x_1561_, 0);
v_isSharedCheck_1572_ = !lean_is_exclusive(v___x_1561_);
if (v_isSharedCheck_1572_ == 0)
{
v___x_1564_ = v___x_1561_;
v_isShared_1565_ = v_isSharedCheck_1572_;
goto v_resetjp_1563_;
}
else
{
lean_inc(v_a_1562_);
lean_dec(v___x_1561_);
v___x_1564_ = lean_box(0);
v_isShared_1565_ = v_isSharedCheck_1572_;
goto v_resetjp_1563_;
}
v_resetjp_1563_:
{
lean_object* v___x_1566_; lean_object* v___x_1567_; lean_object* v___x_1568_; lean_object* v___x_1570_; 
v___x_1566_ = lean_box(v___x_1512_);
v___x_1567_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1567_, 0, v_a_1562_);
lean_ctor_set(v___x_1567_, 1, v___x_1566_);
v___x_1568_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1568_, 0, v_a_1560_);
lean_ctor_set(v___x_1568_, 1, v___x_1567_);
if (v_isShared_1565_ == 0)
{
lean_ctor_set(v___x_1564_, 0, v___x_1568_);
v___x_1570_ = v___x_1564_;
goto v_reusejp_1569_;
}
else
{
lean_object* v_reuseFailAlloc_1571_; 
v_reuseFailAlloc_1571_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1571_, 0, v___x_1568_);
v___x_1570_ = v_reuseFailAlloc_1571_;
goto v_reusejp_1569_;
}
v_reusejp_1569_:
{
return v___x_1570_;
}
}
}
}
}
else
{
lean_object* v_a_1574_; lean_object* v___x_1576_; uint8_t v_isShared_1577_; uint8_t v_isSharedCheck_1581_; 
lean_dec(v_a_1524_);
lean_dec(v_a_1519_);
v_a_1574_ = lean_ctor_get(v___x_1548_, 0);
v_isSharedCheck_1581_ = !lean_is_exclusive(v___x_1548_);
if (v_isSharedCheck_1581_ == 0)
{
v___x_1576_ = v___x_1548_;
v_isShared_1577_ = v_isSharedCheck_1581_;
goto v_resetjp_1575_;
}
else
{
lean_inc(v_a_1574_);
lean_dec(v___x_1548_);
v___x_1576_ = lean_box(0);
v_isShared_1577_ = v_isSharedCheck_1581_;
goto v_resetjp_1575_;
}
v_resetjp_1575_:
{
lean_object* v___x_1579_; 
if (v_isShared_1577_ == 0)
{
v___x_1579_ = v___x_1576_;
goto v_reusejp_1578_;
}
else
{
lean_object* v_reuseFailAlloc_1580_; 
v_reuseFailAlloc_1580_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1580_, 0, v_a_1574_);
v___x_1579_ = v_reuseFailAlloc_1580_;
goto v_reusejp_1578_;
}
v_reusejp_1578_:
{
return v___x_1579_;
}
}
}
}
}
}
else
{
lean_object* v_a_1584_; lean_object* v___x_1586_; uint8_t v_isShared_1587_; uint8_t v_isSharedCheck_1591_; 
lean_dec(v_a_1519_);
lean_dec_ref(v___y_1513_);
lean_dec_ref(v_e_1511_);
lean_dec_ref(v_00_u03b1_1510_);
lean_dec(v___x_1509_);
v_a_1584_ = lean_ctor_get(v___x_1523_, 0);
v_isSharedCheck_1591_ = !lean_is_exclusive(v___x_1523_);
if (v_isSharedCheck_1591_ == 0)
{
v___x_1586_ = v___x_1523_;
v_isShared_1587_ = v_isSharedCheck_1591_;
goto v_resetjp_1585_;
}
else
{
lean_inc(v_a_1584_);
lean_dec(v___x_1523_);
v___x_1586_ = lean_box(0);
v_isShared_1587_ = v_isSharedCheck_1591_;
goto v_resetjp_1585_;
}
v_resetjp_1585_:
{
lean_object* v___x_1589_; 
if (v_isShared_1587_ == 0)
{
v___x_1589_ = v___x_1586_;
goto v_reusejp_1588_;
}
else
{
lean_object* v_reuseFailAlloc_1590_; 
v_reuseFailAlloc_1590_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1590_, 0, v_a_1584_);
v___x_1589_ = v_reuseFailAlloc_1590_;
goto v_reusejp_1588_;
}
v_reusejp_1588_:
{
return v___x_1589_;
}
}
}
}
else
{
lean_object* v_a_1592_; lean_object* v___x_1594_; uint8_t v_isShared_1595_; uint8_t v_isSharedCheck_1599_; 
lean_dec_ref(v___y_1513_);
lean_dec_ref(v_e_1511_);
lean_dec_ref(v_00_u03b1_1510_);
lean_dec(v___x_1509_);
lean_dec(v___x_1508_);
lean_dec(v___x_1507_);
v_a_1592_ = lean_ctor_get(v___x_1518_, 0);
v_isSharedCheck_1599_ = !lean_is_exclusive(v___x_1518_);
if (v_isSharedCheck_1599_ == 0)
{
v___x_1594_ = v___x_1518_;
v_isShared_1595_ = v_isSharedCheck_1599_;
goto v_resetjp_1593_;
}
else
{
lean_inc(v_a_1592_);
lean_dec(v___x_1518_);
v___x_1594_ = lean_box(0);
v_isShared_1595_ = v_isSharedCheck_1599_;
goto v_resetjp_1593_;
}
v_resetjp_1593_:
{
lean_object* v___x_1597_; 
if (v_isShared_1595_ == 0)
{
v___x_1597_ = v___x_1594_;
goto v_reusejp_1596_;
}
else
{
lean_object* v_reuseFailAlloc_1598_; 
v_reuseFailAlloc_1598_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1598_, 0, v_a_1592_);
v___x_1597_ = v_reuseFailAlloc_1598_;
goto v_reusejp_1596_;
}
v_reusejp_1596_:
{
return v___x_1597_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__0___boxed(lean_object* v___x_1600_, lean_object* v___x_1601_, lean_object* v___x_1602_, lean_object* v___x_1603_, lean_object* v___x_1604_, lean_object* v_00_u03b1_1605_, lean_object* v_e_1606_, lean_object* v___x_1607_, lean_object* v___y_1608_, lean_object* v___y_1609_, lean_object* v___y_1610_, lean_object* v___y_1611_, lean_object* v___y_1612_){
_start:
{
uint8_t v___x_5449__boxed_1613_; uint8_t v___x_5453__boxed_1614_; lean_object* v_res_1615_; 
v___x_5449__boxed_1613_ = lean_unbox(v___x_1601_);
v___x_5453__boxed_1614_ = lean_unbox(v___x_1607_);
v_res_1615_ = lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__0(v___x_1600_, v___x_5449__boxed_1613_, v___x_1602_, v___x_1603_, v___x_1604_, v_00_u03b1_1605_, v_e_1606_, v___x_5453__boxed_1614_, v___y_1608_, v___y_1609_, v___y_1610_, v___y_1611_);
lean_dec(v___y_1611_);
lean_dec_ref(v___y_1610_);
lean_dec(v___y_1609_);
return v_res_1615_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1(uint8_t v___x_1635_, lean_object* v_u_1636_, lean_object* v_00_u03b1_1637_, lean_object* v_e_1638_, lean_object* v___y_1639_, lean_object* v___y_1640_, lean_object* v___y_1641_, lean_object* v___y_1642_){
_start:
{
lean_object* v___x_1644_; 
lean_inc_ref(v_00_u03b1_1637_);
lean_inc(v_u_1636_);
v___x_1644_ = lp_mathlib_Mathlib_Meta_NormNum_inferDivisionSemiring(v_u_1636_, v_00_u03b1_1637_, v___y_1639_, v___y_1640_, v___y_1641_, v___y_1642_);
if (lean_obj_tag(v___x_1644_) == 0)
{
lean_object* v_a_1645_; lean_object* v___x_1646_; lean_object* v___x_1647_; lean_object* v___x_1648_; lean_object* v___x_1649_; lean_object* v___x_1650_; lean_object* v___x_1651_; uint8_t v___x_1652_; lean_object* v___x_1653_; lean_object* v___x_1654_; lean_object* v___x_1655_; lean_object* v___f_1656_; uint8_t v___x_1657_; lean_object* v___x_1658_; 
v_a_1645_ = lean_ctor_get(v___x_1644_, 0);
lean_inc(v_a_1645_);
lean_dec_ref_known(v___x_1644_, 1);
v___x_1646_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__1));
v___x_1647_ = lean_box(0);
lean_inc(v_u_1636_);
v___x_1648_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1648_, 0, v_u_1636_);
lean_ctor_set(v___x_1648_, 1, v___x_1647_);
lean_inc_ref_n(v___x_1648_, 2);
v___x_1649_ = l_Lean_Expr_const___override(v___x_1646_, v___x_1648_);
lean_inc_ref_n(v_00_u03b1_1637_, 2);
v___x_1650_ = l_Lean_Expr_app___override(v___x_1649_, v_00_u03b1_1637_);
v___x_1651_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1651_, 0, v___x_1650_);
v___x_1652_ = 0;
v___x_1653_ = lean_box(0);
v___x_1654_ = lean_box(v___x_1652_);
v___x_1655_ = lean_box(v___x_1635_);
v___f_1656_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__0___boxed), 13, 8);
lean_closure_set(v___f_1656_, 0, v___x_1651_);
lean_closure_set(v___f_1656_, 1, v___x_1654_);
lean_closure_set(v___f_1656_, 2, v___x_1653_);
lean_closure_set(v___f_1656_, 3, v___x_1647_);
lean_closure_set(v___f_1656_, 4, v___x_1648_);
lean_closure_set(v___f_1656_, 5, v_00_u03b1_1637_);
lean_closure_set(v___f_1656_, 6, v_e_1638_);
lean_closure_set(v___f_1656_, 7, v___x_1655_);
v___x_1657_ = 0;
v___x_1658_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalRatCast_spec__0___redArg(v___f_1656_, v___x_1657_, v___y_1639_, v___y_1640_, v___y_1641_, v___y_1642_);
if (lean_obj_tag(v___x_1658_) == 0)
{
lean_object* v_a_1659_; lean_object* v_snd_1660_; lean_object* v_snd_1661_; uint8_t v___x_1662_; 
v_a_1659_ = lean_ctor_get(v___x_1658_, 0);
lean_inc(v_a_1659_);
lean_dec_ref_known(v___x_1658_, 1);
v_snd_1660_ = lean_ctor_get(v_a_1659_, 1);
lean_inc(v_snd_1660_);
v_snd_1661_ = lean_ctor_get(v_snd_1660_, 1);
v___x_1662_ = lean_unbox(v_snd_1661_);
if (v___x_1662_ == 0)
{
lean_object* v___x_1663_; lean_object* v___x_1664_; 
lean_dec(v_snd_1660_);
lean_dec(v_a_1659_);
lean_dec_ref_known(v___x_1648_, 2);
lean_dec(v_a_1645_);
lean_dec_ref(v_00_u03b1_1637_);
lean_dec(v_u_1636_);
v___x_1663_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1);
v___x_1664_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0___redArg(v___x_1663_, v___y_1639_, v___y_1640_, v___y_1641_, v___y_1642_);
return v___x_1664_;
}
else
{
lean_object* v_fst_1665_; lean_object* v_fst_1666_; lean_object* v___x_1668_; uint8_t v_isShared_1669_; uint8_t v_isSharedCheck_1794_; 
v_fst_1665_ = lean_ctor_get(v_a_1659_, 0);
lean_inc(v_fst_1665_);
lean_dec(v_a_1659_);
v_fst_1666_ = lean_ctor_get(v_snd_1660_, 0);
v_isSharedCheck_1794_ = !lean_is_exclusive(v_snd_1660_);
if (v_isSharedCheck_1794_ == 0)
{
lean_object* v_unused_1795_; 
v_unused_1795_ = lean_ctor_get(v_snd_1660_, 1);
lean_dec(v_unused_1795_);
v___x_1668_ = v_snd_1660_;
v_isShared_1669_ = v_isSharedCheck_1794_;
goto v_resetjp_1667_;
}
else
{
lean_inc(v_fst_1666_);
lean_dec(v_snd_1660_);
v___x_1668_ = lean_box(0);
v_isShared_1669_ = v_isSharedCheck_1794_;
goto v_resetjp_1667_;
}
v_resetjp_1667_:
{
lean_object* v___x_1670_; lean_object* v___x_1671_; lean_object* v___x_1672_; lean_object* v___x_1673_; lean_object* v___x_1674_; 
v___x_1670_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__3));
lean_inc_ref(v___x_1648_);
v___x_1671_ = l_Lean_Expr_const___override(v___x_1670_, v___x_1648_);
lean_inc_ref(v_00_u03b1_1637_);
v___x_1672_ = l_Lean_Expr_app___override(v___x_1671_, v_00_u03b1_1637_);
lean_inc(v_a_1645_);
v___x_1673_ = l_Lean_Expr_app___override(v___x_1672_, v_a_1645_);
v___x_1674_ = l_Lean_Meta_matchesInstance(v_fst_1665_, v___x_1673_, v___y_1639_, v___y_1640_, v___y_1641_, v___y_1642_);
if (lean_obj_tag(v___x_1674_) == 0)
{
lean_object* v_a_1675_; lean_object* v___x_1676_; uint8_t v___x_1775_; 
v_a_1675_ = lean_ctor_get(v___x_1674_, 0);
lean_inc(v_a_1675_);
lean_dec_ref_known(v___x_1674_, 1);
lean_inc(v_u_1636_);
v___x_1676_ = l_Lean_Level_succ___override(v_u_1636_);
v___x_1775_ = lean_unbox(v_a_1675_);
lean_dec(v_a_1675_);
if (v___x_1775_ == 0)
{
lean_object* v___x_1776_; lean_object* v___x_1777_; lean_object* v_a_1778_; lean_object* v___x_1780_; uint8_t v_isShared_1781_; uint8_t v_isSharedCheck_1785_; 
lean_dec(v___x_1676_);
lean_del_object(v___x_1668_);
lean_dec(v_fst_1666_);
lean_dec_ref_known(v___x_1648_, 2);
lean_dec(v_a_1645_);
lean_dec_ref(v_00_u03b1_1637_);
lean_dec(v_u_1636_);
v___x_1776_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1);
v___x_1777_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0___redArg(v___x_1776_, v___y_1639_, v___y_1640_, v___y_1641_, v___y_1642_);
v_a_1778_ = lean_ctor_get(v___x_1777_, 0);
v_isSharedCheck_1785_ = !lean_is_exclusive(v___x_1777_);
if (v_isSharedCheck_1785_ == 0)
{
v___x_1780_ = v___x_1777_;
v_isShared_1781_ = v_isSharedCheck_1785_;
goto v_resetjp_1779_;
}
else
{
lean_inc(v_a_1778_);
lean_dec(v___x_1777_);
v___x_1780_ = lean_box(0);
v_isShared_1781_ = v_isSharedCheck_1785_;
goto v_resetjp_1779_;
}
v_resetjp_1779_:
{
lean_object* v___x_1783_; 
if (v_isShared_1781_ == 0)
{
v___x_1783_ = v___x_1780_;
goto v_reusejp_1782_;
}
else
{
lean_object* v_reuseFailAlloc_1784_; 
v_reuseFailAlloc_1784_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1784_, 0, v_a_1778_);
v___x_1783_ = v_reuseFailAlloc_1784_;
goto v_reusejp_1782_;
}
v_reusejp_1782_:
{
return v___x_1783_;
}
}
}
else
{
goto v___jp_1677_;
}
v___jp_1677_:
{
lean_object* v___x_1678_; lean_object* v___x_1679_; lean_object* v___x_1680_; 
v___x_1678_ = lean_box(0);
v___x_1679_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5, &lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNNRatDivNat___lam__0___closed__5);
lean_inc(v_fst_1666_);
v___x_1680_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v___x_1678_, v___x_1679_, v_fst_1666_, v___x_1657_, v___y_1639_, v___y_1640_, v___y_1641_, v___y_1642_);
if (lean_obj_tag(v___x_1680_) == 0)
{
lean_object* v_a_1681_; lean_object* v___x_1683_; uint8_t v_isShared_1684_; uint8_t v_isSharedCheck_1774_; 
v_a_1681_ = lean_ctor_get(v___x_1680_, 0);
v_isSharedCheck_1774_ = !lean_is_exclusive(v___x_1680_);
if (v_isSharedCheck_1774_ == 0)
{
v___x_1683_ = v___x_1680_;
v_isShared_1684_ = v_isSharedCheck_1774_;
goto v_resetjp_1682_;
}
else
{
lean_inc(v_a_1681_);
lean_dec(v___x_1680_);
v___x_1683_ = lean_box(0);
v_isShared_1684_ = v_isSharedCheck_1774_;
goto v_resetjp_1682_;
}
v_resetjp_1682_:
{
switch(lean_obj_tag(v_a_1681_))
{
case 1:
{
lean_object* v_lit_1685_; lean_object* v_proof_1686_; lean_object* v___x_1688_; uint8_t v_isShared_1689_; uint8_t v_isSharedCheck_1719_; 
lean_dec(v___x_1676_);
lean_del_object(v___x_1668_);
lean_dec(v_u_1636_);
v_lit_1685_ = lean_ctor_get(v_a_1681_, 1);
v_proof_1686_ = lean_ctor_get(v_a_1681_, 2);
v_isSharedCheck_1719_ = !lean_is_exclusive(v_a_1681_);
if (v_isSharedCheck_1719_ == 0)
{
lean_object* v_unused_1720_; 
v_unused_1720_ = lean_ctor_get(v_a_1681_, 0);
lean_dec(v_unused_1720_);
v___x_1688_ = v_a_1681_;
v_isShared_1689_ = v_isSharedCheck_1719_;
goto v_resetjp_1687_;
}
else
{
lean_inc(v_proof_1686_);
lean_inc(v_lit_1685_);
lean_dec(v_a_1681_);
v___x_1688_ = lean_box(0);
v_isShared_1689_ = v_isSharedCheck_1719_;
goto v_resetjp_1687_;
}
v_resetjp_1687_:
{
lean_object* v___x_1690_; lean_object* v___x_1691_; lean_object* v___x_1692_; lean_object* v___x_1693_; lean_object* v___x_1694_; lean_object* v___x_1695_; lean_object* v___x_1696_; lean_object* v___x_1697_; lean_object* v___x_1698_; lean_object* v___x_1699_; lean_object* v___x_1700_; lean_object* v___x_1701_; lean_object* v___x_1702_; lean_object* v___x_1703_; lean_object* v___x_1704_; lean_object* v___x_1705_; lean_object* v___x_1706_; lean_object* v___x_1707_; lean_object* v___x_1708_; lean_object* v___x_1709_; lean_object* v___x_1710_; lean_object* v___x_1711_; lean_object* v___x_1712_; lean_object* v___x_1714_; 
v___x_1690_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__1));
lean_inc_ref_n(v___x_1648_, 4);
v___x_1691_ = l_Lean_Expr_const___override(v___x_1690_, v___x_1648_);
lean_inc_ref_n(v_00_u03b1_1637_, 4);
v___x_1692_ = l_Lean_Expr_app___override(v___x_1691_, v_00_u03b1_1637_);
v___x_1693_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__4));
v___x_1694_ = l_Lean_Expr_const___override(v___x_1693_, v___x_1648_);
v___x_1695_ = l_Lean_Expr_app___override(v___x_1694_, v_00_u03b1_1637_);
v___x_1696_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__7));
v___x_1697_ = l_Lean_Expr_const___override(v___x_1696_, v___x_1648_);
v___x_1698_ = l_Lean_Expr_app___override(v___x_1697_, v_00_u03b1_1637_);
v___x_1699_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___closed__10));
v___x_1700_ = l_Lean_Expr_const___override(v___x_1699_, v___x_1648_);
v___x_1701_ = l_Lean_Expr_app___override(v___x_1700_, v_00_u03b1_1637_);
lean_inc(v_a_1645_);
v___x_1702_ = l_Lean_Expr_app___override(v___x_1701_, v_a_1645_);
v___x_1703_ = l_Lean_Expr_app___override(v___x_1698_, v___x_1702_);
v___x_1704_ = l_Lean_Expr_app___override(v___x_1695_, v___x_1703_);
v___x_1705_ = l_Lean_Expr_app___override(v___x_1692_, v___x_1704_);
v___x_1706_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__5));
v___x_1707_ = l_Lean_Expr_const___override(v___x_1706_, v___x_1648_);
v___x_1708_ = l_Lean_Expr_app___override(v___x_1707_, v_00_u03b1_1637_);
v___x_1709_ = l_Lean_Expr_app___override(v___x_1708_, v_a_1645_);
v___x_1710_ = l_Lean_Expr_app___override(v___x_1709_, v_fst_1666_);
lean_inc_ref(v_lit_1685_);
v___x_1711_ = l_Lean_Expr_app___override(v___x_1710_, v_lit_1685_);
v___x_1712_ = l_Lean_Expr_app___override(v___x_1711_, v_proof_1686_);
if (v_isShared_1689_ == 0)
{
lean_ctor_set(v___x_1688_, 2, v___x_1712_);
lean_ctor_set(v___x_1688_, 0, v___x_1705_);
v___x_1714_ = v___x_1688_;
goto v_reusejp_1713_;
}
else
{
lean_object* v_reuseFailAlloc_1718_; 
v_reuseFailAlloc_1718_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1718_, 0, v___x_1705_);
lean_ctor_set(v_reuseFailAlloc_1718_, 1, v_lit_1685_);
lean_ctor_set(v_reuseFailAlloc_1718_, 2, v___x_1712_);
v___x_1714_ = v_reuseFailAlloc_1718_;
goto v_reusejp_1713_;
}
v_reusejp_1713_:
{
lean_object* v___x_1716_; 
if (v_isShared_1684_ == 0)
{
lean_ctor_set(v___x_1683_, 0, v___x_1714_);
v___x_1716_ = v___x_1683_;
goto v_reusejp_1715_;
}
else
{
lean_object* v_reuseFailAlloc_1717_; 
v_reuseFailAlloc_1717_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1717_, 0, v___x_1714_);
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
case 3:
{
lean_object* v_q_1721_; lean_object* v_n_1722_; lean_object* v_d_1723_; lean_object* v_proof_1724_; lean_object* v___x_1726_; uint8_t v_isShared_1727_; uint8_t v_isSharedCheck_1770_; 
lean_del_object(v___x_1683_);
v_q_1721_ = lean_ctor_get(v_a_1681_, 1);
v_n_1722_ = lean_ctor_get(v_a_1681_, 2);
v_d_1723_ = lean_ctor_get(v_a_1681_, 3);
v_proof_1724_ = lean_ctor_get(v_a_1681_, 4);
v_isSharedCheck_1770_ = !lean_is_exclusive(v_a_1681_);
if (v_isSharedCheck_1770_ == 0)
{
lean_object* v_unused_1771_; 
v_unused_1771_ = lean_ctor_get(v_a_1681_, 0);
lean_dec(v_unused_1771_);
v___x_1726_ = v_a_1681_;
v_isShared_1727_ = v_isSharedCheck_1770_;
goto v_resetjp_1725_;
}
else
{
lean_inc(v_proof_1724_);
lean_inc(v_d_1723_);
lean_inc(v_n_1722_);
lean_inc(v_q_1721_);
lean_dec(v_a_1681_);
v___x_1726_ = lean_box(0);
v_isShared_1727_ = v_isSharedCheck_1770_;
goto v_resetjp_1725_;
}
v_resetjp_1725_:
{
lean_object* v___x_1728_; 
lean_inc(v_a_1645_);
lean_inc_ref(v_00_u03b1_1637_);
v___x_1728_ = lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f(v_u_1636_, v_00_u03b1_1637_, v_a_1645_, v___y_1639_, v___y_1640_, v___y_1641_, v___y_1642_);
if (lean_obj_tag(v___x_1728_) == 0)
{
lean_object* v_a_1729_; lean_object* v___x_1731_; uint8_t v_isShared_1732_; uint8_t v_isSharedCheck_1761_; 
v_a_1729_ = lean_ctor_get(v___x_1728_, 0);
v_isSharedCheck_1761_ = !lean_is_exclusive(v___x_1728_);
if (v_isSharedCheck_1761_ == 0)
{
v___x_1731_ = v___x_1728_;
v_isShared_1732_ = v_isSharedCheck_1761_;
goto v_resetjp_1730_;
}
else
{
lean_inc(v_a_1729_);
lean_dec(v___x_1728_);
v___x_1731_ = lean_box(0);
v_isShared_1732_ = v_isSharedCheck_1761_;
goto v_resetjp_1730_;
}
v_resetjp_1730_:
{
if (lean_obj_tag(v_a_1729_) == 1)
{
lean_object* v_val_1733_; lean_object* v___x_1734_; lean_object* v___x_1735_; lean_object* v___x_1736_; lean_object* v___x_1737_; lean_object* v___x_1739_; 
v_val_1733_ = lean_ctor_get(v_a_1729_, 0);
lean_inc(v_val_1733_);
lean_dec_ref_known(v_a_1729_, 1);
v___x_1734_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__10));
lean_inc_ref(v___x_1648_);
v___x_1735_ = l_Lean_Expr_const___override(v___x_1734_, v___x_1648_);
lean_inc_ref(v_00_u03b1_1637_);
v___x_1736_ = l_Lean_Expr_app___override(v___x_1735_, v_00_u03b1_1637_);
v___x_1737_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__12));
if (v_isShared_1669_ == 0)
{
lean_ctor_set_tag(v___x_1668_, 1);
lean_ctor_set(v___x_1668_, 1, v___x_1647_);
lean_ctor_set(v___x_1668_, 0, v___x_1676_);
v___x_1739_ = v___x_1668_;
goto v_reusejp_1738_;
}
else
{
lean_object* v_reuseFailAlloc_1758_; 
v_reuseFailAlloc_1758_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1758_, 0, v___x_1676_);
lean_ctor_set(v_reuseFailAlloc_1758_, 1, v___x_1647_);
v___x_1739_ = v_reuseFailAlloc_1758_;
goto v_reusejp_1738_;
}
v_reusejp_1738_:
{
lean_object* v___x_1740_; lean_object* v___x_1741_; lean_object* v___x_1742_; lean_object* v___x_1743_; lean_object* v___x_1744_; lean_object* v___x_1745_; lean_object* v___x_1746_; lean_object* v___x_1747_; lean_object* v___x_1748_; lean_object* v___x_1749_; lean_object* v___x_1750_; lean_object* v___x_1751_; lean_object* v___x_1753_; 
v___x_1740_ = l_Lean_Expr_const___override(v___x_1737_, v___x_1739_);
v___x_1741_ = l_Lean_Expr_app___override(v___x_1740_, v___x_1736_);
lean_inc(v_a_1645_);
v___x_1742_ = l_Lean_Expr_app___override(v___x_1741_, v_a_1645_);
v___x_1743_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___closed__7));
v___x_1744_ = l_Lean_Expr_const___override(v___x_1743_, v___x_1648_);
v___x_1745_ = l_Lean_Expr_app___override(v___x_1744_, v_00_u03b1_1637_);
v___x_1746_ = l_Lean_Expr_app___override(v___x_1745_, v_a_1645_);
v___x_1747_ = l_Lean_Expr_app___override(v___x_1746_, v_val_1733_);
v___x_1748_ = l_Lean_Expr_app___override(v___x_1747_, v_fst_1666_);
lean_inc_ref(v_n_1722_);
v___x_1749_ = l_Lean_Expr_app___override(v___x_1748_, v_n_1722_);
lean_inc_ref(v_d_1723_);
v___x_1750_ = l_Lean_Expr_app___override(v___x_1749_, v_d_1723_);
v___x_1751_ = l_Lean_Expr_app___override(v___x_1750_, v_proof_1724_);
if (v_isShared_1727_ == 0)
{
lean_ctor_set(v___x_1726_, 4, v___x_1751_);
lean_ctor_set(v___x_1726_, 0, v___x_1742_);
v___x_1753_ = v___x_1726_;
goto v_reusejp_1752_;
}
else
{
lean_object* v_reuseFailAlloc_1757_; 
v_reuseFailAlloc_1757_ = lean_alloc_ctor(3, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1757_, 0, v___x_1742_);
lean_ctor_set(v_reuseFailAlloc_1757_, 1, v_q_1721_);
lean_ctor_set(v_reuseFailAlloc_1757_, 2, v_n_1722_);
lean_ctor_set(v_reuseFailAlloc_1757_, 3, v_d_1723_);
lean_ctor_set(v_reuseFailAlloc_1757_, 4, v___x_1751_);
v___x_1753_ = v_reuseFailAlloc_1757_;
goto v_reusejp_1752_;
}
v_reusejp_1752_:
{
lean_object* v___x_1755_; 
if (v_isShared_1732_ == 0)
{
lean_ctor_set(v___x_1731_, 0, v___x_1753_);
v___x_1755_ = v___x_1731_;
goto v_reusejp_1754_;
}
else
{
lean_object* v_reuseFailAlloc_1756_; 
v_reuseFailAlloc_1756_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1756_, 0, v___x_1753_);
v___x_1755_ = v_reuseFailAlloc_1756_;
goto v_reusejp_1754_;
}
v_reusejp_1754_:
{
return v___x_1755_;
}
}
}
}
else
{
lean_object* v___x_1759_; lean_object* v___x_1760_; 
lean_del_object(v___x_1731_);
lean_dec(v_a_1729_);
lean_del_object(v___x_1726_);
lean_dec_ref(v_proof_1724_);
lean_dec_ref(v_d_1723_);
lean_dec_ref(v_n_1722_);
lean_dec_ref(v_q_1721_);
lean_dec(v___x_1676_);
lean_del_object(v___x_1668_);
lean_dec(v_fst_1666_);
lean_dec_ref_known(v___x_1648_, 2);
lean_dec(v_a_1645_);
lean_dec_ref(v_00_u03b1_1637_);
v___x_1759_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1);
v___x_1760_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0___redArg(v___x_1759_, v___y_1639_, v___y_1640_, v___y_1641_, v___y_1642_);
return v___x_1760_;
}
}
}
else
{
lean_object* v_a_1762_; lean_object* v___x_1764_; uint8_t v_isShared_1765_; uint8_t v_isSharedCheck_1769_; 
lean_del_object(v___x_1726_);
lean_dec_ref(v_proof_1724_);
lean_dec_ref(v_d_1723_);
lean_dec_ref(v_n_1722_);
lean_dec_ref(v_q_1721_);
lean_dec(v___x_1676_);
lean_del_object(v___x_1668_);
lean_dec(v_fst_1666_);
lean_dec_ref_known(v___x_1648_, 2);
lean_dec(v_a_1645_);
lean_dec_ref(v_00_u03b1_1637_);
v_a_1762_ = lean_ctor_get(v___x_1728_, 0);
v_isSharedCheck_1769_ = !lean_is_exclusive(v___x_1728_);
if (v_isSharedCheck_1769_ == 0)
{
v___x_1764_ = v___x_1728_;
v_isShared_1765_ = v_isSharedCheck_1769_;
goto v_resetjp_1763_;
}
else
{
lean_inc(v_a_1762_);
lean_dec(v___x_1728_);
v___x_1764_ = lean_box(0);
v_isShared_1765_ = v_isSharedCheck_1769_;
goto v_resetjp_1763_;
}
v_resetjp_1763_:
{
lean_object* v___x_1767_; 
if (v_isShared_1765_ == 0)
{
v___x_1767_ = v___x_1764_;
goto v_reusejp_1766_;
}
else
{
lean_object* v_reuseFailAlloc_1768_; 
v_reuseFailAlloc_1768_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1768_, 0, v_a_1762_);
v___x_1767_ = v_reuseFailAlloc_1768_;
goto v_reusejp_1766_;
}
v_reusejp_1766_:
{
return v___x_1767_;
}
}
}
}
}
default: 
{
lean_object* v___x_1772_; lean_object* v___x_1773_; 
lean_del_object(v___x_1683_);
lean_dec(v_a_1681_);
lean_dec(v___x_1676_);
lean_del_object(v___x_1668_);
lean_dec(v_fst_1666_);
lean_dec_ref_known(v___x_1648_, 2);
lean_dec(v_a_1645_);
lean_dec_ref(v_00_u03b1_1637_);
lean_dec(v_u_1636_);
v___x_1772_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1);
v___x_1773_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0___redArg(v___x_1772_, v___y_1639_, v___y_1640_, v___y_1641_, v___y_1642_);
return v___x_1773_;
}
}
}
}
else
{
lean_dec(v___x_1676_);
lean_del_object(v___x_1668_);
lean_dec(v_fst_1666_);
lean_dec_ref_known(v___x_1648_, 2);
lean_dec(v_a_1645_);
lean_dec_ref(v_00_u03b1_1637_);
lean_dec(v_u_1636_);
return v___x_1680_;
}
}
}
else
{
lean_object* v_a_1786_; lean_object* v___x_1788_; uint8_t v_isShared_1789_; uint8_t v_isSharedCheck_1793_; 
lean_del_object(v___x_1668_);
lean_dec(v_fst_1666_);
lean_dec_ref_known(v___x_1648_, 2);
lean_dec(v_a_1645_);
lean_dec_ref(v_00_u03b1_1637_);
lean_dec(v_u_1636_);
v_a_1786_ = lean_ctor_get(v___x_1674_, 0);
v_isSharedCheck_1793_ = !lean_is_exclusive(v___x_1674_);
if (v_isSharedCheck_1793_ == 0)
{
v___x_1788_ = v___x_1674_;
v_isShared_1789_ = v_isSharedCheck_1793_;
goto v_resetjp_1787_;
}
else
{
lean_inc(v_a_1786_);
lean_dec(v___x_1674_);
v___x_1788_ = lean_box(0);
v_isShared_1789_ = v_isSharedCheck_1793_;
goto v_resetjp_1787_;
}
v_resetjp_1787_:
{
lean_object* v___x_1791_; 
if (v_isShared_1789_ == 0)
{
v___x_1791_ = v___x_1788_;
goto v_reusejp_1790_;
}
else
{
lean_object* v_reuseFailAlloc_1792_; 
v_reuseFailAlloc_1792_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1792_, 0, v_a_1786_);
v___x_1791_ = v_reuseFailAlloc_1792_;
goto v_reusejp_1790_;
}
v_reusejp_1790_:
{
return v___x_1791_;
}
}
}
}
}
}
else
{
lean_object* v_a_1796_; lean_object* v___x_1798_; uint8_t v_isShared_1799_; uint8_t v_isSharedCheck_1803_; 
lean_dec_ref_known(v___x_1648_, 2);
lean_dec(v_a_1645_);
lean_dec_ref(v_00_u03b1_1637_);
lean_dec(v_u_1636_);
v_a_1796_ = lean_ctor_get(v___x_1658_, 0);
v_isSharedCheck_1803_ = !lean_is_exclusive(v___x_1658_);
if (v_isSharedCheck_1803_ == 0)
{
v___x_1798_ = v___x_1658_;
v_isShared_1799_ = v_isSharedCheck_1803_;
goto v_resetjp_1797_;
}
else
{
lean_inc(v_a_1796_);
lean_dec(v___x_1658_);
v___x_1798_ = lean_box(0);
v_isShared_1799_ = v_isSharedCheck_1803_;
goto v_resetjp_1797_;
}
v_resetjp_1797_:
{
lean_object* v___x_1801_; 
if (v_isShared_1799_ == 0)
{
v___x_1801_ = v___x_1798_;
goto v_reusejp_1800_;
}
else
{
lean_object* v_reuseFailAlloc_1802_; 
v_reuseFailAlloc_1802_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1802_, 0, v_a_1796_);
v___x_1801_ = v_reuseFailAlloc_1802_;
goto v_reusejp_1800_;
}
v_reusejp_1800_:
{
return v___x_1801_;
}
}
}
}
else
{
lean_object* v_a_1804_; lean_object* v___x_1806_; uint8_t v_isShared_1807_; uint8_t v_isSharedCheck_1811_; 
lean_dec_ref(v_e_1638_);
lean_dec_ref(v_00_u03b1_1637_);
lean_dec(v_u_1636_);
v_a_1804_ = lean_ctor_get(v___x_1644_, 0);
v_isSharedCheck_1811_ = !lean_is_exclusive(v___x_1644_);
if (v_isSharedCheck_1811_ == 0)
{
v___x_1806_ = v___x_1644_;
v_isShared_1807_ = v_isSharedCheck_1811_;
goto v_resetjp_1805_;
}
else
{
lean_inc(v_a_1804_);
lean_dec(v___x_1644_);
v___x_1806_ = lean_box(0);
v_isShared_1807_ = v_isSharedCheck_1811_;
goto v_resetjp_1805_;
}
v_resetjp_1805_:
{
lean_object* v___x_1809_; 
if (v_isShared_1807_ == 0)
{
v___x_1809_ = v___x_1806_;
goto v_reusejp_1808_;
}
else
{
lean_object* v_reuseFailAlloc_1810_; 
v_reuseFailAlloc_1810_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1810_, 0, v_a_1804_);
v___x_1809_ = v_reuseFailAlloc_1810_;
goto v_reusejp_1808_;
}
v_reusejp_1808_:
{
return v___x_1809_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1___boxed(lean_object* v___x_1812_, lean_object* v_u_1813_, lean_object* v_00_u03b1_1814_, lean_object* v_e_1815_, lean_object* v___y_1816_, lean_object* v___y_1817_, lean_object* v___y_1818_, lean_object* v___y_1819_, lean_object* v___y_1820_){
_start:
{
uint8_t v___x_5715__boxed_1821_; lean_object* v_res_1822_; 
v___x_5715__boxed_1821_ = lean_unbox(v___x_1812_);
v_res_1822_ = lp_mathlib_Mathlib_Meta_NormNum_evalNNRatCast___lam__1(v___x_5715__boxed_1821_, v_u_1813_, v_00_u03b1_1814_, v_e_1815_, v___y_1816_, v___y_1817_, v___y_1818_, v___y_1819_);
lean_dec(v___y_1819_);
lean_dec_ref(v___y_1818_);
lean_dec(v___y_1817_);
lean_dec_ref(v___y_1816_);
return v_res_1822_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___lam__0(uint8_t v___x_1837_, lean_object* v_ds_u03b1_1838_, lean_object* v___x_1839_, lean_object* v___y_1840_, lean_object* v___y_1841_, lean_object* v___y_1842_, lean_object* v___y_1843_){
_start:
{
lean_object* v_keyedConfig_1845_; uint8_t v_trackZetaDelta_1846_; lean_object* v_zetaDeltaSet_1847_; lean_object* v_lctx_1848_; lean_object* v_localInstances_1849_; lean_object* v_defEqCtx_x3f_1850_; lean_object* v_synthPendingDepth_1851_; lean_object* v_customCanUnfoldPredicate_x3f_1852_; uint8_t v_univApprox_1853_; uint8_t v_inTypeClassResolution_1854_; uint8_t v_cacheInferType_1855_; lean_object* v___x_1857_; uint8_t v_isShared_1858_; uint8_t v_isSharedCheck_1864_; 
v_keyedConfig_1845_ = lean_ctor_get(v___y_1840_, 0);
v_trackZetaDelta_1846_ = lean_ctor_get_uint8(v___y_1840_, sizeof(void*)*7);
v_zetaDeltaSet_1847_ = lean_ctor_get(v___y_1840_, 1);
v_lctx_1848_ = lean_ctor_get(v___y_1840_, 2);
v_localInstances_1849_ = lean_ctor_get(v___y_1840_, 3);
v_defEqCtx_x3f_1850_ = lean_ctor_get(v___y_1840_, 4);
v_synthPendingDepth_1851_ = lean_ctor_get(v___y_1840_, 5);
v_customCanUnfoldPredicate_x3f_1852_ = lean_ctor_get(v___y_1840_, 6);
v_univApprox_1853_ = lean_ctor_get_uint8(v___y_1840_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1854_ = lean_ctor_get_uint8(v___y_1840_, sizeof(void*)*7 + 2);
v_cacheInferType_1855_ = lean_ctor_get_uint8(v___y_1840_, sizeof(void*)*7 + 3);
v_isSharedCheck_1864_ = !lean_is_exclusive(v___y_1840_);
if (v_isSharedCheck_1864_ == 0)
{
v___x_1857_ = v___y_1840_;
v_isShared_1858_ = v_isSharedCheck_1864_;
goto v_resetjp_1856_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_1852_);
lean_inc(v_synthPendingDepth_1851_);
lean_inc(v_defEqCtx_x3f_1850_);
lean_inc(v_localInstances_1849_);
lean_inc(v_lctx_1848_);
lean_inc(v_zetaDeltaSet_1847_);
lean_inc(v_keyedConfig_1845_);
lean_dec(v___y_1840_);
v___x_1857_ = lean_box(0);
v_isShared_1858_ = v_isSharedCheck_1864_;
goto v_resetjp_1856_;
}
v_resetjp_1856_:
{
lean_object* v___x_1859_; lean_object* v___x_1861_; 
v___x_1859_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1837_, v_keyedConfig_1845_);
if (v_isShared_1858_ == 0)
{
lean_ctor_set(v___x_1857_, 0, v___x_1859_);
v___x_1861_ = v___x_1857_;
goto v_reusejp_1860_;
}
else
{
lean_object* v_reuseFailAlloc_1863_; 
v_reuseFailAlloc_1863_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_1863_, 0, v___x_1859_);
lean_ctor_set(v_reuseFailAlloc_1863_, 1, v_zetaDeltaSet_1847_);
lean_ctor_set(v_reuseFailAlloc_1863_, 2, v_lctx_1848_);
lean_ctor_set(v_reuseFailAlloc_1863_, 3, v_localInstances_1849_);
lean_ctor_set(v_reuseFailAlloc_1863_, 4, v_defEqCtx_x3f_1850_);
lean_ctor_set(v_reuseFailAlloc_1863_, 5, v_synthPendingDepth_1851_);
lean_ctor_set(v_reuseFailAlloc_1863_, 6, v_customCanUnfoldPredicate_x3f_1852_);
lean_ctor_set_uint8(v_reuseFailAlloc_1863_, sizeof(void*)*7, v_trackZetaDelta_1846_);
lean_ctor_set_uint8(v_reuseFailAlloc_1863_, sizeof(void*)*7 + 1, v_univApprox_1853_);
lean_ctor_set_uint8(v_reuseFailAlloc_1863_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1854_);
lean_ctor_set_uint8(v_reuseFailAlloc_1863_, sizeof(void*)*7 + 3, v_cacheInferType_1855_);
v___x_1861_ = v_reuseFailAlloc_1863_;
goto v_reusejp_1860_;
}
v_reusejp_1860_:
{
lean_object* v___x_1862_; 
v___x_1862_ = lp_Qq_Qq_assertDefEqQ___redArg(v_ds_u03b1_1838_, v___x_1839_, v___x_1861_, v___y_1841_, v___y_1842_, v___y_1843_);
lean_dec_ref(v___x_1861_);
return v___x_1862_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___lam__0___boxed(lean_object* v___x_1865_, lean_object* v_ds_u03b1_1866_, lean_object* v___x_1867_, lean_object* v___y_1868_, lean_object* v___y_1869_, lean_object* v___y_1870_, lean_object* v___y_1871_, lean_object* v___y_1872_){
_start:
{
uint8_t v___x_6378__boxed_1873_; lean_object* v_res_1874_; 
v___x_6378__boxed_1873_ = lean_unbox(v___x_1865_);
v_res_1874_ = lp_mathlib_Mathlib_Meta_NormNum_Result_inv___lam__0(v___x_6378__boxed_1873_, v_ds_u03b1_1866_, v___x_1867_, v___y_1868_, v___y_1869_, v___y_1870_, v___y_1871_);
lean_dec(v___y_1871_);
lean_dec_ref(v___y_1870_);
lean_dec(v___y_1869_);
return v_res_1874_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Mathlib_Meta_NormNum_Result_inv_spec__0(lean_object* v_a_1875_){
_start:
{
lean_object* v___x_1876_; lean_object* v___x_1877_; 
v___x_1876_ = lean_nat_to_int(v_a_1875_);
v___x_1877_ = l_Rat_ofInt(v___x_1876_);
return v___x_1877_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__2(void){
_start:
{
lean_object* v___x_1884_; lean_object* v___x_1885_; 
v___x_1884_ = lean_unsigned_to_nat(0u);
v___x_1885_ = lp_mathlib_Nat_cast___at___00Mathlib_Meta_NormNum_Result_inv_spec__0(v___x_1884_);
return v___x_1885_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__24(void){
_start:
{
lean_object* v___x_1927_; lean_object* v___x_1928_; 
v___x_1927_ = lean_unsigned_to_nat(1u);
v___x_1928_ = lp_mathlib_Nat_cast___at___00Mathlib_Meta_NormNum_Result_inv_spec__0(v___x_1927_);
return v___x_1928_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__29(void){
_start:
{
lean_object* v___x_1941_; lean_object* v___x_1942_; 
v___x_1941_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__24, &lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__24_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__24);
v___x_1942_ = l_Rat_neg(v___x_1941_);
return v___x_1942_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv(lean_object* v_u_1943_, lean_object* v_00_u03b1_1944_, lean_object* v_a_1945_, lean_object* v_ra_1946_, lean_object* v_ds_u03b1_1947_, lean_object* v_cz_u03b1_x3f_1948_, lean_object* v_a_1949_, lean_object* v_a_1950_, lean_object* v_a_1951_, lean_object* v_a_1952_){
_start:
{
lean_object* v___x_1976_; 
lean_inc_ref(v_ra_1946_);
lean_inc_ref(v_ds_u03b1_1947_);
lean_inc_ref(v_a_1945_);
lean_inc_ref(v_00_u03b1_1944_);
lean_inc(v_u_1943_);
v___x_1976_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toNNRat_x27(v_u_1943_, v_00_u03b1_1944_, v_a_1945_, v_ds_u03b1_1947_, v_ra_1946_);
if (lean_obj_tag(v___x_1976_) == 1)
{
lean_object* v_val_1977_; lean_object* v___x_1979_; uint8_t v_isShared_1980_; uint8_t v_isSharedCheck_2082_; 
v_val_1977_ = lean_ctor_get(v___x_1976_, 0);
v_isSharedCheck_2082_ = !lean_is_exclusive(v___x_1976_);
if (v_isSharedCheck_2082_ == 0)
{
v___x_1979_ = v___x_1976_;
v_isShared_1980_ = v_isSharedCheck_2082_;
goto v_resetjp_1978_;
}
else
{
lean_inc(v_val_1977_);
lean_dec(v___x_1976_);
v___x_1979_ = lean_box(0);
v_isShared_1980_ = v_isSharedCheck_2082_;
goto v_resetjp_1978_;
}
v_resetjp_1978_:
{
lean_object* v_snd_1981_; lean_object* v_snd_1982_; lean_object* v_fst_1983_; lean_object* v_fst_1984_; lean_object* v_fst_1985_; lean_object* v_snd_1986_; lean_object* v___x_1988_; uint8_t v_isShared_1989_; uint8_t v_isSharedCheck_2081_; 
v_snd_1981_ = lean_ctor_get(v_val_1977_, 1);
lean_inc(v_snd_1981_);
v_snd_1982_ = lean_ctor_get(v_snd_1981_, 1);
lean_inc(v_snd_1982_);
v_fst_1983_ = lean_ctor_get(v_val_1977_, 0);
lean_inc(v_fst_1983_);
lean_dec(v_val_1977_);
v_fst_1984_ = lean_ctor_get(v_snd_1981_, 0);
lean_inc(v_fst_1984_);
lean_dec(v_snd_1981_);
v_fst_1985_ = lean_ctor_get(v_snd_1982_, 0);
v_snd_1986_ = lean_ctor_get(v_snd_1982_, 1);
v_isSharedCheck_2081_ = !lean_is_exclusive(v_snd_1982_);
if (v_isSharedCheck_2081_ == 0)
{
v___x_1988_ = v_snd_1982_;
v_isShared_1989_ = v_isSharedCheck_2081_;
goto v_resetjp_1987_;
}
else
{
lean_inc(v_snd_1986_);
lean_inc(v_fst_1985_);
lean_dec(v_snd_1982_);
v___x_1988_ = lean_box(0);
v_isShared_1989_ = v_isSharedCheck_2081_;
goto v_resetjp_1987_;
}
v_resetjp_1987_:
{
lean_object* v___x_1990_; uint8_t v___x_1991_; 
v___x_1990_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__2, &lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__2_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__2);
lean_inc(v_fst_1983_);
v___x_1991_ = l_Rat_blt(v___x_1990_, v_fst_1983_);
if (v___x_1991_ == 0)
{
lean_dec(v_snd_1986_);
lean_dec(v_fst_1985_);
lean_dec(v_fst_1984_);
lean_dec(v_fst_1983_);
lean_dec(v_cz_u03b1_x3f_1948_);
if (lean_obj_tag(v_ra_1946_) == 1)
{
lean_object* v_inst_1992_; lean_object* v_lit_1993_; lean_object* v_proof_1994_; lean_object* v___x_1996_; uint8_t v_isShared_1997_; uint8_t v_isSharedCheck_2014_; 
v_inst_1992_ = lean_ctor_get(v_ra_1946_, 0);
v_lit_1993_ = lean_ctor_get(v_ra_1946_, 1);
v_proof_1994_ = lean_ctor_get(v_ra_1946_, 2);
v_isSharedCheck_2014_ = !lean_is_exclusive(v_ra_1946_);
if (v_isSharedCheck_2014_ == 0)
{
v___x_1996_ = v_ra_1946_;
v_isShared_1997_ = v_isSharedCheck_2014_;
goto v_resetjp_1995_;
}
else
{
lean_inc(v_proof_1994_);
lean_inc(v_lit_1993_);
lean_inc(v_inst_1992_);
lean_dec(v_ra_1946_);
v___x_1996_ = lean_box(0);
v_isShared_1997_ = v_isSharedCheck_2014_;
goto v_resetjp_1995_;
}
v_resetjp_1995_:
{
lean_object* v___x_1998_; lean_object* v___x_2000_; 
v___x_1998_ = lean_box(0);
if (v_isShared_1989_ == 0)
{
lean_ctor_set_tag(v___x_1988_, 1);
lean_ctor_set(v___x_1988_, 1, v___x_1998_);
lean_ctor_set(v___x_1988_, 0, v_u_1943_);
v___x_2000_ = v___x_1988_;
goto v_reusejp_1999_;
}
else
{
lean_object* v_reuseFailAlloc_2013_; 
v_reuseFailAlloc_2013_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2013_, 0, v_u_1943_);
lean_ctor_set(v_reuseFailAlloc_2013_, 1, v___x_1998_);
v___x_2000_ = v_reuseFailAlloc_2013_;
goto v_reusejp_1999_;
}
v_reusejp_1999_:
{
lean_object* v___x_2001_; lean_object* v___x_2002_; lean_object* v___x_2003_; lean_object* v___x_2004_; lean_object* v___x_2005_; lean_object* v___x_2006_; lean_object* v___x_2008_; 
v___x_2001_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__4));
v___x_2002_ = l_Lean_Expr_const___override(v___x_2001_, v___x_2000_);
v___x_2003_ = l_Lean_Expr_app___override(v___x_2002_, v_00_u03b1_1944_);
v___x_2004_ = l_Lean_Expr_app___override(v___x_2003_, v_ds_u03b1_1947_);
v___x_2005_ = l_Lean_Expr_app___override(v___x_2004_, v_a_1945_);
v___x_2006_ = l_Lean_Expr_app___override(v___x_2005_, v_proof_1994_);
if (v_isShared_1997_ == 0)
{
lean_ctor_set(v___x_1996_, 2, v___x_2006_);
v___x_2008_ = v___x_1996_;
goto v_reusejp_2007_;
}
else
{
lean_object* v_reuseFailAlloc_2012_; 
v_reuseFailAlloc_2012_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2012_, 0, v_inst_1992_);
lean_ctor_set(v_reuseFailAlloc_2012_, 1, v_lit_1993_);
lean_ctor_set(v_reuseFailAlloc_2012_, 2, v___x_2006_);
v___x_2008_ = v_reuseFailAlloc_2012_;
goto v_reusejp_2007_;
}
v_reusejp_2007_:
{
lean_object* v___x_2010_; 
if (v_isShared_1980_ == 0)
{
lean_ctor_set_tag(v___x_1979_, 0);
lean_ctor_set(v___x_1979_, 0, v___x_2008_);
v___x_2010_ = v___x_1979_;
goto v_reusejp_2009_;
}
else
{
lean_object* v_reuseFailAlloc_2011_; 
v_reuseFailAlloc_2011_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2011_, 0, v___x_2008_);
v___x_2010_ = v_reuseFailAlloc_2011_;
goto v_reusejp_2009_;
}
v_reusejp_2009_:
{
return v___x_2010_;
}
}
}
}
}
else
{
lean_object* v___x_2015_; lean_object* v___x_2016_; 
lean_del_object(v___x_1988_);
lean_del_object(v___x_1979_);
lean_dec_ref(v_ds_u03b1_1947_);
lean_dec_ref(v_ra_1946_);
lean_dec_ref(v_a_1945_);
lean_dec_ref(v_00_u03b1_1944_);
lean_dec(v_u_1943_);
v___x_2015_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1);
v___x_2016_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0___redArg(v___x_2015_, v_a_1949_, v_a_1950_, v_a_1951_, v_a_1952_);
return v___x_2016_;
}
}
else
{
lean_del_object(v___x_1979_);
if (lean_obj_tag(v_cz_u03b1_x3f_1948_) == 1)
{
lean_object* v_val_2017_; lean_object* v___x_2019_; uint8_t v_isShared_2020_; uint8_t v_isSharedCheck_2068_; 
lean_dec_ref(v_ra_1946_);
v_val_2017_ = lean_ctor_get(v_cz_u03b1_x3f_1948_, 0);
v_isSharedCheck_2068_ = !lean_is_exclusive(v_cz_u03b1_x3f_1948_);
if (v_isSharedCheck_2068_ == 0)
{
v___x_2019_ = v_cz_u03b1_x3f_1948_;
v_isShared_2020_ = v_isSharedCheck_2068_;
goto v_resetjp_2018_;
}
else
{
lean_inc(v_val_2017_);
lean_dec(v_cz_u03b1_x3f_1948_);
v___x_2019_ = lean_box(0);
v_isShared_2020_ = v_isSharedCheck_2068_;
goto v_resetjp_2018_;
}
v_resetjp_2018_:
{
lean_object* v_qb_2021_; lean_object* v___x_2022_; lean_object* v___x_2023_; lean_object* v___x_2024_; lean_object* v_lit2_2025_; lean_object* v___x_2026_; lean_object* v___x_2027_; lean_object* v___x_2029_; 
v_qb_2021_ = l_Rat_inv(v_fst_1983_);
v___x_2022_ = lp_batteries_Lean_Expr_natLit_x21(v_fst_1984_);
v___x_2023_ = lean_unsigned_to_nat(1u);
v___x_2024_ = lean_nat_sub(v___x_2022_, v___x_2023_);
lean_dec(v___x_2022_);
v_lit2_2025_ = l_Lean_mkRawNatLit(v___x_2024_);
v___x_2026_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__7));
v___x_2027_ = lean_box(0);
lean_inc(v_u_1943_);
if (v_isShared_1989_ == 0)
{
lean_ctor_set_tag(v___x_1988_, 1);
lean_ctor_set(v___x_1988_, 1, v___x_2027_);
lean_ctor_set(v___x_1988_, 0, v_u_1943_);
v___x_2029_ = v___x_1988_;
goto v_reusejp_2028_;
}
else
{
lean_object* v_reuseFailAlloc_2067_; 
v_reuseFailAlloc_2067_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2067_, 0, v_u_1943_);
lean_ctor_set(v_reuseFailAlloc_2067_, 1, v___x_2027_);
v___x_2029_ = v_reuseFailAlloc_2067_;
goto v_reusejp_2028_;
}
v_reusejp_2028_:
{
lean_object* v___x_2030_; lean_object* v___x_2031_; lean_object* v___x_2032_; lean_object* v___x_2033_; lean_object* v___x_2034_; lean_object* v___x_2035_; lean_object* v___x_2036_; lean_object* v___x_2037_; lean_object* v___x_2038_; lean_object* v___x_2039_; lean_object* v___x_2040_; lean_object* v___x_2041_; lean_object* v___x_2042_; lean_object* v___x_2043_; lean_object* v___x_2044_; lean_object* v___x_2045_; lean_object* v___x_2046_; lean_object* v___x_2047_; lean_object* v___x_2048_; lean_object* v___x_2049_; lean_object* v___x_2050_; lean_object* v___x_2051_; lean_object* v___x_2052_; lean_object* v___x_2053_; lean_object* v___x_2054_; lean_object* v___x_2055_; lean_object* v___x_2056_; lean_object* v___x_2057_; lean_object* v___x_2058_; lean_object* v___x_2059_; lean_object* v___x_2060_; lean_object* v___x_2061_; lean_object* v___x_2062_; lean_object* v___x_2063_; lean_object* v___x_2065_; 
lean_inc_ref_n(v___x_2029_, 6);
v___x_2030_ = l_Lean_Expr_const___override(v___x_2026_, v___x_2029_);
lean_inc_ref_n(v_00_u03b1_1944_, 7);
v___x_2031_ = l_Lean_Expr_app___override(v___x_2030_, v_00_u03b1_1944_);
v___x_2032_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__10));
v___x_2033_ = l_Lean_Expr_const___override(v___x_2032_, v___x_2029_);
v___x_2034_ = l_Lean_Expr_app___override(v___x_2033_, v_00_u03b1_1944_);
v___x_2035_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__13));
v___x_2036_ = l_Lean_Expr_const___override(v___x_2035_, v___x_2029_);
v___x_2037_ = l_Lean_Expr_app___override(v___x_2036_, v_00_u03b1_1944_);
v___x_2038_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__16));
v___x_2039_ = l_Lean_Expr_const___override(v___x_2038_, v___x_2029_);
v___x_2040_ = l_Lean_Expr_app___override(v___x_2039_, v_00_u03b1_1944_);
v___x_2041_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__19));
v___x_2042_ = l_Lean_Expr_const___override(v___x_2041_, v___x_2029_);
v___x_2043_ = l_Lean_Expr_app___override(v___x_2042_, v_00_u03b1_1944_);
v___x_2044_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__21));
v___x_2045_ = l_Lean_Expr_const___override(v___x_2044_, v___x_2029_);
v___x_2046_ = l_Lean_Expr_app___override(v___x_2045_, v_00_u03b1_1944_);
lean_inc_ref_n(v_ds_u03b1_1947_, 2);
v___x_2047_ = l_Lean_Expr_app___override(v___x_2046_, v_ds_u03b1_1947_);
v___x_2048_ = l_Lean_Expr_app___override(v___x_2043_, v___x_2047_);
v___x_2049_ = l_Lean_Expr_app___override(v___x_2040_, v___x_2048_);
v___x_2050_ = l_Lean_Expr_app___override(v___x_2037_, v___x_2049_);
v___x_2051_ = l_Lean_Expr_app___override(v___x_2034_, v___x_2050_);
v___x_2052_ = l_Lean_Expr_app___override(v___x_2031_, v___x_2051_);
lean_inc_ref(v_a_1945_);
v___x_2053_ = l_Lean_Expr_app___override(v___x_2052_, v_a_1945_);
v___x_2054_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__23));
v___x_2055_ = l_Lean_Expr_const___override(v___x_2054_, v___x_2029_);
v___x_2056_ = l_Lean_Expr_app___override(v___x_2055_, v_00_u03b1_1944_);
v___x_2057_ = l_Lean_Expr_app___override(v___x_2056_, v_ds_u03b1_1947_);
v___x_2058_ = l_Lean_Expr_app___override(v___x_2057_, v_val_2017_);
v___x_2059_ = l_Lean_Expr_app___override(v___x_2058_, v_a_1945_);
v___x_2060_ = l_Lean_Expr_app___override(v___x_2059_, v_lit2_2025_);
lean_inc(v_fst_1985_);
v___x_2061_ = l_Lean_Expr_app___override(v___x_2060_, v_fst_1985_);
v___x_2062_ = l_Lean_Expr_app___override(v___x_2061_, v_snd_1986_);
v___x_2063_ = lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27(v_u_1943_, v_00_u03b1_1944_, v___x_2053_, v_ds_u03b1_1947_, v_qb_2021_, v_fst_1985_, v_fst_1984_, v___x_2062_);
if (v_isShared_2020_ == 0)
{
lean_ctor_set_tag(v___x_2019_, 0);
lean_ctor_set(v___x_2019_, 0, v___x_2063_);
v___x_2065_ = v___x_2019_;
goto v_reusejp_2064_;
}
else
{
lean_object* v_reuseFailAlloc_2066_; 
v_reuseFailAlloc_2066_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2066_, 0, v___x_2063_);
v___x_2065_ = v_reuseFailAlloc_2066_;
goto v_reusejp_2064_;
}
v_reusejp_2064_:
{
return v___x_2065_;
}
}
}
}
else
{
lean_object* v___x_2069_; uint8_t v___x_2070_; 
lean_del_object(v___x_1988_);
lean_dec(v_snd_1986_);
lean_dec(v_fst_1985_);
lean_dec(v_fst_1984_);
lean_dec(v_cz_u03b1_x3f_1948_);
v___x_2069_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__24, &lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__24_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__24);
v___x_2070_ = l_instDecidableEqRat_decEq(v_fst_1983_, v___x_2069_);
lean_dec(v_fst_1983_);
if (v___x_2070_ == 0)
{
lean_object* v___x_2071_; lean_object* v___x_2072_; lean_object* v_a_2073_; lean_object* v___x_2075_; uint8_t v_isShared_2076_; uint8_t v_isSharedCheck_2080_; 
lean_dec_ref(v_ds_u03b1_1947_);
lean_dec_ref(v_ra_1946_);
lean_dec_ref(v_a_1945_);
lean_dec_ref(v_00_u03b1_1944_);
lean_dec(v_u_1943_);
v___x_2071_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1);
v___x_2072_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0___redArg(v___x_2071_, v_a_1949_, v_a_1950_, v_a_1951_, v_a_1952_);
v_a_2073_ = lean_ctor_get(v___x_2072_, 0);
v_isSharedCheck_2080_ = !lean_is_exclusive(v___x_2072_);
if (v_isSharedCheck_2080_ == 0)
{
v___x_2075_ = v___x_2072_;
v_isShared_2076_ = v_isSharedCheck_2080_;
goto v_resetjp_2074_;
}
else
{
lean_inc(v_a_2073_);
lean_dec(v___x_2072_);
v___x_2075_ = lean_box(0);
v_isShared_2076_ = v_isSharedCheck_2080_;
goto v_resetjp_2074_;
}
v_resetjp_2074_:
{
lean_object* v___x_2078_; 
if (v_isShared_2076_ == 0)
{
v___x_2078_ = v___x_2075_;
goto v_reusejp_2077_;
}
else
{
lean_object* v_reuseFailAlloc_2079_; 
v_reuseFailAlloc_2079_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2079_, 0, v_a_2073_);
v___x_2078_ = v_reuseFailAlloc_2079_;
goto v_reusejp_2077_;
}
v_reusejp_2077_:
{
return v___x_2078_;
}
}
}
else
{
goto v___jp_1954_;
}
}
}
}
}
}
else
{
lean_object* v___x_2083_; 
lean_dec(v___x_1976_);
lean_inc_ref(v_00_u03b1_1944_);
lean_inc(v_u_1943_);
v___x_2083_ = lp_mathlib_Mathlib_Meta_NormNum_inferDivisionRing(v_u_1943_, v_00_u03b1_1944_, v_a_1949_, v_a_1950_, v_a_1951_, v_a_1952_);
if (lean_obj_tag(v___x_2083_) == 0)
{
lean_object* v_a_2084_; lean_object* v___x_2086_; uint8_t v_isShared_2087_; uint8_t v_isSharedCheck_2231_; 
v_a_2084_ = lean_ctor_get(v___x_2083_, 0);
v_isSharedCheck_2231_ = !lean_is_exclusive(v___x_2083_);
if (v_isSharedCheck_2231_ == 0)
{
v___x_2086_ = v___x_2083_;
v_isShared_2087_ = v_isSharedCheck_2231_;
goto v_resetjp_2085_;
}
else
{
lean_inc(v_a_2084_);
lean_dec(v___x_2083_);
v___x_2086_ = lean_box(0);
v_isShared_2087_ = v_isSharedCheck_2231_;
goto v_resetjp_2085_;
}
v_resetjp_2085_:
{
lean_object* v___x_2088_; lean_object* v___x_2089_; lean_object* v___x_2112_; lean_object* v___x_2113_; lean_object* v___x_2114_; lean_object* v___x_2115_; uint8_t v___x_2116_; lean_object* v___x_2117_; lean_object* v___f_2118_; uint8_t v___x_2119_; lean_object* v___x_2120_; 
v___x_2088_ = lean_box(0);
lean_inc(v_u_1943_);
v___x_2089_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2089_, 0, v_u_1943_);
lean_ctor_set(v___x_2089_, 1, v___x_2088_);
v___x_2112_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__13));
lean_inc_ref(v___x_2089_);
v___x_2113_ = l_Lean_Expr_const___override(v___x_2112_, v___x_2089_);
lean_inc_ref(v_00_u03b1_1944_);
v___x_2114_ = l_Lean_Expr_app___override(v___x_2113_, v_00_u03b1_1944_);
lean_inc(v_a_2084_);
v___x_2115_ = l_Lean_Expr_app___override(v___x_2114_, v_a_2084_);
v___x_2116_ = 1;
v___x_2117_ = lean_box(v___x_2116_);
lean_inc_ref(v_ds_u03b1_1947_);
v___f_2118_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_NormNum_Result_inv___lam__0___boxed), 8, 3);
lean_closure_set(v___f_2118_, 0, v___x_2117_);
lean_closure_set(v___f_2118_, 1, v_ds_u03b1_1947_);
lean_closure_set(v___f_2118_, 2, v___x_2115_);
v___x_2119_ = 0;
v___x_2120_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalRatCast_spec__0___redArg(v___f_2118_, v___x_2119_, v_a_1949_, v_a_1950_, v_a_1951_, v_a_1952_);
if (lean_obj_tag(v___x_2120_) == 0)
{
lean_object* v___x_2122_; uint8_t v_isShared_2123_; uint8_t v_isSharedCheck_2221_; 
v_isSharedCheck_2221_ = !lean_is_exclusive(v___x_2120_);
if (v_isSharedCheck_2221_ == 0)
{
lean_object* v_unused_2222_; 
v_unused_2222_ = lean_ctor_get(v___x_2120_, 0);
lean_dec(v_unused_2222_);
v___x_2122_ = v___x_2120_;
v_isShared_2123_ = v_isSharedCheck_2221_;
goto v_resetjp_2121_;
}
else
{
lean_dec(v___x_2120_);
v___x_2122_ = lean_box(0);
v_isShared_2123_ = v_isSharedCheck_2221_;
goto v_resetjp_2121_;
}
v_resetjp_2121_:
{
lean_object* v___y_2125_; lean_object* v___y_2126_; lean_object* v___y_2127_; lean_object* v___y_2128_; lean_object* v___y_2129_; lean_object* v_a_2189_; lean_object* v___x_2209_; 
lean_inc_ref(v_ra_1946_);
lean_inc(v_a_2084_);
lean_inc_ref(v_a_1945_);
lean_inc_ref(v_00_u03b1_1944_);
lean_inc(v_u_1943_);
v___x_2209_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27(v_u_1943_, v_00_u03b1_1944_, v_a_1945_, v_a_2084_, v_ra_1946_);
if (lean_obj_tag(v___x_2209_) == 0)
{
lean_object* v___x_2210_; lean_object* v___x_2211_; lean_object* v_a_2212_; lean_object* v___x_2214_; uint8_t v_isShared_2215_; uint8_t v_isSharedCheck_2219_; 
lean_del_object(v___x_2122_);
lean_dec_ref_known(v___x_2089_, 2);
lean_del_object(v___x_2086_);
lean_dec(v_a_2084_);
lean_dec(v_cz_u03b1_x3f_1948_);
lean_dec_ref(v_ds_u03b1_1947_);
lean_dec_ref(v_ra_1946_);
lean_dec_ref(v_a_1945_);
lean_dec_ref(v_00_u03b1_1944_);
lean_dec(v_u_1943_);
v___x_2210_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1);
v___x_2211_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0___redArg(v___x_2210_, v_a_1949_, v_a_1950_, v_a_1951_, v_a_1952_);
v_a_2212_ = lean_ctor_get(v___x_2211_, 0);
v_isSharedCheck_2219_ = !lean_is_exclusive(v___x_2211_);
if (v_isSharedCheck_2219_ == 0)
{
v___x_2214_ = v___x_2211_;
v_isShared_2215_ = v_isSharedCheck_2219_;
goto v_resetjp_2213_;
}
else
{
lean_inc(v_a_2212_);
lean_dec(v___x_2211_);
v___x_2214_ = lean_box(0);
v_isShared_2215_ = v_isSharedCheck_2219_;
goto v_resetjp_2213_;
}
v_resetjp_2213_:
{
lean_object* v___x_2217_; 
if (v_isShared_2215_ == 0)
{
v___x_2217_ = v___x_2214_;
goto v_reusejp_2216_;
}
else
{
lean_object* v_reuseFailAlloc_2218_; 
v_reuseFailAlloc_2218_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2218_, 0, v_a_2212_);
v___x_2217_ = v_reuseFailAlloc_2218_;
goto v_reusejp_2216_;
}
v_reusejp_2216_:
{
return v___x_2217_;
}
}
}
else
{
lean_object* v_val_2220_; 
v_val_2220_ = lean_ctor_get(v___x_2209_, 0);
lean_inc(v_val_2220_);
lean_dec_ref_known(v___x_2209_, 1);
v_a_2189_ = v_val_2220_;
goto v___jp_2188_;
}
v___jp_2124_:
{
if (lean_obj_tag(v_cz_u03b1_x3f_1948_) == 1)
{
lean_object* v_val_2130_; lean_object* v___x_2131_; lean_object* v___x_2132_; lean_object* v___x_2133_; lean_object* v___x_2134_; lean_object* v___x_2135_; lean_object* v___x_2136_; lean_object* v___x_2137_; lean_object* v___x_2138_; lean_object* v___x_2139_; lean_object* v___x_2140_; lean_object* v___x_2141_; lean_object* v___x_2142_; lean_object* v___x_2143_; lean_object* v___x_2144_; lean_object* v___x_2145_; lean_object* v___x_2146_; lean_object* v___x_2147_; lean_object* v___x_2148_; lean_object* v___x_2149_; lean_object* v___x_2150_; lean_object* v___x_2151_; lean_object* v___x_2152_; lean_object* v___x_2153_; lean_object* v___x_2154_; lean_object* v___x_2155_; lean_object* v___x_2156_; lean_object* v___x_2157_; lean_object* v___x_2158_; lean_object* v___x_2159_; lean_object* v___x_2160_; lean_object* v___x_2161_; lean_object* v___x_2162_; lean_object* v___x_2163_; lean_object* v___x_2164_; lean_object* v___x_2165_; lean_object* v___x_2166_; lean_object* v___x_2167_; lean_object* v___x_2168_; lean_object* v___x_2169_; lean_object* v___x_2170_; lean_object* v___x_2171_; lean_object* v___x_2172_; lean_object* v___x_2174_; 
lean_dec_ref(v___y_2128_);
lean_del_object(v___x_2086_);
lean_dec_ref(v_ra_1946_);
v_val_2130_ = lean_ctor_get(v_cz_u03b1_x3f_1948_, 0);
lean_inc(v_val_2130_);
lean_dec_ref_known(v_cz_u03b1_x3f_1948_, 1);
v___x_2131_ = l_Lean_Expr_appArg_x21(v___y_2127_);
lean_dec_ref(v___y_2127_);
v___x_2132_ = lp_batteries_Lean_Expr_natLit_x21(v___x_2131_);
v___x_2133_ = lean_unsigned_to_nat(1u);
v___x_2134_ = lean_nat_sub(v___x_2132_, v___x_2133_);
lean_dec(v___x_2132_);
v___x_2135_ = l_Lean_mkRawNatLit(v___x_2134_);
v___x_2136_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__7));
lean_inc_ref_n(v___x_2089_, 6);
v___x_2137_ = l_Lean_Expr_const___override(v___x_2136_, v___x_2089_);
lean_inc_ref_n(v_00_u03b1_1944_, 7);
v___x_2138_ = l_Lean_Expr_app___override(v___x_2137_, v_00_u03b1_1944_);
v___x_2139_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__10));
v___x_2140_ = l_Lean_Expr_const___override(v___x_2139_, v___x_2089_);
v___x_2141_ = l_Lean_Expr_app___override(v___x_2140_, v_00_u03b1_1944_);
v___x_2142_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__13));
v___x_2143_ = l_Lean_Expr_const___override(v___x_2142_, v___x_2089_);
v___x_2144_ = l_Lean_Expr_app___override(v___x_2143_, v_00_u03b1_1944_);
v___x_2145_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__16));
v___x_2146_ = l_Lean_Expr_const___override(v___x_2145_, v___x_2089_);
v___x_2147_ = l_Lean_Expr_app___override(v___x_2146_, v_00_u03b1_1944_);
v___x_2148_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__19));
v___x_2149_ = l_Lean_Expr_const___override(v___x_2148_, v___x_2089_);
v___x_2150_ = l_Lean_Expr_app___override(v___x_2149_, v_00_u03b1_1944_);
v___x_2151_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__21));
v___x_2152_ = l_Lean_Expr_const___override(v___x_2151_, v___x_2089_);
v___x_2153_ = l_Lean_Expr_app___override(v___x_2152_, v_00_u03b1_1944_);
v___x_2154_ = l_Lean_Expr_app___override(v___x_2153_, v_ds_u03b1_1947_);
v___x_2155_ = l_Lean_Expr_app___override(v___x_2150_, v___x_2154_);
v___x_2156_ = l_Lean_Expr_app___override(v___x_2147_, v___x_2155_);
v___x_2157_ = l_Lean_Expr_app___override(v___x_2144_, v___x_2156_);
v___x_2158_ = l_Lean_Expr_app___override(v___x_2141_, v___x_2157_);
v___x_2159_ = l_Lean_Expr_app___override(v___x_2138_, v___x_2158_);
lean_inc_ref(v_a_1945_);
v___x_2160_ = l_Lean_Expr_app___override(v___x_2159_, v_a_1945_);
v___x_2161_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__7, &lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__7_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__1___closed__7);
lean_inc_ref(v___y_2125_);
v___x_2162_ = l_Lean_Expr_app___override(v___x_2161_, v___y_2125_);
v___x_2163_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__28));
v___x_2164_ = l_Lean_Expr_const___override(v___x_2163_, v___x_2089_);
v___x_2165_ = l_Lean_Expr_app___override(v___x_2164_, v_00_u03b1_1944_);
lean_inc(v_a_2084_);
v___x_2166_ = l_Lean_Expr_app___override(v___x_2165_, v_a_2084_);
v___x_2167_ = l_Lean_Expr_app___override(v___x_2166_, v_val_2130_);
v___x_2168_ = l_Lean_Expr_app___override(v___x_2167_, v_a_1945_);
v___x_2169_ = l_Lean_Expr_app___override(v___x_2168_, v___x_2135_);
v___x_2170_ = l_Lean_Expr_app___override(v___x_2169_, v___y_2125_);
v___x_2171_ = l_Lean_Expr_app___override(v___x_2170_, v___y_2129_);
v___x_2172_ = lp_mathlib_Mathlib_Meta_NormNum_Result_isRat(v_u_1943_, v_00_u03b1_1944_, v___x_2160_, v_a_2084_, v___y_2126_, v___x_2162_, v___x_2131_, v___x_2171_);
if (v_isShared_2123_ == 0)
{
lean_ctor_set(v___x_2122_, 0, v___x_2172_);
v___x_2174_ = v___x_2122_;
goto v_reusejp_2173_;
}
else
{
lean_object* v_reuseFailAlloc_2175_; 
v_reuseFailAlloc_2175_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2175_, 0, v___x_2172_);
v___x_2174_ = v_reuseFailAlloc_2175_;
goto v_reusejp_2173_;
}
v_reusejp_2173_:
{
return v___x_2174_;
}
}
else
{
lean_object* v___x_2176_; uint8_t v___x_2177_; 
lean_dec_ref(v___y_2129_);
lean_dec_ref(v___y_2127_);
lean_dec_ref(v___y_2126_);
lean_dec_ref(v___y_2125_);
lean_del_object(v___x_2122_);
lean_dec(v_cz_u03b1_x3f_1948_);
lean_dec_ref(v_ds_u03b1_1947_);
lean_dec(v_u_1943_);
v___x_2176_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__29, &lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__29_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__29);
v___x_2177_ = l_instDecidableEqRat_decEq(v___y_2128_, v___x_2176_);
lean_dec_ref(v___y_2128_);
if (v___x_2177_ == 0)
{
lean_object* v___x_2178_; lean_object* v___x_2179_; lean_object* v_a_2180_; lean_object* v___x_2182_; uint8_t v_isShared_2183_; uint8_t v_isSharedCheck_2187_; 
lean_dec_ref_known(v___x_2089_, 2);
lean_del_object(v___x_2086_);
lean_dec(v_a_2084_);
lean_dec_ref(v_ra_1946_);
lean_dec_ref(v_a_1945_);
lean_dec_ref(v_00_u03b1_1944_);
v___x_2178_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1);
v___x_2179_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0___redArg(v___x_2178_, v_a_1949_, v_a_1950_, v_a_1951_, v_a_1952_);
v_a_2180_ = lean_ctor_get(v___x_2179_, 0);
v_isSharedCheck_2187_ = !lean_is_exclusive(v___x_2179_);
if (v_isSharedCheck_2187_ == 0)
{
v___x_2182_ = v___x_2179_;
v_isShared_2183_ = v_isSharedCheck_2187_;
goto v_resetjp_2181_;
}
else
{
lean_inc(v_a_2180_);
lean_dec(v___x_2179_);
v___x_2182_ = lean_box(0);
v_isShared_2183_ = v_isSharedCheck_2187_;
goto v_resetjp_2181_;
}
v_resetjp_2181_:
{
lean_object* v___x_2185_; 
if (v_isShared_2183_ == 0)
{
v___x_2185_ = v___x_2182_;
goto v_reusejp_2184_;
}
else
{
lean_object* v_reuseFailAlloc_2186_; 
v_reuseFailAlloc_2186_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2186_, 0, v_a_2180_);
v___x_2185_ = v_reuseFailAlloc_2186_;
goto v_reusejp_2184_;
}
v_reusejp_2184_:
{
return v___x_2185_;
}
}
}
else
{
goto v___jp_2090_;
}
}
}
v___jp_2188_:
{
lean_object* v_snd_2190_; lean_object* v_snd_2191_; lean_object* v_fst_2192_; lean_object* v_fst_2193_; lean_object* v_fst_2194_; lean_object* v_snd_2195_; lean_object* v___x_2196_; lean_object* v___x_2197_; uint8_t v___x_2198_; 
v_snd_2190_ = lean_ctor_get(v_a_2189_, 1);
lean_inc(v_snd_2190_);
v_snd_2191_ = lean_ctor_get(v_snd_2190_, 1);
lean_inc(v_snd_2191_);
v_fst_2192_ = lean_ctor_get(v_a_2189_, 0);
lean_inc_n(v_fst_2192_, 3);
lean_dec_ref(v_a_2189_);
v_fst_2193_ = lean_ctor_get(v_snd_2190_, 0);
lean_inc(v_fst_2193_);
lean_dec(v_snd_2190_);
v_fst_2194_ = lean_ctor_get(v_snd_2191_, 0);
lean_inc(v_fst_2194_);
v_snd_2195_ = lean_ctor_get(v_snd_2191_, 1);
lean_inc(v_snd_2195_);
lean_dec(v_snd_2191_);
v___x_2196_ = l_Rat_inv(v_fst_2192_);
v___x_2197_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__2, &lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__2_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__2);
v___x_2198_ = l_Rat_blt(v_fst_2192_, v___x_2197_);
if (v___x_2198_ == 0)
{
lean_object* v___x_2199_; lean_object* v___x_2200_; lean_object* v_a_2201_; lean_object* v___x_2203_; uint8_t v_isShared_2204_; uint8_t v_isSharedCheck_2208_; 
lean_dec_ref(v___x_2196_);
lean_dec(v_snd_2195_);
lean_dec(v_fst_2194_);
lean_dec(v_fst_2193_);
lean_dec(v_fst_2192_);
lean_del_object(v___x_2122_);
lean_dec_ref_known(v___x_2089_, 2);
lean_del_object(v___x_2086_);
lean_dec(v_a_2084_);
lean_dec(v_cz_u03b1_x3f_1948_);
lean_dec_ref(v_ds_u03b1_1947_);
lean_dec_ref(v_ra_1946_);
lean_dec_ref(v_a_1945_);
lean_dec_ref(v_00_u03b1_1944_);
lean_dec(v_u_1943_);
v___x_2199_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1);
v___x_2200_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0___redArg(v___x_2199_, v_a_1949_, v_a_1950_, v_a_1951_, v_a_1952_);
v_a_2201_ = lean_ctor_get(v___x_2200_, 0);
v_isSharedCheck_2208_ = !lean_is_exclusive(v___x_2200_);
if (v_isSharedCheck_2208_ == 0)
{
v___x_2203_ = v___x_2200_;
v_isShared_2204_ = v_isSharedCheck_2208_;
goto v_resetjp_2202_;
}
else
{
lean_inc(v_a_2201_);
lean_dec(v___x_2200_);
v___x_2203_ = lean_box(0);
v_isShared_2204_ = v_isSharedCheck_2208_;
goto v_resetjp_2202_;
}
v_resetjp_2202_:
{
lean_object* v___x_2206_; 
if (v_isShared_2204_ == 0)
{
v___x_2206_ = v___x_2203_;
goto v_reusejp_2205_;
}
else
{
lean_object* v_reuseFailAlloc_2207_; 
v_reuseFailAlloc_2207_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2207_, 0, v_a_2201_);
v___x_2206_ = v_reuseFailAlloc_2207_;
goto v_reusejp_2205_;
}
v_reusejp_2205_:
{
return v___x_2206_;
}
}
}
else
{
v___y_2125_ = v_fst_2194_;
v___y_2126_ = v___x_2196_;
v___y_2127_ = v_fst_2193_;
v___y_2128_ = v_fst_2192_;
v___y_2129_ = v_snd_2195_;
goto v___jp_2124_;
}
}
}
}
else
{
lean_object* v_a_2223_; lean_object* v___x_2225_; uint8_t v_isShared_2226_; uint8_t v_isSharedCheck_2230_; 
lean_dec_ref_known(v___x_2089_, 2);
lean_del_object(v___x_2086_);
lean_dec(v_a_2084_);
lean_dec(v_cz_u03b1_x3f_1948_);
lean_dec_ref(v_ds_u03b1_1947_);
lean_dec_ref(v_ra_1946_);
lean_dec_ref(v_a_1945_);
lean_dec_ref(v_00_u03b1_1944_);
lean_dec(v_u_1943_);
v_a_2223_ = lean_ctor_get(v___x_2120_, 0);
v_isSharedCheck_2230_ = !lean_is_exclusive(v___x_2120_);
if (v_isSharedCheck_2230_ == 0)
{
v___x_2225_ = v___x_2120_;
v_isShared_2226_ = v_isSharedCheck_2230_;
goto v_resetjp_2224_;
}
else
{
lean_inc(v_a_2223_);
lean_dec(v___x_2120_);
v___x_2225_ = lean_box(0);
v_isShared_2226_ = v_isSharedCheck_2230_;
goto v_resetjp_2224_;
}
v_resetjp_2224_:
{
lean_object* v___x_2228_; 
if (v_isShared_2226_ == 0)
{
v___x_2228_ = v___x_2225_;
goto v_reusejp_2227_;
}
else
{
lean_object* v_reuseFailAlloc_2229_; 
v_reuseFailAlloc_2229_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2229_, 0, v_a_2223_);
v___x_2228_ = v_reuseFailAlloc_2229_;
goto v_reusejp_2227_;
}
v_reusejp_2227_:
{
return v___x_2228_;
}
}
}
v___jp_2090_:
{
if (lean_obj_tag(v_ra_1946_) == 2)
{
lean_object* v_inst_2091_; lean_object* v_lit_2092_; lean_object* v_proof_2093_; lean_object* v___x_2095_; uint8_t v_isShared_2096_; uint8_t v_isSharedCheck_2109_; 
v_inst_2091_ = lean_ctor_get(v_ra_1946_, 0);
v_lit_2092_ = lean_ctor_get(v_ra_1946_, 1);
v_proof_2093_ = lean_ctor_get(v_ra_1946_, 2);
v_isSharedCheck_2109_ = !lean_is_exclusive(v_ra_1946_);
if (v_isSharedCheck_2109_ == 0)
{
v___x_2095_ = v_ra_1946_;
v_isShared_2096_ = v_isSharedCheck_2109_;
goto v_resetjp_2094_;
}
else
{
lean_inc(v_proof_2093_);
lean_inc(v_lit_2092_);
lean_inc(v_inst_2091_);
lean_dec(v_ra_1946_);
v___x_2095_ = lean_box(0);
v_isShared_2096_ = v_isSharedCheck_2109_;
goto v_resetjp_2094_;
}
v_resetjp_2094_:
{
lean_object* v___x_2097_; lean_object* v___x_2098_; lean_object* v___x_2099_; lean_object* v___x_2100_; lean_object* v___x_2101_; lean_object* v___x_2102_; lean_object* v___x_2104_; 
v___x_2097_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__26));
v___x_2098_ = l_Lean_Expr_const___override(v___x_2097_, v___x_2089_);
v___x_2099_ = l_Lean_Expr_app___override(v___x_2098_, v_00_u03b1_1944_);
v___x_2100_ = l_Lean_Expr_app___override(v___x_2099_, v_a_2084_);
v___x_2101_ = l_Lean_Expr_app___override(v___x_2100_, v_a_1945_);
v___x_2102_ = l_Lean_Expr_app___override(v___x_2101_, v_proof_2093_);
if (v_isShared_2096_ == 0)
{
lean_ctor_set(v___x_2095_, 2, v___x_2102_);
v___x_2104_ = v___x_2095_;
goto v_reusejp_2103_;
}
else
{
lean_object* v_reuseFailAlloc_2108_; 
v_reuseFailAlloc_2108_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2108_, 0, v_inst_2091_);
lean_ctor_set(v_reuseFailAlloc_2108_, 1, v_lit_2092_);
lean_ctor_set(v_reuseFailAlloc_2108_, 2, v___x_2102_);
v___x_2104_ = v_reuseFailAlloc_2108_;
goto v_reusejp_2103_;
}
v_reusejp_2103_:
{
lean_object* v___x_2106_; 
if (v_isShared_2087_ == 0)
{
lean_ctor_set(v___x_2086_, 0, v___x_2104_);
v___x_2106_ = v___x_2086_;
goto v_reusejp_2105_;
}
else
{
lean_object* v_reuseFailAlloc_2107_; 
v_reuseFailAlloc_2107_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2107_, 0, v___x_2104_);
v___x_2106_ = v_reuseFailAlloc_2107_;
goto v_reusejp_2105_;
}
v_reusejp_2105_:
{
return v___x_2106_;
}
}
}
}
else
{
lean_object* v___x_2110_; lean_object* v___x_2111_; 
lean_dec_ref_known(v___x_2089_, 2);
lean_del_object(v___x_2086_);
lean_dec(v_a_2084_);
lean_dec_ref(v_ra_1946_);
lean_dec_ref(v_a_1945_);
lean_dec_ref(v_00_u03b1_1944_);
v___x_2110_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1);
v___x_2111_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0___redArg(v___x_2110_, v_a_1949_, v_a_1950_, v_a_1951_, v_a_1952_);
return v___x_2111_;
}
}
}
}
else
{
lean_object* v_a_2232_; lean_object* v___x_2234_; uint8_t v_isShared_2235_; uint8_t v_isSharedCheck_2239_; 
lean_dec(v_cz_u03b1_x3f_1948_);
lean_dec_ref(v_ds_u03b1_1947_);
lean_dec_ref(v_ra_1946_);
lean_dec_ref(v_a_1945_);
lean_dec_ref(v_00_u03b1_1944_);
lean_dec(v_u_1943_);
v_a_2232_ = lean_ctor_get(v___x_2083_, 0);
v_isSharedCheck_2239_ = !lean_is_exclusive(v___x_2083_);
if (v_isSharedCheck_2239_ == 0)
{
v___x_2234_ = v___x_2083_;
v_isShared_2235_ = v_isSharedCheck_2239_;
goto v_resetjp_2233_;
}
else
{
lean_inc(v_a_2232_);
lean_dec(v___x_2083_);
v___x_2234_ = lean_box(0);
v_isShared_2235_ = v_isSharedCheck_2239_;
goto v_resetjp_2233_;
}
v_resetjp_2233_:
{
lean_object* v___x_2237_; 
if (v_isShared_2235_ == 0)
{
v___x_2237_ = v___x_2234_;
goto v_reusejp_2236_;
}
else
{
lean_object* v_reuseFailAlloc_2238_; 
v_reuseFailAlloc_2238_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2238_, 0, v_a_2232_);
v___x_2237_ = v_reuseFailAlloc_2238_;
goto v_reusejp_2236_;
}
v_reusejp_2236_:
{
return v___x_2237_;
}
}
}
}
v___jp_1954_:
{
if (lean_obj_tag(v_ra_1946_) == 1)
{
lean_object* v_inst_1955_; lean_object* v_lit_1956_; lean_object* v_proof_1957_; lean_object* v___x_1959_; uint8_t v_isShared_1960_; uint8_t v_isSharedCheck_1973_; 
v_inst_1955_ = lean_ctor_get(v_ra_1946_, 0);
v_lit_1956_ = lean_ctor_get(v_ra_1946_, 1);
v_proof_1957_ = lean_ctor_get(v_ra_1946_, 2);
v_isSharedCheck_1973_ = !lean_is_exclusive(v_ra_1946_);
if (v_isSharedCheck_1973_ == 0)
{
v___x_1959_ = v_ra_1946_;
v_isShared_1960_ = v_isSharedCheck_1973_;
goto v_resetjp_1958_;
}
else
{
lean_inc(v_proof_1957_);
lean_inc(v_lit_1956_);
lean_inc(v_inst_1955_);
lean_dec(v_ra_1946_);
v___x_1959_ = lean_box(0);
v_isShared_1960_ = v_isSharedCheck_1973_;
goto v_resetjp_1958_;
}
v_resetjp_1958_:
{
lean_object* v___x_1961_; lean_object* v___x_1962_; lean_object* v___x_1963_; lean_object* v___x_1964_; lean_object* v___x_1965_; lean_object* v___x_1966_; lean_object* v___x_1967_; lean_object* v___x_1968_; lean_object* v___x_1970_; 
v___x_1961_ = lean_box(0);
v___x_1962_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1962_, 0, v_u_1943_);
lean_ctor_set(v___x_1962_, 1, v___x_1961_);
v___x_1963_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__1));
v___x_1964_ = l_Lean_Expr_const___override(v___x_1963_, v___x_1962_);
v___x_1965_ = l_Lean_Expr_app___override(v___x_1964_, v_00_u03b1_1944_);
v___x_1966_ = l_Lean_Expr_app___override(v___x_1965_, v_ds_u03b1_1947_);
v___x_1967_ = l_Lean_Expr_app___override(v___x_1966_, v_a_1945_);
v___x_1968_ = l_Lean_Expr_app___override(v___x_1967_, v_proof_1957_);
if (v_isShared_1960_ == 0)
{
lean_ctor_set(v___x_1959_, 2, v___x_1968_);
v___x_1970_ = v___x_1959_;
goto v_reusejp_1969_;
}
else
{
lean_object* v_reuseFailAlloc_1972_; 
v_reuseFailAlloc_1972_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1972_, 0, v_inst_1955_);
lean_ctor_set(v_reuseFailAlloc_1972_, 1, v_lit_1956_);
lean_ctor_set(v_reuseFailAlloc_1972_, 2, v___x_1968_);
v___x_1970_ = v_reuseFailAlloc_1972_;
goto v_reusejp_1969_;
}
v_reusejp_1969_:
{
lean_object* v___x_1971_; 
v___x_1971_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1971_, 0, v___x_1970_);
return v___x_1971_;
}
}
}
else
{
lean_object* v___x_1974_; lean_object* v___x_1975_; 
lean_dec_ref(v_ds_u03b1_1947_);
lean_dec_ref(v_ra_1946_);
lean_dec_ref(v_a_1945_);
lean_dec_ref(v_00_u03b1_1944_);
lean_dec(v_u_1943_);
v___x_1974_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1);
v___x_1975_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0___redArg(v___x_1974_, v_a_1949_, v_a_1950_, v_a_1951_, v_a_1952_);
return v___x_1975_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___boxed(lean_object* v_u_2240_, lean_object* v_00_u03b1_2241_, lean_object* v_a_2242_, lean_object* v_ra_2243_, lean_object* v_ds_u03b1_2244_, lean_object* v_cz_u03b1_x3f_2245_, lean_object* v_a_2246_, lean_object* v_a_2247_, lean_object* v_a_2248_, lean_object* v_a_2249_, lean_object* v_a_2250_){
_start:
{
lean_object* v_res_2251_; 
v_res_2251_ = lp_mathlib_Mathlib_Meta_NormNum_Result_inv(v_u_2240_, v_00_u03b1_2241_, v_a_2242_, v_ra_2243_, v_ds_u03b1_2244_, v_cz_u03b1_x3f_2245_, v_a_2246_, v_a_2247_, v_a_2248_, v_a_2249_);
lean_dec(v_a_2249_);
lean_dec_ref(v_a_2248_);
lean_dec(v_a_2247_);
lean_dec_ref(v_a_2246_);
return v_res_2251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Nat_cast___at___00Mathlib_Meta_NormNum_Result_inv_spec__0_spec__0(lean_object* v_a_2252_){
_start:
{
lean_object* v___x_2253_; 
v___x_2253_ = lean_nat_to_int(v_a_2252_);
return v___x_2253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalInv___lam__1(lean_object* v_u_2254_, lean_object* v_00_u03b1_2255_, lean_object* v_e_2256_, lean_object* v___y_2257_, lean_object* v___y_2258_, lean_object* v___y_2259_, lean_object* v___y_2260_){
_start:
{
lean_object* v___x_2262_; 
v___x_2262_ = l_Lean_Meta_whnfR(v_e_2256_, v___y_2257_, v___y_2258_, v___y_2259_, v___y_2260_);
if (lean_obj_tag(v___x_2262_) == 0)
{
lean_object* v_a_2263_; 
v_a_2263_ = lean_ctor_get(v___x_2262_, 0);
lean_inc(v_a_2263_);
lean_dec_ref_known(v___x_2262_, 1);
if (lean_obj_tag(v_a_2263_) == 5)
{
lean_object* v_fn_2264_; lean_object* v_arg_2265_; uint8_t v___x_2266_; lean_object* v___x_2267_; 
v_fn_2264_ = lean_ctor_get(v_a_2263_, 0);
lean_inc_ref(v_fn_2264_);
v_arg_2265_ = lean_ctor_get(v_a_2263_, 1);
lean_inc_ref_n(v_arg_2265_, 2);
lean_dec_ref_known(v_a_2263_, 2);
v___x_2266_ = 0;
lean_inc_ref(v_00_u03b1_2255_);
lean_inc(v_u_2254_);
v___x_2267_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v_u_2254_, v_00_u03b1_2255_, v_arg_2265_, v___x_2266_, v___y_2257_, v___y_2258_, v___y_2259_, v___y_2260_);
if (lean_obj_tag(v___x_2267_) == 0)
{
lean_object* v_a_2268_; lean_object* v___x_2269_; 
v_a_2268_ = lean_ctor_get(v___x_2267_, 0);
lean_inc(v_a_2268_);
lean_dec_ref_known(v___x_2267_, 1);
lean_inc_ref(v_00_u03b1_2255_);
lean_inc(v_u_2254_);
v___x_2269_ = lp_mathlib_Mathlib_Meta_NormNum_inferDivisionSemiring(v_u_2254_, v_00_u03b1_2255_, v___y_2257_, v___y_2258_, v___y_2259_, v___y_2260_);
if (lean_obj_tag(v___x_2269_) == 0)
{
lean_object* v_a_2270_; lean_object* v___x_2283_; lean_object* v___x_2284_; lean_object* v___x_2285_; lean_object* v___x_2286_; lean_object* v___x_2287_; lean_object* v___x_2288_; lean_object* v___x_2289_; lean_object* v___x_2290_; lean_object* v___x_2291_; lean_object* v___x_2292_; lean_object* v___x_2293_; lean_object* v___x_2294_; lean_object* v___x_2295_; lean_object* v___x_2296_; lean_object* v___x_2297_; lean_object* v___x_2298_; lean_object* v___x_2299_; lean_object* v___x_2300_; lean_object* v___x_2301_; lean_object* v___x_2302_; lean_object* v___x_2303_; lean_object* v___x_2304_; lean_object* v___x_2305_; lean_object* v___x_2306_; lean_object* v___x_2307_; lean_object* v___x_2308_; lean_object* v___f_2309_; lean_object* v___x_2310_; 
v_a_2270_ = lean_ctor_get(v___x_2269_, 0);
lean_inc_n(v_a_2270_, 2);
lean_dec_ref_known(v___x_2269_, 1);
v___x_2283_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__7));
v___x_2284_ = lean_box(0);
lean_inc(v_u_2254_);
v___x_2285_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2285_, 0, v_u_2254_);
lean_ctor_set(v___x_2285_, 1, v___x_2284_);
lean_inc_ref_n(v___x_2285_, 5);
v___x_2286_ = l_Lean_Expr_const___override(v___x_2283_, v___x_2285_);
lean_inc_ref_n(v_00_u03b1_2255_, 6);
v___x_2287_ = l_Lean_Expr_app___override(v___x_2286_, v_00_u03b1_2255_);
v___x_2288_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__10));
v___x_2289_ = l_Lean_Expr_const___override(v___x_2288_, v___x_2285_);
v___x_2290_ = l_Lean_Expr_app___override(v___x_2289_, v_00_u03b1_2255_);
v___x_2291_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__13));
v___x_2292_ = l_Lean_Expr_const___override(v___x_2291_, v___x_2285_);
v___x_2293_ = l_Lean_Expr_app___override(v___x_2292_, v_00_u03b1_2255_);
v___x_2294_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__16));
v___x_2295_ = l_Lean_Expr_const___override(v___x_2294_, v___x_2285_);
v___x_2296_ = l_Lean_Expr_app___override(v___x_2295_, v_00_u03b1_2255_);
v___x_2297_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__19));
v___x_2298_ = l_Lean_Expr_const___override(v___x_2297_, v___x_2285_);
v___x_2299_ = l_Lean_Expr_app___override(v___x_2298_, v_00_u03b1_2255_);
v___x_2300_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_inv___closed__21));
v___x_2301_ = l_Lean_Expr_const___override(v___x_2300_, v___x_2285_);
v___x_2302_ = l_Lean_Expr_app___override(v___x_2301_, v_00_u03b1_2255_);
v___x_2303_ = l_Lean_Expr_app___override(v___x_2302_, v_a_2270_);
v___x_2304_ = l_Lean_Expr_app___override(v___x_2299_, v___x_2303_);
v___x_2305_ = l_Lean_Expr_app___override(v___x_2296_, v___x_2304_);
v___x_2306_ = l_Lean_Expr_app___override(v___x_2293_, v___x_2305_);
v___x_2307_ = l_Lean_Expr_app___override(v___x_2290_, v___x_2306_);
v___x_2308_ = l_Lean_Expr_app___override(v___x_2287_, v___x_2307_);
v___f_2309_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_NormNum_evalRatCast___lam__0___boxed), 7, 2);
lean_closure_set(v___f_2309_, 0, v_fn_2264_);
lean_closure_set(v___f_2309_, 1, v___x_2308_);
v___x_2310_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalRatCast_spec__0___redArg(v___f_2309_, v___x_2266_, v___y_2257_, v___y_2258_, v___y_2259_, v___y_2260_);
if (lean_obj_tag(v___x_2310_) == 0)
{
lean_object* v_a_2311_; uint8_t v___x_2312_; 
v_a_2311_ = lean_ctor_get(v___x_2310_, 0);
lean_inc(v_a_2311_);
lean_dec_ref_known(v___x_2310_, 1);
v___x_2312_ = lean_unbox(v_a_2311_);
lean_dec(v_a_2311_);
if (v___x_2312_ == 0)
{
lean_object* v___x_2313_; lean_object* v___x_2314_; lean_object* v_a_2315_; lean_object* v___x_2317_; uint8_t v_isShared_2318_; uint8_t v_isSharedCheck_2322_; 
lean_dec(v_a_2270_);
lean_dec(v_a_2268_);
lean_dec_ref(v_arg_2265_);
lean_dec_ref(v_00_u03b1_2255_);
lean_dec(v_u_2254_);
v___x_2313_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1);
v___x_2314_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0___redArg(v___x_2313_, v___y_2257_, v___y_2258_, v___y_2259_, v___y_2260_);
v_a_2315_ = lean_ctor_get(v___x_2314_, 0);
v_isSharedCheck_2322_ = !lean_is_exclusive(v___x_2314_);
if (v_isSharedCheck_2322_ == 0)
{
v___x_2317_ = v___x_2314_;
v_isShared_2318_ = v_isSharedCheck_2322_;
goto v_resetjp_2316_;
}
else
{
lean_inc(v_a_2315_);
lean_dec(v___x_2314_);
v___x_2317_ = lean_box(0);
v_isShared_2318_ = v_isSharedCheck_2322_;
goto v_resetjp_2316_;
}
v_resetjp_2316_:
{
lean_object* v___x_2320_; 
if (v_isShared_2318_ == 0)
{
v___x_2320_ = v___x_2317_;
goto v_reusejp_2319_;
}
else
{
lean_object* v_reuseFailAlloc_2321_; 
v_reuseFailAlloc_2321_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2321_, 0, v_a_2315_);
v___x_2320_ = v_reuseFailAlloc_2321_;
goto v_reusejp_2319_;
}
v_reusejp_2319_:
{
return v___x_2320_;
}
}
}
else
{
goto v___jp_2271_;
}
}
else
{
lean_object* v_a_2323_; lean_object* v___x_2325_; uint8_t v_isShared_2326_; uint8_t v_isSharedCheck_2330_; 
lean_dec(v_a_2270_);
lean_dec(v_a_2268_);
lean_dec_ref(v_arg_2265_);
lean_dec_ref(v_00_u03b1_2255_);
lean_dec(v_u_2254_);
v_a_2323_ = lean_ctor_get(v___x_2310_, 0);
v_isSharedCheck_2330_ = !lean_is_exclusive(v___x_2310_);
if (v_isSharedCheck_2330_ == 0)
{
v___x_2325_ = v___x_2310_;
v_isShared_2326_ = v_isSharedCheck_2330_;
goto v_resetjp_2324_;
}
else
{
lean_inc(v_a_2323_);
lean_dec(v___x_2310_);
v___x_2325_ = lean_box(0);
v_isShared_2326_ = v_isSharedCheck_2330_;
goto v_resetjp_2324_;
}
v_resetjp_2324_:
{
lean_object* v___x_2328_; 
if (v_isShared_2326_ == 0)
{
v___x_2328_ = v___x_2325_;
goto v_reusejp_2327_;
}
else
{
lean_object* v_reuseFailAlloc_2329_; 
v_reuseFailAlloc_2329_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2329_, 0, v_a_2323_);
v___x_2328_ = v_reuseFailAlloc_2329_;
goto v_reusejp_2327_;
}
v_reusejp_2327_:
{
return v___x_2328_;
}
}
}
v___jp_2271_:
{
lean_object* v___x_2272_; 
lean_inc(v_a_2270_);
lean_inc_ref(v_00_u03b1_2255_);
lean_inc(v_u_2254_);
v___x_2272_ = lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f(v_u_2254_, v_00_u03b1_2255_, v_a_2270_, v___y_2257_, v___y_2258_, v___y_2259_, v___y_2260_);
if (lean_obj_tag(v___x_2272_) == 0)
{
lean_object* v_a_2273_; lean_object* v___x_2274_; 
v_a_2273_ = lean_ctor_get(v___x_2272_, 0);
lean_inc(v_a_2273_);
lean_dec_ref_known(v___x_2272_, 1);
v___x_2274_ = lp_mathlib_Mathlib_Meta_NormNum_Result_inv(v_u_2254_, v_00_u03b1_2255_, v_arg_2265_, v_a_2268_, v_a_2270_, v_a_2273_, v___y_2257_, v___y_2258_, v___y_2259_, v___y_2260_);
return v___x_2274_;
}
else
{
lean_object* v_a_2275_; lean_object* v___x_2277_; uint8_t v_isShared_2278_; uint8_t v_isSharedCheck_2282_; 
lean_dec(v_a_2270_);
lean_dec(v_a_2268_);
lean_dec_ref(v_arg_2265_);
lean_dec_ref(v_00_u03b1_2255_);
lean_dec(v_u_2254_);
v_a_2275_ = lean_ctor_get(v___x_2272_, 0);
v_isSharedCheck_2282_ = !lean_is_exclusive(v___x_2272_);
if (v_isSharedCheck_2282_ == 0)
{
v___x_2277_ = v___x_2272_;
v_isShared_2278_ = v_isSharedCheck_2282_;
goto v_resetjp_2276_;
}
else
{
lean_inc(v_a_2275_);
lean_dec(v___x_2272_);
v___x_2277_ = lean_box(0);
v_isShared_2278_ = v_isSharedCheck_2282_;
goto v_resetjp_2276_;
}
v_resetjp_2276_:
{
lean_object* v___x_2280_; 
if (v_isShared_2278_ == 0)
{
v___x_2280_ = v___x_2277_;
goto v_reusejp_2279_;
}
else
{
lean_object* v_reuseFailAlloc_2281_; 
v_reuseFailAlloc_2281_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2281_, 0, v_a_2275_);
v___x_2280_ = v_reuseFailAlloc_2281_;
goto v_reusejp_2279_;
}
v_reusejp_2279_:
{
return v___x_2280_;
}
}
}
}
}
else
{
lean_object* v_a_2331_; lean_object* v___x_2333_; uint8_t v_isShared_2334_; uint8_t v_isSharedCheck_2338_; 
lean_dec(v_a_2268_);
lean_dec_ref(v_arg_2265_);
lean_dec_ref(v_fn_2264_);
lean_dec_ref(v_00_u03b1_2255_);
lean_dec(v_u_2254_);
v_a_2331_ = lean_ctor_get(v___x_2269_, 0);
v_isSharedCheck_2338_ = !lean_is_exclusive(v___x_2269_);
if (v_isSharedCheck_2338_ == 0)
{
v___x_2333_ = v___x_2269_;
v_isShared_2334_ = v_isSharedCheck_2338_;
goto v_resetjp_2332_;
}
else
{
lean_inc(v_a_2331_);
lean_dec(v___x_2269_);
v___x_2333_ = lean_box(0);
v_isShared_2334_ = v_isSharedCheck_2338_;
goto v_resetjp_2332_;
}
v_resetjp_2332_:
{
lean_object* v___x_2336_; 
if (v_isShared_2334_ == 0)
{
v___x_2336_ = v___x_2333_;
goto v_reusejp_2335_;
}
else
{
lean_object* v_reuseFailAlloc_2337_; 
v_reuseFailAlloc_2337_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2337_, 0, v_a_2331_);
v___x_2336_ = v_reuseFailAlloc_2337_;
goto v_reusejp_2335_;
}
v_reusejp_2335_:
{
return v___x_2336_;
}
}
}
}
else
{
lean_dec_ref(v_arg_2265_);
lean_dec_ref(v_fn_2264_);
lean_dec_ref(v_00_u03b1_2255_);
lean_dec(v_u_2254_);
return v___x_2267_;
}
}
else
{
lean_object* v___x_2339_; lean_object* v___x_2340_; 
lean_dec(v_a_2263_);
lean_dec_ref(v_00_u03b1_2255_);
lean_dec(v_u_2254_);
v___x_2339_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalMkRat___lam__0___closed__1);
v___x_2340_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferCharZeroOfRing_spec__0___redArg(v___x_2339_, v___y_2257_, v___y_2258_, v___y_2259_, v___y_2260_);
return v___x_2340_;
}
}
else
{
lean_object* v_a_2341_; lean_object* v___x_2343_; uint8_t v_isShared_2344_; uint8_t v_isSharedCheck_2348_; 
lean_dec_ref(v_00_u03b1_2255_);
lean_dec(v_u_2254_);
v_a_2341_ = lean_ctor_get(v___x_2262_, 0);
v_isSharedCheck_2348_ = !lean_is_exclusive(v___x_2262_);
if (v_isSharedCheck_2348_ == 0)
{
v___x_2343_ = v___x_2262_;
v_isShared_2344_ = v_isSharedCheck_2348_;
goto v_resetjp_2342_;
}
else
{
lean_inc(v_a_2341_);
lean_dec(v___x_2262_);
v___x_2343_ = lean_box(0);
v_isShared_2344_ = v_isSharedCheck_2348_;
goto v_resetjp_2342_;
}
v_resetjp_2342_:
{
lean_object* v___x_2346_; 
if (v_isShared_2344_ == 0)
{
v___x_2346_ = v___x_2343_;
goto v_reusejp_2345_;
}
else
{
lean_object* v_reuseFailAlloc_2347_; 
v_reuseFailAlloc_2347_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2347_, 0, v_a_2341_);
v___x_2346_ = v_reuseFailAlloc_2347_;
goto v_reusejp_2345_;
}
v_reusejp_2345_:
{
return v___x_2346_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalInv___lam__1___boxed(lean_object* v_u_2349_, lean_object* v_00_u03b1_2350_, lean_object* v_e_2351_, lean_object* v___y_2352_, lean_object* v___y_2353_, lean_object* v___y_2354_, lean_object* v___y_2355_, lean_object* v___y_2356_){
_start:
{
lean_object* v_res_2357_; 
v_res_2357_ = lp_mathlib_Mathlib_Meta_NormNum_evalInv___lam__1(v_u_2349_, v_00_u03b1_2350_, v_e_2351_, v___y_2352_, v___y_2353_, v___y_2354_, v___y_2355_);
lean_dec(v___y_2355_);
lean_dec_ref(v___y_2354_);
lean_dec(v___y_2353_);
lean_dec_ref(v___y_2352_);
return v_res_2357_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Rat_Cast_CharZero(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Inv(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Rat_Cast_CharZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_NormNum_Inv(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1 = _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing___auto__1);
lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing_x3f___auto__1 = _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing_x3f___auto__1();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing_x3f___auto__1);
lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfAddMonoidWithOne___auto__1 = _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfAddMonoidWithOne___auto__1();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfAddMonoidWithOne___auto__1);
lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfAddMonoidWithOne_x3f___auto__1 = _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfAddMonoidWithOne_x3f___auto__1();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfAddMonoidWithOne_x3f___auto__1);
lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing___auto__1 = _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing___auto__1();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing___auto__1);
lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___auto__1 = _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___auto__1();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f___auto__1);
lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing_x3f___auto__1 = _init_lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing_x3f___auto__1();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing_x3f___auto__1);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Rat_Cast_CharZero(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_NormNum_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_NormNum_Inv(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Rat_Cast_CharZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_NormNum_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Inv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_NormNum_Inv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_NormNum_Inv(builtin);
}
#ifdef __cplusplus
}
#endif
