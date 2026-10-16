// Lean compiler output
// Module: Mathlib.Tactic.NormNum.Pow
// Imports: public import Init public meta import Init public import Mathlib.Data.Int.Cast.Lemmas public import Mathlib.Tactic.NormNum.Basic public import Mathlib.Util.Qq
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
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Expr_lit___override(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_Meta_whnfR(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Level_succ___override(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* lp_Qq_Qq_synthInstanceQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_derive(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_Qq_Qq_QuotedDefEq_check___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isRat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_Lean_Expr_natLit_x21(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* l_Lean_mkRawNatLit(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_land(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* l_Lean_checkExponent(lean_object*, uint8_t, lean_object*, lean_object*);
lean_object* lean_nat_log2(lean_object*);
lean_object* l_Lean_Expr_forallE___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toInt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_appArg_x21(lean_object*);
lean_object* lean_nat_to_int(lean_object*);
uint8_t lean_int_dec_le(lean_object*, lean_object*);
lean_object* lean_int_neg(lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isInt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_mkRat(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__2;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__3;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__4;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__6_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__7;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__8;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "pow"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__10_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__9_value),LEAN_SCALAR_PTR_LITERAL(155, 64, 52, 77, 166, 227, 131, 174)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__10_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__11;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__12_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__13_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "NormNum"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__14_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "IsNatPowT"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__15_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trans"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__16_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__17_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__17_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__13_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__17_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__17_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__17_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__17_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__15_value),LEAN_SCALAR_PTR_LITERAL(25, 110, 83, 36, 199, 102, 50, 3)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__17_value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(235, 151, 249, 160, 178, 191, 173, 97)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__17_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__18;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "bit1"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__19_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__20_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__20_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__13_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__20_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__20_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__20_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__20_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__15_value),LEAN_SCALAR_PTR_LITERAL(25, 110, 83, 36, 199, 102, 50, 3)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__20_value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__19_value),LEAN_SCALAR_PTR_LITERAL(242, 27, 44, 226, 40, 32, 76, 85)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__20_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__21;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "bit0"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__22_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__23_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__23_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__23_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__13_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__23_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__23_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__23_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__23_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__15_value),LEAN_SCALAR_PTR_LITERAL(25, 110, 83, 36, 199, 102, 50, 3)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__23_value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__22_value),LEAN_SCALAR_PTR_LITERAL(234, 86, 46, 227, 88, 93, 232, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__23 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__23_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__24;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "run"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__13_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__3_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__3_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__3_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__15_value),LEAN_SCALAR_PTR_LITERAL(25, 110, 83, 36, 199, 102, 50, 3)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__3_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__2_value),LEAN_SCALAR_PTR_LITERAL(247, 14, 73, 108, 79, 252, 180, 37)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__4;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "natPow_one"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__6_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__13_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__6_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__6_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__5_value),LEAN_SCALAR_PTR_LITERAL(237, 72, 246, 207, 144, 117, 183, 210)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__7;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "one_natPow"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__9_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__13_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__9_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__9_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__9_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__8_value),LEAN_SCALAR_PTR_LITERAL(91, 155, 71, 63, 121, 231, 136, 61)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__9_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__10;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__11_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__12;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "zero_natPow"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__14_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__13_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__14_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__14_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__14_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__13_value),LEAN_SCALAR_PTR_LITERAL(47, 32, 29, 118, 104, 53, 214, 114)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__14_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__15;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "natPow_zero"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__17_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__17_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__13_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__17_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__17_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__17_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__16_value),LEAN_SCALAR_PTR_LITERAL(36, 77, 124, 10, 215, 97, 42, 30)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__17_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__18;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPow(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Mathlib_Meta_NormNum_evalIntPow_spec__0(lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__0;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "refl"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__1_value),LEAN_SCALAR_PTR_LITERAL(72, 6, 107, 181, 0, 125, 21, 187)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__3;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__4;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Int"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "negOfNat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__5_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__6_value),LEAN_SCALAR_PTR_LITERAL(100, 231, 152, 184, 84, 220, 144, 243)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__8;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "intPow_negOfNat_bit1"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__10_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__13_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__10_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__10_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__9_value),LEAN_SCALAR_PTR_LITERAL(189, 225, 75, 30, 58, 48, 37, 30)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__10_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__11;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ofNat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__5_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__13_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__12_value),LEAN_SCALAR_PTR_LITERAL(192, 66, 133, 102, 95, 170, 134, 92)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__13_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__14;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "intPow_negOfNat_bit0"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__16_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__13_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__16_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__16_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__16_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__15_value),LEAN_SCALAR_PTR_LITERAL(5, 237, 29, 68, 179, 73, 34, 171)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__16_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__17;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "intPow_ofNat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__19_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__19_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__13_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__19_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__19_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__19_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__18_value),LEAN_SCALAR_PTR_LITERAL(132, 225, 179, 61, 81, 223, 186, 143)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__19_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__20;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntPow(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "isNat_pow"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__13_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__0_value),LEAN_SCALAR_PTR_LITERAL(112, 201, 124, 92, 42, 240, 48, 227)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "isInt_pow"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__13_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__3_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__2_value),LEAN_SCALAR_PTR_LITERAL(102, 124, 229, 85, 154, 89, 161, 1)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "isNNRat_pow"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__5_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__13_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__5_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__5_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__4_value),LEAN_SCALAR_PTR_LITERAL(115, 202, 38, 169, 239, 95, 52, 237)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "DivisionRing"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "toRing"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__6_value),LEAN_SCALAR_PTR_LITERAL(34, 214, 17, 155, 7, 71, 232, 190)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__7_value),LEAN_SCALAR_PTR_LITERAL(196, 15, 37, 9, 106, 139, 236, 93)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "isRat_pow"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__10_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__13_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__10_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__10_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__9_value),LEAN_SCALAR_PTR_LITERAL(251, 125, 199, 248, 200, 166, 70, 212)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__10_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow_core(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalPow_spec__1___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalPow_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalPow_spec__1(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalPow_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_evalPow_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_evalPow_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalPow_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalPow_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__1;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HPow"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hPow"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(155, 188, 136, 200, 106, 253, 76, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(32, 63, 208, 57, 56, 184, 164, 144)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instHPow"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(213, 197, 76, 235, 199, 0, 254, 199)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "NPow"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toPow"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(39, 79, 240, 225, 164, 207, 253, 237)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(56, 108, 173, 227, 4, 14, 173, 115)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Monoid"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "toNPow"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(162, 147, 2, 115, 233, 179, 113, 5)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__13_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(224, 31, 132, 245, 47, 70, 119, 231)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Semiring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "toMonoid"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(37, 127, 172, 14, 25, 240, 239, 179)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__16_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(86, 172, 133, 187, 121, 84, 206, 170)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__16_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_NormNum_evalPow___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalPow___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "evalPow"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalPow___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalPow___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__13_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalPow___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___closed__2_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalPow___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___closed__1_value),LEAN_SCALAR_PTR_LITERAL(20, 127, 10, 12, 24, 242, 24, 142)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalPow___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___closed__2_value),LEAN_SCALAR_PTR_LITERAL(1, 1, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___closed__3_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalPow_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalPow_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "DivisionSemiring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 208, 64, 71, 63, 26, 215, 130)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__5_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "ZPow"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(73, 207, 245, 197, 62, 206, 208, 46)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(110, 230, 127, 66, 154, 153, 82, 195)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "DivInvMonoid"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "toZPow"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__6_value),LEAN_SCALAR_PTR_LITERAL(231, 106, 236, 89, 112, 21, 122, 113)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__7_value),LEAN_SCALAR_PTR_LITERAL(128, 241, 50, 125, 164, 95, 147, 60)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "GroupWithZero"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toDivInvMonoid"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__9_value),LEAN_SCALAR_PTR_LITERAL(55, 132, 4, 209, 65, 207, 153, 53)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__10_value),LEAN_SCALAR_PTR_LITERAL(172, 54, 25, 155, 165, 99, 150, 23)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "toGroupWithZero"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 208, 64, 71, 63, 26, 215, 130)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__13_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__12_value),LEAN_SCALAR_PTR_LITERAL(164, 129, 71, 97, 30, 189, 214, 64)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "toSemiring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 208, 64, 71, 63, 26, 215, 130)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__15_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__14_value),LEAN_SCALAR_PTR_LITERAL(241, 35, 131, 210, 203, 216, 146, 177)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "isNat_zpow_pos"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__17_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__17_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__13_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__17_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__17_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__17_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__16_value),LEAN_SCALAR_PTR_LITERAL(195, 150, 29, 128, 69, 252, 236, 63)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__6_value),LEAN_SCALAR_PTR_LITERAL(34, 214, 17, 155, 7, 71, 232, 190)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "isInt_zpow_pos"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__20_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__20_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__13_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__20_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__20_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__20_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__19_value),LEAN_SCALAR_PTR_LITERAL(64, 7, 97, 219, 153, 27, 68, 115)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__20_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "isNNRat_zpow_pos"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__22_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__22_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__22_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__13_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__22_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__22_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__22_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__21_value),LEAN_SCALAR_PTR_LITERAL(54, 112, 104, 80, 119, 46, 144, 72)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__22_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "isRat_zpow_pos"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__23_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__24_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__24_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__24_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__13_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__24_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__24_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__24_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__23_value),LEAN_SCALAR_PTR_LITERAL(28, 145, 174, 240, 18, 162, 60, 162)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__24_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Inv"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__25_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "inv"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__27_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__25_value),LEAN_SCALAR_PTR_LITERAL(142, 68, 231, 210, 96, 163, 154, 19)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__27_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__26_value),LEAN_SCALAR_PTR_LITERAL(63, 31, 248, 222, 13, 64, 40, 141)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__27_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "InvOneClass"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__28_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toInv"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__29_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__30_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__28_value),LEAN_SCALAR_PTR_LITERAL(120, 190, 7, 179, 62, 236, 21, 116)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__30_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__29_value),LEAN_SCALAR_PTR_LITERAL(28, 25, 248, 9, 15, 85, 72, 194)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__30_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "DivInvOneMonoid"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__31_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "toInvOneClass"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__32_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__33_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__31_value),LEAN_SCALAR_PTR_LITERAL(162, 155, 123, 0, 237, 243, 28, 65)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__33_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__32_value),LEAN_SCALAR_PTR_LITERAL(181, 224, 200, 199, 184, 130, 54, 26)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__33_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "DivisionMonoid"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__34_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "toDivInvOneMonoid"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__35_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__36_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__34_value),LEAN_SCALAR_PTR_LITERAL(16, 242, 184, 157, 107, 26, 18, 78)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__36_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__35_value),LEAN_SCALAR_PTR_LITERAL(60, 63, 43, 77, 240, 6, 89, 70)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__36_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "toDivisionMonoid"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__37_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__38_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__9_value),LEAN_SCALAR_PTR_LITERAL(55, 132, 4, 209, 65, 207, 153, 53)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__38_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__37_value),LEAN_SCALAR_PTR_LITERAL(198, 76, 78, 187, 42, 89, 29, 20)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__38_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "isNat_zpow_neg"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__39 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__39_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__40_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__40_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__40_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__13_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__40_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__40_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__40_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__39_value),LEAN_SCALAR_PTR_LITERAL(189, 216, 239, 38, 161, 243, 39, 234)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__40 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__40_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "isInt_zpow_neg"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__41 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__41_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__42_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__42_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__42_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__13_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__42_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__42_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__42_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__41_value),LEAN_SCALAR_PTR_LITERAL(127, 77, 207, 121, 61, 20, 73, 208)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__42 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__42_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "isNNRat_zpow_neg"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__43 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__43_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__44_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__44_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__44_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__13_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__44_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__44_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__44_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__43_value),LEAN_SCALAR_PTR_LITERAL(15, 129, 3, 41, 145, 219, 233, 172)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__44 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__44_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "isRat_zpow_neg"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__45 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__45_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__46_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__46_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__46_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__13_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__46_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__46_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__46_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__45_value),LEAN_SCALAR_PTR_LITERAL(157, 201, 146, 217, 100, 189, 245, 149)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__46 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__46_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "evalZPow"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__13_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___closed__2_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___closed__1_value),LEAN_SCALAR_PTR_LITERAL(131, 76, 3, 162, 6, 40, 205, 22)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalZPow___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___closed__2_value),LEAN_SCALAR_PTR_LITERAL(1, 1, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___closed__3_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___closed__3_value;
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__2(void){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; 
v___x_4_ = lean_box(0);
v___x_5_ = l_Lean_Level_succ___override(v___x_4_);
return v___x_5_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__3(void){
_start:
{
lean_object* v___x_6_; lean_object* v___x_7_; lean_object* v___x_8_; 
v___x_6_ = lean_box(0);
v___x_7_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__2, &lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__2);
v___x_8_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_8_, 0, v___x_7_);
lean_ctor_set(v___x_8_, 1, v___x_6_);
return v___x_8_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__4(void){
_start:
{
lean_object* v___x_9_; lean_object* v___x_10_; lean_object* v___x_11_; 
v___x_9_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__3, &lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__3);
v___x_10_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__1));
v___x_11_ = l_Lean_Expr_const___override(v___x_10_, v___x_9_);
return v___x_11_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__7(void){
_start:
{
lean_object* v___x_15_; lean_object* v___x_16_; lean_object* v___x_17_; 
v___x_15_ = lean_box(0);
v___x_16_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__6));
v___x_17_ = l_Lean_Expr_const___override(v___x_16_, v___x_15_);
return v___x_17_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__8(void){
_start:
{
lean_object* v___x_18_; lean_object* v___x_19_; lean_object* v___x_20_; 
v___x_18_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__7, &lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__7);
v___x_19_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__4, &lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__4);
v___x_20_ = l_Lean_Expr_app___override(v___x_19_, v___x_18_);
return v___x_20_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__11(void){
_start:
{
lean_object* v___x_25_; lean_object* v___x_26_; lean_object* v___x_27_; 
v___x_25_ = lean_box(0);
v___x_26_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__10));
v___x_27_ = l_Lean_Expr_const___override(v___x_26_, v___x_25_);
return v___x_27_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__18(void){
_start:
{
lean_object* v___x_39_; lean_object* v___x_40_; lean_object* v___x_41_; 
v___x_39_ = lean_box(0);
v___x_40_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__17));
v___x_41_ = l_Lean_Expr_const___override(v___x_40_, v___x_39_);
return v___x_41_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__21(void){
_start:
{
lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; 
v___x_49_ = lean_box(0);
v___x_50_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__20));
v___x_51_ = l_Lean_Expr_const___override(v___x_50_, v___x_49_);
return v___x_51_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__24(void){
_start:
{
lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; 
v___x_59_ = lean_box(0);
v___x_60_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__23));
v___x_61_ = l_Lean_Expr_const___override(v___x_60_, v___x_59_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg(lean_object* v_depth_62_, lean_object* v_a_63_, lean_object* v_b_u2080_64_, lean_object* v_c_u2080_65_, lean_object* v_b_66_, lean_object* v_p_67_){
_start:
{
lean_object* v_b_x27_68_; lean_object* v___x_69_; uint8_t v___x_70_; 
v_b_x27_68_ = lp_batteries_Lean_Expr_natLit_x21(v_b_66_);
v___x_69_ = lean_unsigned_to_nat(1u);
v___x_70_ = lean_nat_dec_le(v_depth_62_, v___x_69_);
if (v___x_70_ == 0)
{
lean_object* v_d_71_; lean_object* v___x_72_; lean_object* v_hi_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v_fst_76_; lean_object* v_snd_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v_fst_85_; lean_object* v_snd_86_; lean_object* v___x_88_; uint8_t v_isShared_89_; uint8_t v_isSharedCheck_102_; 
v_d_71_ = lean_nat_shiftr(v_depth_62_, v___x_69_);
v___x_72_ = lean_nat_shiftr(v_b_x27_68_, v_d_71_);
lean_dec(v_b_x27_68_);
v_hi_73_ = l_Lean_mkRawNatLit(v___x_72_);
v___x_74_ = lean_nat_sub(v_depth_62_, v_d_71_);
lean_inc_ref(v_p_67_);
lean_inc_ref_n(v_hi_73_, 3);
lean_inc_ref_n(v_a_63_, 3);
v___x_75_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg(v___x_74_, v_a_63_, v_b_u2080_64_, v_c_u2080_65_, v_hi_73_, v_p_67_);
lean_dec(v___x_74_);
v_fst_76_ = lean_ctor_get(v___x_75_, 0);
lean_inc_n(v_fst_76_, 3);
v_snd_77_ = lean_ctor_get(v___x_75_, 1);
lean_inc(v_snd_77_);
lean_dec_ref(v___x_75_);
v___x_78_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__8, &lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__8_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__8);
v___x_79_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__11, &lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__11_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__11);
v___x_80_ = l_Lean_Expr_app___override(v___x_79_, v_a_63_);
v___x_81_ = l_Lean_Expr_app___override(v___x_80_, v_hi_73_);
v___x_82_ = l_Lean_Expr_app___override(v___x_78_, v___x_81_);
v___x_83_ = l_Lean_Expr_app___override(v___x_82_, v_fst_76_);
lean_inc_ref(v_b_66_);
v___x_84_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg(v_d_71_, v_a_63_, v_hi_73_, v_fst_76_, v_b_66_, v___x_83_);
lean_dec(v_d_71_);
v_fst_85_ = lean_ctor_get(v___x_84_, 0);
v_snd_86_ = lean_ctor_get(v___x_84_, 1);
v_isSharedCheck_102_ = !lean_is_exclusive(v___x_84_);
if (v_isSharedCheck_102_ == 0)
{
v___x_88_ = v___x_84_;
v_isShared_89_ = v_isSharedCheck_102_;
goto v_resetjp_87_;
}
else
{
lean_inc(v_snd_86_);
lean_inc(v_fst_85_);
lean_dec(v___x_84_);
v___x_88_ = lean_box(0);
v_isShared_89_ = v_isSharedCheck_102_;
goto v_resetjp_87_;
}
v_resetjp_87_:
{
lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_100_; 
v___x_90_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__18, &lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__18_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__18);
v___x_91_ = l_Lean_Expr_app___override(v___x_90_, v_a_63_);
v___x_92_ = l_Lean_Expr_app___override(v___x_91_, v_hi_73_);
v___x_93_ = l_Lean_Expr_app___override(v___x_92_, v_fst_76_);
v___x_94_ = l_Lean_Expr_app___override(v___x_93_, v_p_67_);
v___x_95_ = l_Lean_Expr_app___override(v___x_94_, v_b_66_);
lean_inc(v_fst_85_);
v___x_96_ = l_Lean_Expr_app___override(v___x_95_, v_fst_85_);
v___x_97_ = l_Lean_Expr_app___override(v___x_96_, v_snd_77_);
v___x_98_ = l_Lean_Expr_app___override(v___x_97_, v_snd_86_);
if (v_isShared_89_ == 0)
{
lean_ctor_set(v___x_88_, 1, v___x_98_);
v___x_100_ = v___x_88_;
goto v_reusejp_99_;
}
else
{
lean_object* v_reuseFailAlloc_101_; 
v_reuseFailAlloc_101_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_101_, 0, v_fst_85_);
lean_ctor_set(v_reuseFailAlloc_101_, 1, v___x_98_);
v___x_100_ = v_reuseFailAlloc_101_;
goto v_reusejp_99_;
}
v_reusejp_99_:
{
return v___x_100_;
}
}
}
else
{
lean_object* v_c_u2080_x27_103_; lean_object* v___x_104_; lean_object* v___x_105_; uint8_t v___x_106_; 
lean_dec_ref(v_p_67_);
lean_dec_ref(v_b_66_);
v_c_u2080_x27_103_ = lp_batteries_Lean_Expr_natLit_x21(v_c_u2080_65_);
v___x_104_ = lean_nat_land(v_b_x27_68_, v___x_69_);
lean_dec(v_b_x27_68_);
v___x_105_ = lean_unsigned_to_nat(0u);
v___x_106_ = lean_nat_dec_eq(v___x_104_, v___x_105_);
lean_dec(v___x_104_);
if (v___x_106_ == 0)
{
lean_object* v_a_x27_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v_c_110_; lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; 
v_a_x27_107_ = lp_batteries_Lean_Expr_natLit_x21(v_a_63_);
v___x_108_ = lean_nat_mul(v_c_u2080_x27_103_, v_a_x27_107_);
lean_dec(v_a_x27_107_);
v___x_109_ = lean_nat_mul(v_c_u2080_x27_103_, v___x_108_);
lean_dec(v___x_108_);
lean_dec(v_c_u2080_x27_103_);
v_c_110_ = l_Lean_mkRawNatLit(v___x_109_);
v___x_111_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__21, &lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__21_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__21);
v___x_112_ = l_Lean_Expr_app___override(v___x_111_, v_a_63_);
v___x_113_ = l_Lean_Expr_app___override(v___x_112_, v_b_u2080_64_);
v___x_114_ = l_Lean_Expr_app___override(v___x_113_, v_c_u2080_65_);
v___x_115_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_115_, 0, v_c_110_);
lean_ctor_set(v___x_115_, 1, v___x_114_);
return v___x_115_;
}
else
{
lean_object* v___x_116_; lean_object* v_c_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; 
v___x_116_ = lean_nat_mul(v_c_u2080_x27_103_, v_c_u2080_x27_103_);
lean_dec(v_c_u2080_x27_103_);
v_c_117_ = l_Lean_mkRawNatLit(v___x_116_);
v___x_118_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__24, &lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__24_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__24);
v___x_119_ = l_Lean_Expr_app___override(v___x_118_, v_a_63_);
v___x_120_ = l_Lean_Expr_app___override(v___x_119_, v_b_u2080_64_);
v___x_121_ = l_Lean_Expr_app___override(v___x_120_, v_c_u2080_65_);
v___x_122_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_122_, 0, v_c_117_);
lean_ctor_set(v___x_122_, 1, v___x_121_);
return v___x_122_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___boxed(lean_object* v_depth_123_, lean_object* v_a_124_, lean_object* v_b_u2080_125_, lean_object* v_c_u2080_126_, lean_object* v_b_127_, lean_object* v_p_128_){
_start:
{
lean_object* v_res_129_; 
v_res_129_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg(v_depth_123_, v_a_124_, v_b_u2080_125_, v_c_u2080_126_, v_b_127_, v_p_128_);
lean_dec(v_depth_123_);
return v_res_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go(lean_object* v_depth_130_, lean_object* v_a_131_, lean_object* v_b_u2080_132_, lean_object* v_c_u2080_133_, lean_object* v_b_134_, lean_object* v_p_135_, lean_object* v_hp_136_){
_start:
{
lean_object* v___x_137_; 
v___x_137_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg(v_depth_130_, v_a_131_, v_b_u2080_132_, v_c_u2080_133_, v_b_134_, v_p_135_);
return v___x_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___boxed(lean_object* v_depth_138_, lean_object* v_a_139_, lean_object* v_b_u2080_140_, lean_object* v_c_u2080_141_, lean_object* v_b_142_, lean_object* v_p_143_, lean_object* v_hp_144_){
_start:
{
lean_object* v_res_145_; 
v_res_145_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go(v_depth_138_, v_a_139_, v_b_u2080_140_, v_c_u2080_141_, v_b_142_, v_p_143_, v_hp_144_);
lean_dec(v_depth_138_);
return v_res_145_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__1(void){
_start:
{
lean_object* v___x_148_; lean_object* v___x_149_; 
v___x_148_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__0));
v___x_149_ = l_Lean_Expr_lit___override(v___x_148_);
return v___x_149_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__4(void){
_start:
{
lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; 
v___x_157_ = lean_box(0);
v___x_158_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__3));
v___x_159_ = l_Lean_Expr_const___override(v___x_158_, v___x_157_);
return v___x_159_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__7(void){
_start:
{
lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; 
v___x_166_ = lean_box(0);
v___x_167_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__6));
v___x_168_ = l_Lean_Expr_const___override(v___x_167_, v___x_166_);
return v___x_168_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__10(void){
_start:
{
lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; 
v___x_175_ = lean_box(0);
v___x_176_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__9));
v___x_177_ = l_Lean_Expr_const___override(v___x_176_, v___x_175_);
return v___x_177_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__12(void){
_start:
{
lean_object* v___x_180_; lean_object* v___x_181_; 
v___x_180_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__11));
v___x_181_ = l_Lean_Expr_lit___override(v___x_180_);
return v___x_181_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__15(void){
_start:
{
lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; 
v___x_188_ = lean_box(0);
v___x_189_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__14));
v___x_190_ = l_Lean_Expr_const___override(v___x_189_, v___x_188_);
return v___x_190_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__18(void){
_start:
{
lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; 
v___x_197_ = lean_box(0);
v___x_198_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__17));
v___x_199_ = l_Lean_Expr_const___override(v___x_198_, v___x_197_);
return v___x_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPow(lean_object* v_a_200_, lean_object* v_b_201_, lean_object* v_a_202_, lean_object* v_a_203_){
_start:
{
lean_object* v___x_205_; lean_object* v___x_206_; uint8_t v___x_207_; 
v___x_205_ = lp_batteries_Lean_Expr_natLit_x21(v_b_201_);
v___x_206_ = lean_unsigned_to_nat(0u);
v___x_207_ = lean_nat_dec_eq(v___x_205_, v___x_206_);
if (v___x_207_ == 0)
{
lean_object* v___x_208_; uint8_t v___x_209_; 
v___x_208_ = lp_batteries_Lean_Expr_natLit_x21(v_a_200_);
v___x_209_ = lean_nat_dec_eq(v___x_208_, v___x_206_);
if (v___x_209_ == 0)
{
lean_object* v___x_210_; uint8_t v___x_211_; 
v___x_210_ = lean_unsigned_to_nat(1u);
v___x_211_ = lean_nat_dec_eq(v___x_208_, v___x_210_);
lean_dec(v___x_208_);
if (v___x_211_ == 0)
{
uint8_t v___x_212_; 
v___x_212_ = lean_nat_dec_eq(v___x_205_, v___x_210_);
if (v___x_212_ == 0)
{
uint8_t v___x_213_; lean_object* v___x_214_; 
v___x_213_ = 1;
lean_inc(v___x_205_);
v___x_214_ = l_Lean_checkExponent(v___x_205_, v___x_213_, v_a_202_, v_a_203_);
if (lean_obj_tag(v___x_214_) == 0)
{
lean_object* v_a_215_; lean_object* v___x_217_; uint8_t v_isShared_218_; uint8_t v_isSharedCheck_251_; 
v_a_215_ = lean_ctor_get(v___x_214_, 0);
v_isSharedCheck_251_ = !lean_is_exclusive(v___x_214_);
if (v_isSharedCheck_251_ == 0)
{
v___x_217_ = v___x_214_;
v_isShared_218_ = v_isSharedCheck_251_;
goto v_resetjp_216_;
}
else
{
lean_inc(v_a_215_);
lean_dec(v___x_214_);
v___x_217_ = lean_box(0);
v_isShared_218_ = v_isSharedCheck_251_;
goto v_resetjp_216_;
}
v_resetjp_216_:
{
uint8_t v___x_219_; 
v___x_219_ = lean_unbox(v_a_215_);
lean_dec(v_a_215_);
if (v___x_219_ == 0)
{
lean_object* v___x_220_; lean_object* v___x_222_; 
lean_dec(v___x_205_);
lean_dec_ref(v_b_201_);
lean_dec_ref(v_a_200_);
v___x_220_ = lean_box(0);
if (v_isShared_218_ == 0)
{
lean_ctor_set(v___x_217_, 0, v___x_220_);
v___x_222_ = v___x_217_;
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
lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v_fst_233_; lean_object* v_snd_234_; lean_object* v___x_236_; uint8_t v_isShared_237_; uint8_t v_isSharedCheck_250_; 
v___x_224_ = lean_nat_log2(v___x_205_);
lean_dec(v___x_205_);
v___x_225_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__1);
v___x_226_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__8, &lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__8_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__8);
v___x_227_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__11, &lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__11_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__11);
lean_inc_ref_n(v_a_200_, 4);
v___x_228_ = l_Lean_Expr_app___override(v___x_227_, v_a_200_);
v___x_229_ = l_Lean_Expr_app___override(v___x_228_, v___x_225_);
v___x_230_ = l_Lean_Expr_app___override(v___x_226_, v___x_229_);
v___x_231_ = l_Lean_Expr_app___override(v___x_230_, v_a_200_);
lean_inc_ref(v_b_201_);
v___x_232_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg(v___x_224_, v_a_200_, v___x_225_, v_a_200_, v_b_201_, v___x_231_);
lean_dec(v___x_224_);
v_fst_233_ = lean_ctor_get(v___x_232_, 0);
v_snd_234_ = lean_ctor_get(v___x_232_, 1);
v_isSharedCheck_250_ = !lean_is_exclusive(v___x_232_);
if (v_isSharedCheck_250_ == 0)
{
v___x_236_ = v___x_232_;
v_isShared_237_ = v_isSharedCheck_250_;
goto v_resetjp_235_;
}
else
{
lean_inc(v_snd_234_);
lean_inc(v_fst_233_);
lean_dec(v___x_232_);
v___x_236_ = lean_box(0);
v_isShared_237_ = v_isSharedCheck_250_;
goto v_resetjp_235_;
}
v_resetjp_235_:
{
lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_244_; 
v___x_238_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__4, &lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__4_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__4);
v___x_239_ = l_Lean_Expr_app___override(v___x_238_, v_a_200_);
v___x_240_ = l_Lean_Expr_app___override(v___x_239_, v_b_201_);
lean_inc(v_fst_233_);
v___x_241_ = l_Lean_Expr_app___override(v___x_240_, v_fst_233_);
v___x_242_ = l_Lean_Expr_app___override(v___x_241_, v_snd_234_);
if (v_isShared_237_ == 0)
{
lean_ctor_set(v___x_236_, 1, v___x_242_);
v___x_244_ = v___x_236_;
goto v_reusejp_243_;
}
else
{
lean_object* v_reuseFailAlloc_249_; 
v_reuseFailAlloc_249_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_249_, 0, v_fst_233_);
lean_ctor_set(v_reuseFailAlloc_249_, 1, v___x_242_);
v___x_244_ = v_reuseFailAlloc_249_;
goto v_reusejp_243_;
}
v_reusejp_243_:
{
lean_object* v___x_245_; lean_object* v___x_247_; 
v___x_245_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_245_, 0, v___x_244_);
if (v_isShared_218_ == 0)
{
lean_ctor_set(v___x_217_, 0, v___x_245_);
v___x_247_ = v___x_217_;
goto v_reusejp_246_;
}
else
{
lean_object* v_reuseFailAlloc_248_; 
v_reuseFailAlloc_248_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_248_, 0, v___x_245_);
v___x_247_ = v_reuseFailAlloc_248_;
goto v_reusejp_246_;
}
v_reusejp_246_:
{
return v___x_247_;
}
}
}
}
}
}
else
{
lean_object* v_a_252_; lean_object* v___x_254_; uint8_t v_isShared_255_; uint8_t v_isSharedCheck_259_; 
lean_dec(v___x_205_);
lean_dec_ref(v_b_201_);
lean_dec_ref(v_a_200_);
v_a_252_ = lean_ctor_get(v___x_214_, 0);
v_isSharedCheck_259_ = !lean_is_exclusive(v___x_214_);
if (v_isSharedCheck_259_ == 0)
{
v___x_254_ = v___x_214_;
v_isShared_255_ = v_isSharedCheck_259_;
goto v_resetjp_253_;
}
else
{
lean_inc(v_a_252_);
lean_dec(v___x_214_);
v___x_254_ = lean_box(0);
v_isShared_255_ = v_isSharedCheck_259_;
goto v_resetjp_253_;
}
v_resetjp_253_:
{
lean_object* v___x_257_; 
if (v_isShared_255_ == 0)
{
v___x_257_ = v___x_254_;
goto v_reusejp_256_;
}
else
{
lean_object* v_reuseFailAlloc_258_; 
v_reuseFailAlloc_258_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_258_, 0, v_a_252_);
v___x_257_ = v_reuseFailAlloc_258_;
goto v_reusejp_256_;
}
v_reusejp_256_:
{
return v___x_257_;
}
}
}
}
else
{
lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v___x_263_; lean_object* v___x_264_; 
lean_dec(v___x_205_);
lean_dec_ref(v_b_201_);
v___x_260_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__7, &lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__7_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__7);
lean_inc_ref(v_a_200_);
v___x_261_ = l_Lean_Expr_app___override(v___x_260_, v_a_200_);
v___x_262_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_262_, 0, v_a_200_);
lean_ctor_set(v___x_262_, 1, v___x_261_);
v___x_263_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_263_, 0, v___x_262_);
v___x_264_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_264_, 0, v___x_263_);
return v___x_264_;
}
}
else
{
lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; 
lean_dec(v___x_205_);
lean_dec_ref(v_a_200_);
v___x_265_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__1);
v___x_266_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__10, &lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__10_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__10);
v___x_267_ = l_Lean_Expr_app___override(v___x_266_, v_b_201_);
v___x_268_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_268_, 0, v___x_265_);
lean_ctor_set(v___x_268_, 1, v___x_267_);
v___x_269_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_269_, 0, v___x_268_);
v___x_270_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_270_, 0, v___x_269_);
return v___x_270_;
}
}
else
{
lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v_b_x27_273_; lean_object* v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; lean_object* v___x_278_; lean_object* v___x_279_; 
lean_dec(v___x_208_);
lean_dec_ref(v_b_201_);
lean_dec_ref(v_a_200_);
v___x_271_ = lean_unsigned_to_nat(1u);
v___x_272_ = lean_nat_sub(v___x_205_, v___x_271_);
lean_dec(v___x_205_);
v_b_x27_273_ = l_Lean_mkRawNatLit(v___x_272_);
v___x_274_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__12, &lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__12_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__12);
v___x_275_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__15, &lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__15_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__15);
v___x_276_ = l_Lean_Expr_app___override(v___x_275_, v_b_x27_273_);
v___x_277_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_277_, 0, v___x_274_);
lean_ctor_set(v___x_277_, 1, v___x_276_);
v___x_278_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_278_, 0, v___x_277_);
v___x_279_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_279_, 0, v___x_278_);
return v___x_279_;
}
}
else
{
lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; lean_object* v___x_285_; 
lean_dec(v___x_205_);
lean_dec_ref(v_b_201_);
v___x_280_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__1);
v___x_281_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__18, &lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__18_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___closed__18);
v___x_282_ = l_Lean_Expr_app___override(v___x_281_, v_a_200_);
v___x_283_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_283_, 0, v___x_280_);
lean_ctor_set(v___x_283_, 1, v___x_282_);
v___x_284_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_284_, 0, v___x_283_);
v___x_285_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_285_, 0, v___x_284_);
return v___x_285_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPow___boxed(lean_object* v_a_286_, lean_object* v_b_287_, lean_object* v_a_288_, lean_object* v_a_289_, lean_object* v_a_290_){
_start:
{
lean_object* v_res_291_; 
v_res_291_ = lp_mathlib_Mathlib_Meta_NormNum_evalNatPow(v_a_286_, v_b_287_, v_a_288_, v_a_289_);
lean_dec(v_a_289_);
lean_dec_ref(v_a_288_);
return v_res_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Mathlib_Meta_NormNum_evalIntPow_spec__0(lean_object* v_a_292_){
_start:
{
lean_object* v___x_293_; 
v___x_293_ = lean_nat_to_int(v_a_292_);
return v___x_293_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__0(void){
_start:
{
lean_object* v___x_294_; lean_object* v___x_295_; 
v___x_294_ = lean_unsigned_to_nat(0u);
v___x_295_ = lean_nat_to_int(v___x_294_);
return v___x_295_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__3(void){
_start:
{
lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; 
v___x_300_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__3, &lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__3);
v___x_301_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__2));
v___x_302_ = l_Lean_Expr_const___override(v___x_301_, v___x_300_);
return v___x_302_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__4(void){
_start:
{
lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; 
v___x_303_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__7, &lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__7);
v___x_304_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__3, &lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__3_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__3);
v___x_305_ = l_Lean_Expr_app___override(v___x_304_, v___x_303_);
return v___x_305_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__8(void){
_start:
{
lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; 
v___x_311_ = lean_box(0);
v___x_312_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__7));
v___x_313_ = l_Lean_Expr_const___override(v___x_312_, v___x_311_);
return v___x_313_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__11(void){
_start:
{
lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v___x_322_; 
v___x_320_ = lean_box(0);
v___x_321_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__10));
v___x_322_ = l_Lean_Expr_const___override(v___x_321_, v___x_320_);
return v___x_322_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__14(void){
_start:
{
lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; 
v___x_327_ = lean_box(0);
v___x_328_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__13));
v___x_329_ = l_Lean_Expr_const___override(v___x_328_, v___x_327_);
return v___x_329_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__17(void){
_start:
{
lean_object* v___x_336_; lean_object* v___x_337_; lean_object* v___x_338_; 
v___x_336_ = lean_box(0);
v___x_337_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__16));
v___x_338_ = l_Lean_Expr_const___override(v___x_337_, v___x_336_);
return v___x_338_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__20(void){
_start:
{
lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; 
v___x_345_ = lean_box(0);
v___x_346_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__19));
v___x_347_ = l_Lean_Expr_const___override(v___x_346_, v___x_345_);
return v___x_347_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntPow(lean_object* v_za_348_, lean_object* v_a_349_, lean_object* v_b_350_, lean_object* v_a_351_, lean_object* v_a_352_){
_start:
{
lean_object* v_a_x27_354_; lean_object* v___x_355_; lean_object* v___x_356_; uint8_t v___x_357_; 
v_a_x27_354_ = l_Lean_Expr_appArg_x21(v_a_349_);
v___x_355_ = lean_unsigned_to_nat(0u);
v___x_356_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__0, &lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__0_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__0);
v___x_357_ = lean_int_dec_le(v___x_356_, v_za_348_);
if (v___x_357_ == 0)
{
lean_object* v_b_x27_358_; lean_object* v___x_359_; lean_object* v___x_360_; lean_object* v_b_u2080_361_; lean_object* v___x_362_; 
v_b_x27_358_ = lp_batteries_Lean_Expr_natLit_x21(v_b_350_);
v___x_359_ = lean_unsigned_to_nat(1u);
v___x_360_ = lean_nat_shiftr(v_b_x27_358_, v___x_359_);
v_b_u2080_361_ = l_Lean_mkRawNatLit(v___x_360_);
lean_inc_ref(v_b_u2080_361_);
lean_inc_ref(v_a_x27_354_);
v___x_362_ = lp_mathlib_Mathlib_Meta_NormNum_evalNatPow(v_a_x27_354_, v_b_u2080_361_, v_a_351_, v_a_352_);
if (lean_obj_tag(v___x_362_) == 0)
{
lean_object* v_a_363_; lean_object* v___x_365_; uint8_t v_isShared_366_; uint8_t v_isSharedCheck_444_; 
v_a_363_ = lean_ctor_get(v___x_362_, 0);
v_isSharedCheck_444_ = !lean_is_exclusive(v___x_362_);
if (v_isSharedCheck_444_ == 0)
{
v___x_365_ = v___x_362_;
v_isShared_366_ = v_isSharedCheck_444_;
goto v_resetjp_364_;
}
else
{
lean_inc(v_a_363_);
lean_dec(v___x_362_);
v___x_365_ = lean_box(0);
v_isShared_366_ = v_isSharedCheck_444_;
goto v_resetjp_364_;
}
v_resetjp_364_:
{
if (lean_obj_tag(v_a_363_) == 0)
{
lean_object* v___x_367_; lean_object* v___x_369_; 
lean_dec_ref(v_b_u2080_361_);
lean_dec(v_b_x27_358_);
lean_dec_ref(v_a_x27_354_);
lean_dec_ref(v_b_350_);
v___x_367_ = lean_box(0);
if (v_isShared_366_ == 0)
{
lean_ctor_set(v___x_365_, 0, v___x_367_);
v___x_369_ = v___x_365_;
goto v_reusejp_368_;
}
else
{
lean_object* v_reuseFailAlloc_370_; 
v_reuseFailAlloc_370_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_370_, 0, v___x_367_);
v___x_369_ = v_reuseFailAlloc_370_;
goto v_reusejp_368_;
}
v_reusejp_368_:
{
return v___x_369_;
}
}
else
{
lean_object* v_val_371_; lean_object* v___x_373_; uint8_t v_isShared_374_; uint8_t v_isSharedCheck_443_; 
v_val_371_ = lean_ctor_get(v_a_363_, 0);
v_isSharedCheck_443_ = !lean_is_exclusive(v_a_363_);
if (v_isSharedCheck_443_ == 0)
{
v___x_373_ = v_a_363_;
v_isShared_374_ = v_isSharedCheck_443_;
goto v_resetjp_372_;
}
else
{
lean_inc(v_val_371_);
lean_dec(v_a_363_);
v___x_373_ = lean_box(0);
v_isShared_374_ = v_isSharedCheck_443_;
goto v_resetjp_372_;
}
v_resetjp_372_:
{
lean_object* v_fst_375_; lean_object* v_snd_376_; lean_object* v___x_378_; uint8_t v_isShared_379_; uint8_t v_isSharedCheck_442_; 
v_fst_375_ = lean_ctor_get(v_val_371_, 0);
v_snd_376_ = lean_ctor_get(v_val_371_, 1);
v_isSharedCheck_442_ = !lean_is_exclusive(v_val_371_);
if (v_isSharedCheck_442_ == 0)
{
v___x_378_ = v_val_371_;
v_isShared_379_ = v_isSharedCheck_442_;
goto v_resetjp_377_;
}
else
{
lean_inc(v_snd_376_);
lean_inc(v_fst_375_);
lean_dec(v_val_371_);
v___x_378_ = lean_box(0);
v_isShared_379_ = v_isSharedCheck_442_;
goto v_resetjp_377_;
}
v_resetjp_377_:
{
lean_object* v___x_380_; lean_object* v___x_381_; uint8_t v___x_382_; 
v___x_380_ = lp_batteries_Lean_Expr_natLit_x21(v_fst_375_);
v___x_381_ = lean_nat_land(v_b_x27_358_, v___x_359_);
lean_dec(v_b_x27_358_);
v___x_382_ = lean_nat_dec_eq(v___x_381_, v___x_355_);
lean_dec(v___x_381_);
if (v___x_382_ == 0)
{
lean_object* v___x_383_; lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v___x_386_; lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_399_; lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_405_; 
v___x_383_ = lp_batteries_Lean_Expr_natLit_x21(v_a_x27_354_);
v___x_384_ = lean_nat_mul(v___x_380_, v___x_383_);
lean_dec(v___x_383_);
v___x_385_ = lean_nat_mul(v___x_380_, v___x_384_);
lean_dec(v___x_384_);
lean_dec(v___x_380_);
v___x_386_ = l_Lean_mkRawNatLit(v___x_385_);
v___x_387_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__4, &lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__4_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__4);
lean_inc_ref_n(v___x_386_, 2);
v___x_388_ = l_Lean_Expr_app___override(v___x_387_, v___x_386_);
lean_inc_ref(v_b_350_);
v___x_389_ = l_Lean_Expr_app___override(v___x_387_, v_b_350_);
v___x_390_ = lp_batteries_Lean_Expr_natLit_x21(v___x_386_);
v___x_391_ = lean_nat_to_int(v___x_390_);
v___x_392_ = lean_int_neg(v___x_391_);
lean_dec(v___x_391_);
v___x_393_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__8, &lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__8_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__8);
v___x_394_ = l_Lean_Expr_app___override(v___x_393_, v___x_386_);
v___x_395_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__11, &lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__11_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__11);
v___x_396_ = l_Lean_Expr_app___override(v___x_395_, v_a_x27_354_);
v___x_397_ = l_Lean_Expr_app___override(v___x_396_, v_b_350_);
v___x_398_ = l_Lean_Expr_app___override(v___x_397_, v___x_386_);
v___x_399_ = l_Lean_Expr_app___override(v___x_398_, v_b_u2080_361_);
v___x_400_ = l_Lean_Expr_app___override(v___x_399_, v_fst_375_);
v___x_401_ = l_Lean_Expr_app___override(v___x_400_, v_snd_376_);
v___x_402_ = l_Lean_Expr_app___override(v___x_401_, v___x_389_);
v___x_403_ = l_Lean_Expr_app___override(v___x_402_, v___x_388_);
if (v_isShared_379_ == 0)
{
lean_ctor_set(v___x_378_, 1, v___x_403_);
lean_ctor_set(v___x_378_, 0, v___x_394_);
v___x_405_ = v___x_378_;
goto v_reusejp_404_;
}
else
{
lean_object* v_reuseFailAlloc_413_; 
v_reuseFailAlloc_413_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_413_, 0, v___x_394_);
lean_ctor_set(v_reuseFailAlloc_413_, 1, v___x_403_);
v___x_405_ = v_reuseFailAlloc_413_;
goto v_reusejp_404_;
}
v_reusejp_404_:
{
lean_object* v___x_406_; lean_object* v___x_408_; 
v___x_406_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_406_, 0, v___x_392_);
lean_ctor_set(v___x_406_, 1, v___x_405_);
if (v_isShared_374_ == 0)
{
lean_ctor_set(v___x_373_, 0, v___x_406_);
v___x_408_ = v___x_373_;
goto v_reusejp_407_;
}
else
{
lean_object* v_reuseFailAlloc_412_; 
v_reuseFailAlloc_412_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_412_, 0, v___x_406_);
v___x_408_ = v_reuseFailAlloc_412_;
goto v_reusejp_407_;
}
v_reusejp_407_:
{
lean_object* v___x_410_; 
if (v_isShared_366_ == 0)
{
lean_ctor_set(v___x_365_, 0, v___x_408_);
v___x_410_ = v___x_365_;
goto v_reusejp_409_;
}
else
{
lean_object* v_reuseFailAlloc_411_; 
v_reuseFailAlloc_411_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_411_, 0, v___x_408_);
v___x_410_ = v_reuseFailAlloc_411_;
goto v_reusejp_409_;
}
v_reusejp_409_:
{
return v___x_410_;
}
}
}
}
else
{
lean_object* v___x_414_; lean_object* v___x_415_; lean_object* v___x_416_; lean_object* v___x_417_; lean_object* v___x_418_; lean_object* v___x_419_; lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v___x_427_; lean_object* v___x_428_; lean_object* v___x_429_; lean_object* v___x_430_; lean_object* v___x_431_; lean_object* v___x_433_; 
v___x_414_ = lean_nat_mul(v___x_380_, v___x_380_);
lean_dec(v___x_380_);
v___x_415_ = l_Lean_mkRawNatLit(v___x_414_);
v___x_416_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__4, &lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__4_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__4);
lean_inc_ref_n(v___x_415_, 2);
v___x_417_ = l_Lean_Expr_app___override(v___x_416_, v___x_415_);
lean_inc_ref(v_b_350_);
v___x_418_ = l_Lean_Expr_app___override(v___x_416_, v_b_350_);
v___x_419_ = lp_batteries_Lean_Expr_natLit_x21(v___x_415_);
v___x_420_ = lean_nat_to_int(v___x_419_);
v___x_421_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__14, &lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__14_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__14);
v___x_422_ = l_Lean_Expr_app___override(v___x_421_, v___x_415_);
v___x_423_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__17, &lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__17_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__17);
v___x_424_ = l_Lean_Expr_app___override(v___x_423_, v_a_x27_354_);
v___x_425_ = l_Lean_Expr_app___override(v___x_424_, v_b_350_);
v___x_426_ = l_Lean_Expr_app___override(v___x_425_, v___x_415_);
v___x_427_ = l_Lean_Expr_app___override(v___x_426_, v_b_u2080_361_);
v___x_428_ = l_Lean_Expr_app___override(v___x_427_, v_fst_375_);
v___x_429_ = l_Lean_Expr_app___override(v___x_428_, v_snd_376_);
v___x_430_ = l_Lean_Expr_app___override(v___x_429_, v___x_418_);
v___x_431_ = l_Lean_Expr_app___override(v___x_430_, v___x_417_);
if (v_isShared_379_ == 0)
{
lean_ctor_set(v___x_378_, 1, v___x_431_);
lean_ctor_set(v___x_378_, 0, v___x_422_);
v___x_433_ = v___x_378_;
goto v_reusejp_432_;
}
else
{
lean_object* v_reuseFailAlloc_441_; 
v_reuseFailAlloc_441_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_441_, 0, v___x_422_);
lean_ctor_set(v_reuseFailAlloc_441_, 1, v___x_431_);
v___x_433_ = v_reuseFailAlloc_441_;
goto v_reusejp_432_;
}
v_reusejp_432_:
{
lean_object* v___x_434_; lean_object* v___x_436_; 
v___x_434_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_434_, 0, v___x_420_);
lean_ctor_set(v___x_434_, 1, v___x_433_);
if (v_isShared_374_ == 0)
{
lean_ctor_set(v___x_373_, 0, v___x_434_);
v___x_436_ = v___x_373_;
goto v_reusejp_435_;
}
else
{
lean_object* v_reuseFailAlloc_440_; 
v_reuseFailAlloc_440_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_440_, 0, v___x_434_);
v___x_436_ = v_reuseFailAlloc_440_;
goto v_reusejp_435_;
}
v_reusejp_435_:
{
lean_object* v___x_438_; 
if (v_isShared_366_ == 0)
{
lean_ctor_set(v___x_365_, 0, v___x_436_);
v___x_438_ = v___x_365_;
goto v_reusejp_437_;
}
else
{
lean_object* v_reuseFailAlloc_439_; 
v_reuseFailAlloc_439_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_439_, 0, v___x_436_);
v___x_438_ = v_reuseFailAlloc_439_;
goto v_reusejp_437_;
}
v_reusejp_437_:
{
return v___x_438_;
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
lean_object* v_a_445_; lean_object* v___x_447_; uint8_t v_isShared_448_; uint8_t v_isSharedCheck_452_; 
lean_dec_ref(v_b_u2080_361_);
lean_dec(v_b_x27_358_);
lean_dec_ref(v_a_x27_354_);
lean_dec_ref(v_b_350_);
v_a_445_ = lean_ctor_get(v___x_362_, 0);
v_isSharedCheck_452_ = !lean_is_exclusive(v___x_362_);
if (v_isSharedCheck_452_ == 0)
{
v___x_447_ = v___x_362_;
v_isShared_448_ = v_isSharedCheck_452_;
goto v_resetjp_446_;
}
else
{
lean_inc(v_a_445_);
lean_dec(v___x_362_);
v___x_447_ = lean_box(0);
v_isShared_448_ = v_isSharedCheck_452_;
goto v_resetjp_446_;
}
v_resetjp_446_:
{
lean_object* v___x_450_; 
if (v_isShared_448_ == 0)
{
v___x_450_ = v___x_447_;
goto v_reusejp_449_;
}
else
{
lean_object* v_reuseFailAlloc_451_; 
v_reuseFailAlloc_451_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_451_, 0, v_a_445_);
v___x_450_ = v_reuseFailAlloc_451_;
goto v_reusejp_449_;
}
v_reusejp_449_:
{
return v___x_450_;
}
}
}
}
else
{
lean_object* v___x_453_; 
lean_inc_ref(v_b_350_);
lean_inc_ref(v_a_x27_354_);
v___x_453_ = lp_mathlib_Mathlib_Meta_NormNum_evalNatPow(v_a_x27_354_, v_b_350_, v_a_351_, v_a_352_);
if (lean_obj_tag(v___x_453_) == 0)
{
lean_object* v_a_454_; lean_object* v___x_456_; uint8_t v_isShared_457_; uint8_t v_isSharedCheck_492_; 
v_a_454_ = lean_ctor_get(v___x_453_, 0);
v_isSharedCheck_492_ = !lean_is_exclusive(v___x_453_);
if (v_isSharedCheck_492_ == 0)
{
v___x_456_ = v___x_453_;
v_isShared_457_ = v_isSharedCheck_492_;
goto v_resetjp_455_;
}
else
{
lean_inc(v_a_454_);
lean_dec(v___x_453_);
v___x_456_ = lean_box(0);
v_isShared_457_ = v_isSharedCheck_492_;
goto v_resetjp_455_;
}
v_resetjp_455_:
{
if (lean_obj_tag(v_a_454_) == 0)
{
lean_object* v___x_458_; lean_object* v___x_460_; 
lean_dec_ref(v_a_x27_354_);
lean_dec_ref(v_b_350_);
v___x_458_ = lean_box(0);
if (v_isShared_457_ == 0)
{
lean_ctor_set(v___x_456_, 0, v___x_458_);
v___x_460_ = v___x_456_;
goto v_reusejp_459_;
}
else
{
lean_object* v_reuseFailAlloc_461_; 
v_reuseFailAlloc_461_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_461_, 0, v___x_458_);
v___x_460_ = v_reuseFailAlloc_461_;
goto v_reusejp_459_;
}
v_reusejp_459_:
{
return v___x_460_;
}
}
else
{
lean_object* v_val_462_; lean_object* v___x_464_; uint8_t v_isShared_465_; uint8_t v_isSharedCheck_491_; 
v_val_462_ = lean_ctor_get(v_a_454_, 0);
v_isSharedCheck_491_ = !lean_is_exclusive(v_a_454_);
if (v_isSharedCheck_491_ == 0)
{
v___x_464_ = v_a_454_;
v_isShared_465_ = v_isSharedCheck_491_;
goto v_resetjp_463_;
}
else
{
lean_inc(v_val_462_);
lean_dec(v_a_454_);
v___x_464_ = lean_box(0);
v_isShared_465_ = v_isSharedCheck_491_;
goto v_resetjp_463_;
}
v_resetjp_463_:
{
lean_object* v_fst_466_; lean_object* v_snd_467_; lean_object* v___x_469_; uint8_t v_isShared_470_; uint8_t v_isSharedCheck_490_; 
v_fst_466_ = lean_ctor_get(v_val_462_, 0);
v_snd_467_ = lean_ctor_get(v_val_462_, 1);
v_isSharedCheck_490_ = !lean_is_exclusive(v_val_462_);
if (v_isSharedCheck_490_ == 0)
{
v___x_469_ = v_val_462_;
v_isShared_470_ = v_isSharedCheck_490_;
goto v_resetjp_468_;
}
else
{
lean_inc(v_snd_467_);
lean_inc(v_fst_466_);
lean_dec(v_val_462_);
v___x_469_ = lean_box(0);
v_isShared_470_ = v_isSharedCheck_490_;
goto v_resetjp_468_;
}
v_resetjp_468_:
{
lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; lean_object* v___x_475_; lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v___x_481_; 
v___x_471_ = lp_batteries_Lean_Expr_natLit_x21(v_fst_466_);
v___x_472_ = lean_nat_to_int(v___x_471_);
v___x_473_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__14, &lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__14_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__14);
lean_inc(v_fst_466_);
v___x_474_ = l_Lean_Expr_app___override(v___x_473_, v_fst_466_);
v___x_475_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__20, &lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__20_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__20);
v___x_476_ = l_Lean_Expr_app___override(v___x_475_, v_a_x27_354_);
v___x_477_ = l_Lean_Expr_app___override(v___x_476_, v_b_350_);
v___x_478_ = l_Lean_Expr_app___override(v___x_477_, v_fst_466_);
v___x_479_ = l_Lean_Expr_app___override(v___x_478_, v_snd_467_);
if (v_isShared_470_ == 0)
{
lean_ctor_set(v___x_469_, 1, v___x_479_);
lean_ctor_set(v___x_469_, 0, v___x_474_);
v___x_481_ = v___x_469_;
goto v_reusejp_480_;
}
else
{
lean_object* v_reuseFailAlloc_489_; 
v_reuseFailAlloc_489_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_489_, 0, v___x_474_);
lean_ctor_set(v_reuseFailAlloc_489_, 1, v___x_479_);
v___x_481_ = v_reuseFailAlloc_489_;
goto v_reusejp_480_;
}
v_reusejp_480_:
{
lean_object* v___x_482_; lean_object* v___x_484_; 
v___x_482_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_482_, 0, v___x_472_);
lean_ctor_set(v___x_482_, 1, v___x_481_);
if (v_isShared_465_ == 0)
{
lean_ctor_set(v___x_464_, 0, v___x_482_);
v___x_484_ = v___x_464_;
goto v_reusejp_483_;
}
else
{
lean_object* v_reuseFailAlloc_488_; 
v_reuseFailAlloc_488_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_488_, 0, v___x_482_);
v___x_484_ = v_reuseFailAlloc_488_;
goto v_reusejp_483_;
}
v_reusejp_483_:
{
lean_object* v___x_486_; 
if (v_isShared_457_ == 0)
{
lean_ctor_set(v___x_456_, 0, v___x_484_);
v___x_486_ = v___x_456_;
goto v_reusejp_485_;
}
else
{
lean_object* v_reuseFailAlloc_487_; 
v_reuseFailAlloc_487_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_487_, 0, v___x_484_);
v___x_486_ = v_reuseFailAlloc_487_;
goto v_reusejp_485_;
}
v_reusejp_485_:
{
return v___x_486_;
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
lean_object* v_a_493_; lean_object* v___x_495_; uint8_t v_isShared_496_; uint8_t v_isSharedCheck_500_; 
lean_dec_ref(v_a_x27_354_);
lean_dec_ref(v_b_350_);
v_a_493_ = lean_ctor_get(v___x_453_, 0);
v_isSharedCheck_500_ = !lean_is_exclusive(v___x_453_);
if (v_isSharedCheck_500_ == 0)
{
v___x_495_ = v___x_453_;
v_isShared_496_ = v_isSharedCheck_500_;
goto v_resetjp_494_;
}
else
{
lean_inc(v_a_493_);
lean_dec(v___x_453_);
v___x_495_ = lean_box(0);
v_isShared_496_ = v_isSharedCheck_500_;
goto v_resetjp_494_;
}
v_resetjp_494_:
{
lean_object* v___x_498_; 
if (v_isShared_496_ == 0)
{
v___x_498_ = v___x_495_;
goto v_reusejp_497_;
}
else
{
lean_object* v_reuseFailAlloc_499_; 
v_reuseFailAlloc_499_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_499_, 0, v_a_493_);
v___x_498_ = v_reuseFailAlloc_499_;
goto v_reusejp_497_;
}
v_reusejp_497_:
{
return v___x_498_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___boxed(lean_object* v_za_501_, lean_object* v_a_502_, lean_object* v_b_503_, lean_object* v_a_504_, lean_object* v_a_505_, lean_object* v_a_506_){
_start:
{
lean_object* v_res_507_; 
v_res_507_ = lp_mathlib_Mathlib_Meta_NormNum_evalIntPow(v_za_501_, v_a_502_, v_b_503_, v_a_504_, v_a_505_);
lean_dec(v_a_505_);
lean_dec_ref(v_a_504_);
lean_dec_ref(v_a_502_);
lean_dec(v_za_501_);
return v_res_507_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow_core(lean_object* v_u_537_, lean_object* v_00_u03b1_538_, lean_object* v_e_539_, lean_object* v_f_540_, lean_object* v_a_541_, lean_object* v_b_542_, lean_object* v_nb_543_, lean_object* v_pb_544_, lean_object* v_s_u03b1_545_, lean_object* v_ra_546_, lean_object* v_a_547_, lean_object* v_a_548_){
_start:
{
switch(lean_obj_tag(v_ra_546_))
{
case 0:
{
lean_object* v___x_550_; lean_object* v___x_551_; 
lean_dec_ref_known(v_ra_546_, 1);
lean_dec_ref(v_s_u03b1_545_);
lean_dec_ref(v_pb_544_);
lean_dec_ref(v_nb_543_);
lean_dec_ref(v_b_542_);
lean_dec_ref(v_a_541_);
lean_dec_ref(v_f_540_);
lean_dec_ref(v_e_539_);
lean_dec_ref(v_00_u03b1_538_);
lean_dec(v_u_537_);
v___x_550_ = lean_box(0);
v___x_551_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_551_, 0, v___x_550_);
return v___x_551_;
}
case 1:
{
lean_object* v_inst_552_; lean_object* v_lit_553_; lean_object* v_proof_554_; lean_object* v___x_556_; uint8_t v_isShared_557_; uint8_t v_isSharedCheck_625_; 
lean_dec_ref(v_e_539_);
v_inst_552_ = lean_ctor_get(v_ra_546_, 0);
v_lit_553_ = lean_ctor_get(v_ra_546_, 1);
v_proof_554_ = lean_ctor_get(v_ra_546_, 2);
v_isSharedCheck_625_ = !lean_is_exclusive(v_ra_546_);
if (v_isSharedCheck_625_ == 0)
{
v___x_556_ = v_ra_546_;
v_isShared_557_ = v_isSharedCheck_625_;
goto v_resetjp_555_;
}
else
{
lean_inc(v_proof_554_);
lean_inc(v_lit_553_);
lean_inc(v_inst_552_);
lean_dec(v_ra_546_);
v___x_556_ = lean_box(0);
v_isShared_557_ = v_isSharedCheck_625_;
goto v_resetjp_555_;
}
v_resetjp_555_:
{
lean_object* v___x_558_; 
lean_inc_ref(v_nb_543_);
lean_inc_ref(v_lit_553_);
v___x_558_ = lp_mathlib_Mathlib_Meta_NormNum_evalNatPow(v_lit_553_, v_nb_543_, v_a_547_, v_a_548_);
if (lean_obj_tag(v___x_558_) == 0)
{
lean_object* v_a_559_; lean_object* v___x_561_; uint8_t v_isShared_562_; uint8_t v_isSharedCheck_616_; 
v_a_559_ = lean_ctor_get(v___x_558_, 0);
v_isSharedCheck_616_ = !lean_is_exclusive(v___x_558_);
if (v_isSharedCheck_616_ == 0)
{
v___x_561_ = v___x_558_;
v_isShared_562_ = v_isSharedCheck_616_;
goto v_resetjp_560_;
}
else
{
lean_inc(v_a_559_);
lean_dec(v___x_558_);
v___x_561_ = lean_box(0);
v_isShared_562_ = v_isSharedCheck_616_;
goto v_resetjp_560_;
}
v_resetjp_560_:
{
if (lean_obj_tag(v_a_559_) == 0)
{
lean_object* v___x_563_; lean_object* v___x_565_; 
lean_del_object(v___x_556_);
lean_dec_ref(v_proof_554_);
lean_dec_ref(v_lit_553_);
lean_dec_ref(v_inst_552_);
lean_dec_ref(v_s_u03b1_545_);
lean_dec_ref(v_pb_544_);
lean_dec_ref(v_nb_543_);
lean_dec_ref(v_b_542_);
lean_dec_ref(v_a_541_);
lean_dec_ref(v_f_540_);
lean_dec_ref(v_00_u03b1_538_);
lean_dec(v_u_537_);
v___x_563_ = lean_box(0);
if (v_isShared_562_ == 0)
{
lean_ctor_set(v___x_561_, 0, v___x_563_);
v___x_565_ = v___x_561_;
goto v_reusejp_564_;
}
else
{
lean_object* v_reuseFailAlloc_566_; 
v_reuseFailAlloc_566_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_566_, 0, v___x_563_);
v___x_565_ = v_reuseFailAlloc_566_;
goto v_reusejp_564_;
}
v_reusejp_564_:
{
return v___x_565_;
}
}
else
{
lean_object* v_val_567_; lean_object* v___x_569_; uint8_t v_isShared_570_; uint8_t v_isSharedCheck_615_; 
v_val_567_ = lean_ctor_get(v_a_559_, 0);
v_isSharedCheck_615_ = !lean_is_exclusive(v_a_559_);
if (v_isSharedCheck_615_ == 0)
{
v___x_569_ = v_a_559_;
v_isShared_570_ = v_isSharedCheck_615_;
goto v_resetjp_568_;
}
else
{
lean_inc(v_val_567_);
lean_dec(v_a_559_);
v___x_569_ = lean_box(0);
v_isShared_570_ = v_isSharedCheck_615_;
goto v_resetjp_568_;
}
v_resetjp_568_:
{
lean_object* v_fst_571_; lean_object* v_snd_572_; lean_object* v___x_574_; uint8_t v_isShared_575_; uint8_t v_isSharedCheck_614_; 
v_fst_571_ = lean_ctor_get(v_val_567_, 0);
v_snd_572_ = lean_ctor_get(v_val_567_, 1);
v_isSharedCheck_614_ = !lean_is_exclusive(v_val_567_);
if (v_isSharedCheck_614_ == 0)
{
v___x_574_ = v_val_567_;
v_isShared_575_ = v_isSharedCheck_614_;
goto v_resetjp_573_;
}
else
{
lean_inc(v_snd_572_);
lean_inc(v_fst_571_);
lean_dec(v_val_567_);
v___x_574_ = lean_box(0);
v_isShared_575_ = v_isSharedCheck_614_;
goto v_resetjp_573_;
}
v_resetjp_573_:
{
lean_object* v___x_576_; lean_object* v___x_578_; 
v___x_576_ = lean_box(0);
lean_inc(v_u_537_);
if (v_isShared_575_ == 0)
{
lean_ctor_set_tag(v___x_574_, 1);
lean_ctor_set(v___x_574_, 1, v___x_576_);
lean_ctor_set(v___x_574_, 0, v_u_537_);
v___x_578_ = v___x_574_;
goto v_reusejp_577_;
}
else
{
lean_object* v_reuseFailAlloc_613_; 
v_reuseFailAlloc_613_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_613_, 0, v_u_537_);
lean_ctor_set(v_reuseFailAlloc_613_, 1, v___x_576_);
v___x_578_ = v_reuseFailAlloc_613_;
goto v_reusejp_577_;
}
v_reusejp_577_:
{
lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; lean_object* v___x_585_; lean_object* v___x_586_; lean_object* v___x_587_; lean_object* v___x_588_; lean_object* v___x_589_; lean_object* v___x_590_; lean_object* v___x_591_; lean_object* v___x_592_; lean_object* v___x_593_; lean_object* v___x_594_; uint8_t v___x_595_; lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v___x_598_; lean_object* v___x_599_; lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v___x_605_; 
v___x_579_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__1));
v___x_580_ = l_Lean_Expr_const___override(v___x_579_, v___x_578_);
lean_inc_ref_n(v_00_u03b1_538_, 2);
v___x_581_ = l_Lean_Expr_app___override(v___x_580_, v_00_u03b1_538_);
v___x_582_ = l_Lean_Expr_app___override(v___x_581_, v_s_u03b1_545_);
lean_inc_ref(v_f_540_);
v___x_583_ = l_Lean_Expr_app___override(v___x_582_, v_f_540_);
v___x_584_ = l_Lean_Expr_app___override(v___x_583_, v_a_541_);
v___x_585_ = l_Lean_Expr_app___override(v___x_584_, v_b_542_);
v___x_586_ = l_Lean_Expr_app___override(v___x_585_, v_lit_553_);
v___x_587_ = l_Lean_Expr_app___override(v___x_586_, v_nb_543_);
lean_inc(v_fst_571_);
v___x_588_ = l_Lean_Expr_app___override(v___x_587_, v_fst_571_);
v___x_589_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__2));
v___x_590_ = l_Lean_Level_succ___override(v_u_537_);
v___x_591_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_591_, 0, v___x_590_);
lean_ctor_set(v___x_591_, 1, v___x_576_);
v___x_592_ = l_Lean_Expr_const___override(v___x_589_, v___x_591_);
v___x_593_ = lean_box(0);
v___x_594_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__7, &lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__7);
v___x_595_ = 0;
v___x_596_ = l_Lean_Expr_forallE___override(v___x_593_, v___x_594_, v_00_u03b1_538_, v___x_595_);
v___x_597_ = l_Lean_Expr_forallE___override(v___x_593_, v_00_u03b1_538_, v___x_596_, v___x_595_);
v___x_598_ = l_Lean_Expr_app___override(v___x_592_, v___x_597_);
v___x_599_ = l_Lean_Expr_app___override(v___x_598_, v_f_540_);
v___x_600_ = l_Lean_Expr_app___override(v___x_588_, v___x_599_);
v___x_601_ = l_Lean_Expr_app___override(v___x_600_, v_proof_554_);
v___x_602_ = l_Lean_Expr_app___override(v___x_601_, v_pb_544_);
v___x_603_ = l_Lean_Expr_app___override(v___x_602_, v_snd_572_);
if (v_isShared_557_ == 0)
{
lean_ctor_set(v___x_556_, 2, v___x_603_);
lean_ctor_set(v___x_556_, 1, v_fst_571_);
v___x_605_ = v___x_556_;
goto v_reusejp_604_;
}
else
{
lean_object* v_reuseFailAlloc_612_; 
v_reuseFailAlloc_612_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_612_, 0, v_inst_552_);
lean_ctor_set(v_reuseFailAlloc_612_, 1, v_fst_571_);
lean_ctor_set(v_reuseFailAlloc_612_, 2, v___x_603_);
v___x_605_ = v_reuseFailAlloc_612_;
goto v_reusejp_604_;
}
v_reusejp_604_:
{
lean_object* v___x_607_; 
if (v_isShared_570_ == 0)
{
lean_ctor_set(v___x_569_, 0, v___x_605_);
v___x_607_ = v___x_569_;
goto v_reusejp_606_;
}
else
{
lean_object* v_reuseFailAlloc_611_; 
v_reuseFailAlloc_611_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_611_, 0, v___x_605_);
v___x_607_ = v_reuseFailAlloc_611_;
goto v_reusejp_606_;
}
v_reusejp_606_:
{
lean_object* v___x_609_; 
if (v_isShared_562_ == 0)
{
lean_ctor_set(v___x_561_, 0, v___x_607_);
v___x_609_ = v___x_561_;
goto v_reusejp_608_;
}
else
{
lean_object* v_reuseFailAlloc_610_; 
v_reuseFailAlloc_610_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_610_, 0, v___x_607_);
v___x_609_ = v_reuseFailAlloc_610_;
goto v_reusejp_608_;
}
v_reusejp_608_:
{
return v___x_609_;
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
lean_object* v_a_617_; lean_object* v___x_619_; uint8_t v_isShared_620_; uint8_t v_isSharedCheck_624_; 
lean_del_object(v___x_556_);
lean_dec_ref(v_proof_554_);
lean_dec_ref(v_lit_553_);
lean_dec_ref(v_inst_552_);
lean_dec_ref(v_s_u03b1_545_);
lean_dec_ref(v_pb_544_);
lean_dec_ref(v_nb_543_);
lean_dec_ref(v_b_542_);
lean_dec_ref(v_a_541_);
lean_dec_ref(v_f_540_);
lean_dec_ref(v_00_u03b1_538_);
lean_dec(v_u_537_);
v_a_617_ = lean_ctor_get(v___x_558_, 0);
v_isSharedCheck_624_ = !lean_is_exclusive(v___x_558_);
if (v_isSharedCheck_624_ == 0)
{
v___x_619_ = v___x_558_;
v_isShared_620_ = v_isSharedCheck_624_;
goto v_resetjp_618_;
}
else
{
lean_inc(v_a_617_);
lean_dec(v___x_558_);
v___x_619_ = lean_box(0);
v_isShared_620_ = v_isSharedCheck_624_;
goto v_resetjp_618_;
}
v_resetjp_618_:
{
lean_object* v___x_622_; 
if (v_isShared_620_ == 0)
{
v___x_622_ = v___x_619_;
goto v_reusejp_621_;
}
else
{
lean_object* v_reuseFailAlloc_623_; 
v_reuseFailAlloc_623_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_623_, 0, v_a_617_);
v___x_622_ = v_reuseFailAlloc_623_;
goto v_reusejp_621_;
}
v_reusejp_621_:
{
return v___x_622_;
}
}
}
}
}
case 2:
{
lean_object* v_inst_626_; lean_object* v___x_627_; 
lean_dec_ref(v_s_u03b1_545_);
v_inst_626_ = lean_ctor_get(v_ra_546_, 0);
lean_inc_ref_n(v_inst_626_, 2);
lean_inc_ref(v_a_541_);
lean_inc_ref(v_00_u03b1_538_);
lean_inc(v_u_537_);
v___x_627_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toInt(v_u_537_, v_00_u03b1_538_, v_a_541_, v_inst_626_, v_ra_546_);
if (lean_obj_tag(v___x_627_) == 0)
{
lean_object* v___x_628_; lean_object* v___x_629_; 
lean_dec_ref(v_inst_626_);
lean_dec_ref(v_pb_544_);
lean_dec_ref(v_nb_543_);
lean_dec_ref(v_b_542_);
lean_dec_ref(v_a_541_);
lean_dec_ref(v_f_540_);
lean_dec_ref(v_e_539_);
lean_dec_ref(v_00_u03b1_538_);
lean_dec(v_u_537_);
v___x_628_ = lean_box(0);
v___x_629_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_629_, 0, v___x_628_);
return v___x_629_;
}
else
{
lean_object* v_val_630_; lean_object* v_snd_631_; lean_object* v_fst_632_; lean_object* v_fst_633_; lean_object* v_snd_634_; lean_object* v___x_635_; 
v_val_630_ = lean_ctor_get(v___x_627_, 0);
lean_inc(v_val_630_);
lean_dec_ref_known(v___x_627_, 1);
v_snd_631_ = lean_ctor_get(v_val_630_, 1);
lean_inc(v_snd_631_);
v_fst_632_ = lean_ctor_get(v_val_630_, 0);
lean_inc(v_fst_632_);
lean_dec(v_val_630_);
v_fst_633_ = lean_ctor_get(v_snd_631_, 0);
lean_inc(v_fst_633_);
v_snd_634_ = lean_ctor_get(v_snd_631_, 1);
lean_inc(v_snd_634_);
lean_dec(v_snd_631_);
lean_inc_ref(v_nb_543_);
v___x_635_ = lp_mathlib_Mathlib_Meta_NormNum_evalIntPow(v_fst_632_, v_fst_633_, v_nb_543_, v_a_547_, v_a_548_);
lean_dec(v_fst_632_);
if (lean_obj_tag(v___x_635_) == 0)
{
lean_object* v_a_636_; lean_object* v___x_638_; uint8_t v_isShared_639_; uint8_t v_isSharedCheck_699_; 
v_a_636_ = lean_ctor_get(v___x_635_, 0);
v_isSharedCheck_699_ = !lean_is_exclusive(v___x_635_);
if (v_isSharedCheck_699_ == 0)
{
v___x_638_ = v___x_635_;
v_isShared_639_ = v_isSharedCheck_699_;
goto v_resetjp_637_;
}
else
{
lean_inc(v_a_636_);
lean_dec(v___x_635_);
v___x_638_ = lean_box(0);
v_isShared_639_ = v_isSharedCheck_699_;
goto v_resetjp_637_;
}
v_resetjp_637_:
{
if (lean_obj_tag(v_a_636_) == 0)
{
lean_object* v___x_640_; lean_object* v___x_642_; 
lean_dec(v_snd_634_);
lean_dec(v_fst_633_);
lean_dec_ref(v_inst_626_);
lean_dec_ref(v_pb_544_);
lean_dec_ref(v_nb_543_);
lean_dec_ref(v_b_542_);
lean_dec_ref(v_a_541_);
lean_dec_ref(v_f_540_);
lean_dec_ref(v_e_539_);
lean_dec_ref(v_00_u03b1_538_);
lean_dec(v_u_537_);
v___x_640_ = lean_box(0);
if (v_isShared_639_ == 0)
{
lean_ctor_set(v___x_638_, 0, v___x_640_);
v___x_642_ = v___x_638_;
goto v_reusejp_641_;
}
else
{
lean_object* v_reuseFailAlloc_643_; 
v_reuseFailAlloc_643_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_643_, 0, v___x_640_);
v___x_642_ = v_reuseFailAlloc_643_;
goto v_reusejp_641_;
}
v_reusejp_641_:
{
return v___x_642_;
}
}
else
{
lean_object* v_val_644_; lean_object* v___x_646_; uint8_t v_isShared_647_; uint8_t v_isSharedCheck_698_; 
v_val_644_ = lean_ctor_get(v_a_636_, 0);
v_isSharedCheck_698_ = !lean_is_exclusive(v_a_636_);
if (v_isSharedCheck_698_ == 0)
{
v___x_646_ = v_a_636_;
v_isShared_647_ = v_isSharedCheck_698_;
goto v_resetjp_645_;
}
else
{
lean_inc(v_val_644_);
lean_dec(v_a_636_);
v___x_646_ = lean_box(0);
v_isShared_647_ = v_isSharedCheck_698_;
goto v_resetjp_645_;
}
v_resetjp_645_:
{
lean_object* v_snd_648_; lean_object* v_fst_649_; lean_object* v___x_651_; uint8_t v_isShared_652_; uint8_t v_isSharedCheck_697_; 
v_snd_648_ = lean_ctor_get(v_val_644_, 1);
v_fst_649_ = lean_ctor_get(v_val_644_, 0);
v_isSharedCheck_697_ = !lean_is_exclusive(v_val_644_);
if (v_isSharedCheck_697_ == 0)
{
v___x_651_ = v_val_644_;
v_isShared_652_ = v_isSharedCheck_697_;
goto v_resetjp_650_;
}
else
{
lean_inc(v_snd_648_);
lean_inc(v_fst_649_);
lean_dec(v_val_644_);
v___x_651_ = lean_box(0);
v_isShared_652_ = v_isSharedCheck_697_;
goto v_resetjp_650_;
}
v_resetjp_650_:
{
lean_object* v_fst_653_; lean_object* v_snd_654_; lean_object* v___x_656_; uint8_t v_isShared_657_; uint8_t v_isSharedCheck_696_; 
v_fst_653_ = lean_ctor_get(v_snd_648_, 0);
v_snd_654_ = lean_ctor_get(v_snd_648_, 1);
v_isSharedCheck_696_ = !lean_is_exclusive(v_snd_648_);
if (v_isSharedCheck_696_ == 0)
{
v___x_656_ = v_snd_648_;
v_isShared_657_ = v_isSharedCheck_696_;
goto v_resetjp_655_;
}
else
{
lean_inc(v_snd_654_);
lean_inc(v_fst_653_);
lean_dec(v_snd_648_);
v___x_656_ = lean_box(0);
v_isShared_657_ = v_isSharedCheck_696_;
goto v_resetjp_655_;
}
v_resetjp_655_:
{
lean_object* v___x_658_; lean_object* v___x_660_; 
v___x_658_ = lean_box(0);
lean_inc(v_u_537_);
if (v_isShared_657_ == 0)
{
lean_ctor_set_tag(v___x_656_, 1);
lean_ctor_set(v___x_656_, 1, v___x_658_);
lean_ctor_set(v___x_656_, 0, v_u_537_);
v___x_660_ = v___x_656_;
goto v_reusejp_659_;
}
else
{
lean_object* v_reuseFailAlloc_695_; 
v_reuseFailAlloc_695_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_695_, 0, v_u_537_);
lean_ctor_set(v_reuseFailAlloc_695_, 1, v___x_658_);
v___x_660_ = v_reuseFailAlloc_695_;
goto v_reusejp_659_;
}
v_reusejp_659_:
{
lean_object* v___x_661_; lean_object* v___x_662_; lean_object* v___x_663_; lean_object* v___x_664_; lean_object* v___x_665_; lean_object* v___x_666_; lean_object* v___x_667_; lean_object* v___x_668_; lean_object* v___x_669_; lean_object* v___x_670_; lean_object* v___x_671_; lean_object* v___x_672_; lean_object* v___x_674_; 
v___x_661_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__3));
v___x_662_ = l_Lean_Expr_const___override(v___x_661_, v___x_660_);
lean_inc_ref(v_00_u03b1_538_);
v___x_663_ = l_Lean_Expr_app___override(v___x_662_, v_00_u03b1_538_);
lean_inc_ref(v_inst_626_);
v___x_664_ = l_Lean_Expr_app___override(v___x_663_, v_inst_626_);
lean_inc_ref(v_f_540_);
v___x_665_ = l_Lean_Expr_app___override(v___x_664_, v_f_540_);
v___x_666_ = l_Lean_Expr_app___override(v___x_665_, v_a_541_);
v___x_667_ = l_Lean_Expr_app___override(v___x_666_, v_b_542_);
v___x_668_ = l_Lean_Expr_app___override(v___x_667_, v_fst_633_);
v___x_669_ = l_Lean_Expr_app___override(v___x_668_, v_nb_543_);
lean_inc(v_fst_653_);
v___x_670_ = l_Lean_Expr_app___override(v___x_669_, v_fst_653_);
v___x_671_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__2));
lean_inc(v_u_537_);
v___x_672_ = l_Lean_Level_succ___override(v_u_537_);
if (v_isShared_652_ == 0)
{
lean_ctor_set_tag(v___x_651_, 1);
lean_ctor_set(v___x_651_, 1, v___x_658_);
lean_ctor_set(v___x_651_, 0, v___x_672_);
v___x_674_ = v___x_651_;
goto v_reusejp_673_;
}
else
{
lean_object* v_reuseFailAlloc_694_; 
v_reuseFailAlloc_694_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_694_, 0, v___x_672_);
lean_ctor_set(v_reuseFailAlloc_694_, 1, v___x_658_);
v___x_674_ = v_reuseFailAlloc_694_;
goto v_reusejp_673_;
}
v_reusejp_673_:
{
lean_object* v___x_675_; lean_object* v___x_676_; lean_object* v___x_677_; uint8_t v___x_678_; lean_object* v___x_679_; lean_object* v___x_680_; lean_object* v___x_681_; lean_object* v___x_682_; lean_object* v___x_683_; lean_object* v___x_684_; lean_object* v___x_685_; lean_object* v___x_686_; lean_object* v___x_687_; lean_object* v___x_689_; 
v___x_675_ = l_Lean_Expr_const___override(v___x_671_, v___x_674_);
v___x_676_ = lean_box(0);
v___x_677_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__7, &lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__7);
v___x_678_ = 0;
lean_inc_ref_n(v_00_u03b1_538_, 2);
v___x_679_ = l_Lean_Expr_forallE___override(v___x_676_, v___x_677_, v_00_u03b1_538_, v___x_678_);
v___x_680_ = l_Lean_Expr_forallE___override(v___x_676_, v_00_u03b1_538_, v___x_679_, v___x_678_);
v___x_681_ = l_Lean_Expr_app___override(v___x_675_, v___x_680_);
v___x_682_ = l_Lean_Expr_app___override(v___x_681_, v_f_540_);
v___x_683_ = l_Lean_Expr_app___override(v___x_670_, v___x_682_);
v___x_684_ = l_Lean_Expr_app___override(v___x_683_, v_snd_634_);
v___x_685_ = l_Lean_Expr_app___override(v___x_684_, v_pb_544_);
v___x_686_ = l_Lean_Expr_app___override(v___x_685_, v_snd_654_);
v___x_687_ = lp_mathlib_Mathlib_Meta_NormNum_Result_isInt(v_u_537_, v_00_u03b1_538_, v_e_539_, v_inst_626_, v_fst_653_, v_fst_649_, v___x_686_);
lean_dec(v_fst_649_);
lean_dec(v_fst_653_);
if (v_isShared_647_ == 0)
{
lean_ctor_set(v___x_646_, 0, v___x_687_);
v___x_689_ = v___x_646_;
goto v_reusejp_688_;
}
else
{
lean_object* v_reuseFailAlloc_693_; 
v_reuseFailAlloc_693_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_693_, 0, v___x_687_);
v___x_689_ = v_reuseFailAlloc_693_;
goto v_reusejp_688_;
}
v_reusejp_688_:
{
lean_object* v___x_691_; 
if (v_isShared_639_ == 0)
{
lean_ctor_set(v___x_638_, 0, v___x_689_);
v___x_691_ = v___x_638_;
goto v_reusejp_690_;
}
else
{
lean_object* v_reuseFailAlloc_692_; 
v_reuseFailAlloc_692_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_692_, 0, v___x_689_);
v___x_691_ = v_reuseFailAlloc_692_;
goto v_reusejp_690_;
}
v_reusejp_690_:
{
return v___x_691_;
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
lean_object* v_a_700_; lean_object* v___x_702_; uint8_t v_isShared_703_; uint8_t v_isSharedCheck_707_; 
lean_dec(v_snd_634_);
lean_dec(v_fst_633_);
lean_dec_ref(v_inst_626_);
lean_dec_ref(v_pb_544_);
lean_dec_ref(v_nb_543_);
lean_dec_ref(v_b_542_);
lean_dec_ref(v_a_541_);
lean_dec_ref(v_f_540_);
lean_dec_ref(v_e_539_);
lean_dec_ref(v_00_u03b1_538_);
lean_dec(v_u_537_);
v_a_700_ = lean_ctor_get(v___x_635_, 0);
v_isSharedCheck_707_ = !lean_is_exclusive(v___x_635_);
if (v_isSharedCheck_707_ == 0)
{
v___x_702_ = v___x_635_;
v_isShared_703_ = v_isSharedCheck_707_;
goto v_resetjp_701_;
}
else
{
lean_inc(v_a_700_);
lean_dec(v___x_635_);
v___x_702_ = lean_box(0);
v_isShared_703_ = v_isSharedCheck_707_;
goto v_resetjp_701_;
}
v_resetjp_701_:
{
lean_object* v___x_705_; 
if (v_isShared_703_ == 0)
{
v___x_705_ = v___x_702_;
goto v_reusejp_704_;
}
else
{
lean_object* v_reuseFailAlloc_706_; 
v_reuseFailAlloc_706_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_706_, 0, v_a_700_);
v___x_705_ = v_reuseFailAlloc_706_;
goto v_reusejp_704_;
}
v_reusejp_704_:
{
return v___x_705_;
}
}
}
}
}
case 3:
{
lean_object* v_inst_708_; lean_object* v_n_709_; lean_object* v_d_710_; lean_object* v_proof_711_; lean_object* v___x_713_; uint8_t v_isShared_714_; uint8_t v_isSharedCheck_816_; 
lean_dec_ref(v_e_539_);
v_inst_708_ = lean_ctor_get(v_ra_546_, 0);
v_n_709_ = lean_ctor_get(v_ra_546_, 2);
v_d_710_ = lean_ctor_get(v_ra_546_, 3);
v_proof_711_ = lean_ctor_get(v_ra_546_, 4);
v_isSharedCheck_816_ = !lean_is_exclusive(v_ra_546_);
if (v_isSharedCheck_816_ == 0)
{
lean_object* v_unused_817_; 
v_unused_817_ = lean_ctor_get(v_ra_546_, 1);
lean_dec(v_unused_817_);
v___x_713_ = v_ra_546_;
v_isShared_714_ = v_isSharedCheck_816_;
goto v_resetjp_712_;
}
else
{
lean_inc(v_proof_711_);
lean_inc(v_d_710_);
lean_inc(v_n_709_);
lean_inc(v_inst_708_);
lean_dec(v_ra_546_);
v___x_713_ = lean_box(0);
v_isShared_714_ = v_isSharedCheck_816_;
goto v_resetjp_712_;
}
v_resetjp_712_:
{
lean_object* v___x_715_; 
lean_inc_ref(v_nb_543_);
lean_inc_ref(v_n_709_);
v___x_715_ = lp_mathlib_Mathlib_Meta_NormNum_evalNatPow(v_n_709_, v_nb_543_, v_a_547_, v_a_548_);
if (lean_obj_tag(v___x_715_) == 0)
{
lean_object* v_a_716_; lean_object* v___x_718_; uint8_t v_isShared_719_; uint8_t v_isSharedCheck_807_; 
v_a_716_ = lean_ctor_get(v___x_715_, 0);
v_isSharedCheck_807_ = !lean_is_exclusive(v___x_715_);
if (v_isSharedCheck_807_ == 0)
{
v___x_718_ = v___x_715_;
v_isShared_719_ = v_isSharedCheck_807_;
goto v_resetjp_717_;
}
else
{
lean_inc(v_a_716_);
lean_dec(v___x_715_);
v___x_718_ = lean_box(0);
v_isShared_719_ = v_isSharedCheck_807_;
goto v_resetjp_717_;
}
v_resetjp_717_:
{
if (lean_obj_tag(v_a_716_) == 0)
{
lean_object* v___x_720_; lean_object* v___x_722_; 
lean_del_object(v___x_713_);
lean_dec_ref(v_proof_711_);
lean_dec_ref(v_d_710_);
lean_dec_ref(v_n_709_);
lean_dec_ref(v_inst_708_);
lean_dec_ref(v_s_u03b1_545_);
lean_dec_ref(v_pb_544_);
lean_dec_ref(v_nb_543_);
lean_dec_ref(v_b_542_);
lean_dec_ref(v_a_541_);
lean_dec_ref(v_f_540_);
lean_dec_ref(v_00_u03b1_538_);
lean_dec(v_u_537_);
v___x_720_ = lean_box(0);
if (v_isShared_719_ == 0)
{
lean_ctor_set(v___x_718_, 0, v___x_720_);
v___x_722_ = v___x_718_;
goto v_reusejp_721_;
}
else
{
lean_object* v_reuseFailAlloc_723_; 
v_reuseFailAlloc_723_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_723_, 0, v___x_720_);
v___x_722_ = v_reuseFailAlloc_723_;
goto v_reusejp_721_;
}
v_reusejp_721_:
{
return v___x_722_;
}
}
else
{
lean_object* v_val_724_; lean_object* v_fst_725_; lean_object* v_snd_726_; lean_object* v___x_728_; uint8_t v_isShared_729_; uint8_t v_isSharedCheck_806_; 
lean_del_object(v___x_718_);
v_val_724_ = lean_ctor_get(v_a_716_, 0);
lean_inc(v_val_724_);
lean_dec_ref_known(v_a_716_, 1);
v_fst_725_ = lean_ctor_get(v_val_724_, 0);
v_snd_726_ = lean_ctor_get(v_val_724_, 1);
v_isSharedCheck_806_ = !lean_is_exclusive(v_val_724_);
if (v_isSharedCheck_806_ == 0)
{
v___x_728_ = v_val_724_;
v_isShared_729_ = v_isSharedCheck_806_;
goto v_resetjp_727_;
}
else
{
lean_inc(v_snd_726_);
lean_inc(v_fst_725_);
lean_dec(v_val_724_);
v___x_728_ = lean_box(0);
v_isShared_729_ = v_isSharedCheck_806_;
goto v_resetjp_727_;
}
v_resetjp_727_:
{
lean_object* v___x_730_; 
lean_inc_ref(v_nb_543_);
lean_inc_ref(v_d_710_);
v___x_730_ = lp_mathlib_Mathlib_Meta_NormNum_evalNatPow(v_d_710_, v_nb_543_, v_a_547_, v_a_548_);
if (lean_obj_tag(v___x_730_) == 0)
{
lean_object* v_a_731_; lean_object* v___x_733_; uint8_t v_isShared_734_; uint8_t v_isSharedCheck_797_; 
v_a_731_ = lean_ctor_get(v___x_730_, 0);
v_isSharedCheck_797_ = !lean_is_exclusive(v___x_730_);
if (v_isSharedCheck_797_ == 0)
{
v___x_733_ = v___x_730_;
v_isShared_734_ = v_isSharedCheck_797_;
goto v_resetjp_732_;
}
else
{
lean_inc(v_a_731_);
lean_dec(v___x_730_);
v___x_733_ = lean_box(0);
v_isShared_734_ = v_isSharedCheck_797_;
goto v_resetjp_732_;
}
v_resetjp_732_:
{
if (lean_obj_tag(v_a_731_) == 0)
{
lean_object* v___x_735_; lean_object* v___x_737_; 
lean_del_object(v___x_728_);
lean_dec(v_snd_726_);
lean_dec(v_fst_725_);
lean_del_object(v___x_713_);
lean_dec_ref(v_proof_711_);
lean_dec_ref(v_d_710_);
lean_dec_ref(v_n_709_);
lean_dec_ref(v_inst_708_);
lean_dec_ref(v_s_u03b1_545_);
lean_dec_ref(v_pb_544_);
lean_dec_ref(v_nb_543_);
lean_dec_ref(v_b_542_);
lean_dec_ref(v_a_541_);
lean_dec_ref(v_f_540_);
lean_dec_ref(v_00_u03b1_538_);
lean_dec(v_u_537_);
v___x_735_ = lean_box(0);
if (v_isShared_734_ == 0)
{
lean_ctor_set(v___x_733_, 0, v___x_735_);
v___x_737_ = v___x_733_;
goto v_reusejp_736_;
}
else
{
lean_object* v_reuseFailAlloc_738_; 
v_reuseFailAlloc_738_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_738_, 0, v___x_735_);
v___x_737_ = v_reuseFailAlloc_738_;
goto v_reusejp_736_;
}
v_reusejp_736_:
{
return v___x_737_;
}
}
else
{
lean_object* v_val_739_; lean_object* v___x_741_; uint8_t v_isShared_742_; uint8_t v_isSharedCheck_796_; 
v_val_739_ = lean_ctor_get(v_a_731_, 0);
v_isSharedCheck_796_ = !lean_is_exclusive(v_a_731_);
if (v_isSharedCheck_796_ == 0)
{
v___x_741_ = v_a_731_;
v_isShared_742_ = v_isSharedCheck_796_;
goto v_resetjp_740_;
}
else
{
lean_inc(v_val_739_);
lean_dec(v_a_731_);
v___x_741_ = lean_box(0);
v_isShared_742_ = v_isSharedCheck_796_;
goto v_resetjp_740_;
}
v_resetjp_740_:
{
lean_object* v_fst_743_; lean_object* v_snd_744_; lean_object* v___x_746_; uint8_t v_isShared_747_; uint8_t v_isSharedCheck_795_; 
v_fst_743_ = lean_ctor_get(v_val_739_, 0);
v_snd_744_ = lean_ctor_get(v_val_739_, 1);
v_isSharedCheck_795_ = !lean_is_exclusive(v_val_739_);
if (v_isSharedCheck_795_ == 0)
{
v___x_746_ = v_val_739_;
v_isShared_747_ = v_isSharedCheck_795_;
goto v_resetjp_745_;
}
else
{
lean_inc(v_snd_744_);
lean_inc(v_fst_743_);
lean_dec(v_val_739_);
v___x_746_ = lean_box(0);
v_isShared_747_ = v_isSharedCheck_795_;
goto v_resetjp_745_;
}
v_resetjp_745_:
{
lean_object* v___x_748_; lean_object* v___x_749_; lean_object* v___x_750_; lean_object* v___x_751_; lean_object* v___x_752_; lean_object* v___x_754_; 
v___x_748_ = lp_batteries_Lean_Expr_natLit_x21(v_fst_725_);
v___x_749_ = lean_nat_to_int(v___x_748_);
v___x_750_ = lp_batteries_Lean_Expr_natLit_x21(v_fst_743_);
v___x_751_ = l_mkRat(v___x_749_, v___x_750_);
v___x_752_ = lean_box(0);
lean_inc(v_u_537_);
if (v_isShared_747_ == 0)
{
lean_ctor_set_tag(v___x_746_, 1);
lean_ctor_set(v___x_746_, 1, v___x_752_);
lean_ctor_set(v___x_746_, 0, v_u_537_);
v___x_754_ = v___x_746_;
goto v_reusejp_753_;
}
else
{
lean_object* v_reuseFailAlloc_794_; 
v_reuseFailAlloc_794_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_794_, 0, v_u_537_);
lean_ctor_set(v_reuseFailAlloc_794_, 1, v___x_752_);
v___x_754_ = v_reuseFailAlloc_794_;
goto v_reusejp_753_;
}
v_reusejp_753_:
{
lean_object* v___x_755_; lean_object* v___x_756_; lean_object* v___x_757_; lean_object* v___x_758_; lean_object* v___x_759_; lean_object* v___x_760_; lean_object* v___x_761_; lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v___x_764_; lean_object* v___x_765_; lean_object* v___x_766_; lean_object* v___x_767_; lean_object* v___x_768_; lean_object* v___x_770_; 
v___x_755_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__5));
v___x_756_ = l_Lean_Expr_const___override(v___x_755_, v___x_754_);
lean_inc_ref(v_00_u03b1_538_);
v___x_757_ = l_Lean_Expr_app___override(v___x_756_, v_00_u03b1_538_);
v___x_758_ = l_Lean_Expr_app___override(v___x_757_, v_s_u03b1_545_);
lean_inc_ref(v_f_540_);
v___x_759_ = l_Lean_Expr_app___override(v___x_758_, v_f_540_);
v___x_760_ = l_Lean_Expr_app___override(v___x_759_, v_a_541_);
v___x_761_ = l_Lean_Expr_app___override(v___x_760_, v_n_709_);
lean_inc(v_fst_725_);
v___x_762_ = l_Lean_Expr_app___override(v___x_761_, v_fst_725_);
v___x_763_ = l_Lean_Expr_app___override(v___x_762_, v_d_710_);
v___x_764_ = l_Lean_Expr_app___override(v___x_763_, v_b_542_);
v___x_765_ = l_Lean_Expr_app___override(v___x_764_, v_nb_543_);
lean_inc(v_fst_743_);
v___x_766_ = l_Lean_Expr_app___override(v___x_765_, v_fst_743_);
v___x_767_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__2));
v___x_768_ = l_Lean_Level_succ___override(v_u_537_);
if (v_isShared_729_ == 0)
{
lean_ctor_set_tag(v___x_728_, 1);
lean_ctor_set(v___x_728_, 1, v___x_752_);
lean_ctor_set(v___x_728_, 0, v___x_768_);
v___x_770_ = v___x_728_;
goto v_reusejp_769_;
}
else
{
lean_object* v_reuseFailAlloc_793_; 
v_reuseFailAlloc_793_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_793_, 0, v___x_768_);
lean_ctor_set(v_reuseFailAlloc_793_, 1, v___x_752_);
v___x_770_ = v_reuseFailAlloc_793_;
goto v_reusejp_769_;
}
v_reusejp_769_:
{
lean_object* v___x_771_; lean_object* v___x_772_; lean_object* v___x_773_; uint8_t v___x_774_; lean_object* v___x_775_; lean_object* v___x_776_; lean_object* v___x_777_; lean_object* v___x_778_; lean_object* v___x_779_; lean_object* v___x_780_; lean_object* v___x_781_; lean_object* v___x_782_; lean_object* v___x_783_; lean_object* v___x_785_; 
v___x_771_ = l_Lean_Expr_const___override(v___x_767_, v___x_770_);
v___x_772_ = lean_box(0);
v___x_773_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__7, &lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__7);
v___x_774_ = 0;
lean_inc_ref(v_00_u03b1_538_);
v___x_775_ = l_Lean_Expr_forallE___override(v___x_772_, v___x_773_, v_00_u03b1_538_, v___x_774_);
v___x_776_ = l_Lean_Expr_forallE___override(v___x_772_, v_00_u03b1_538_, v___x_775_, v___x_774_);
v___x_777_ = l_Lean_Expr_app___override(v___x_771_, v___x_776_);
v___x_778_ = l_Lean_Expr_app___override(v___x_777_, v_f_540_);
v___x_779_ = l_Lean_Expr_app___override(v___x_766_, v___x_778_);
v___x_780_ = l_Lean_Expr_app___override(v___x_779_, v_proof_711_);
v___x_781_ = l_Lean_Expr_app___override(v___x_780_, v_pb_544_);
v___x_782_ = l_Lean_Expr_app___override(v___x_781_, v_snd_726_);
v___x_783_ = l_Lean_Expr_app___override(v___x_782_, v_snd_744_);
if (v_isShared_714_ == 0)
{
lean_ctor_set(v___x_713_, 4, v___x_783_);
lean_ctor_set(v___x_713_, 3, v_fst_743_);
lean_ctor_set(v___x_713_, 2, v_fst_725_);
lean_ctor_set(v___x_713_, 1, v___x_751_);
v___x_785_ = v___x_713_;
goto v_reusejp_784_;
}
else
{
lean_object* v_reuseFailAlloc_792_; 
v_reuseFailAlloc_792_ = lean_alloc_ctor(3, 5, 0);
lean_ctor_set(v_reuseFailAlloc_792_, 0, v_inst_708_);
lean_ctor_set(v_reuseFailAlloc_792_, 1, v___x_751_);
lean_ctor_set(v_reuseFailAlloc_792_, 2, v_fst_725_);
lean_ctor_set(v_reuseFailAlloc_792_, 3, v_fst_743_);
lean_ctor_set(v_reuseFailAlloc_792_, 4, v___x_783_);
v___x_785_ = v_reuseFailAlloc_792_;
goto v_reusejp_784_;
}
v_reusejp_784_:
{
lean_object* v___x_787_; 
if (v_isShared_742_ == 0)
{
lean_ctor_set(v___x_741_, 0, v___x_785_);
v___x_787_ = v___x_741_;
goto v_reusejp_786_;
}
else
{
lean_object* v_reuseFailAlloc_791_; 
v_reuseFailAlloc_791_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_791_, 0, v___x_785_);
v___x_787_ = v_reuseFailAlloc_791_;
goto v_reusejp_786_;
}
v_reusejp_786_:
{
lean_object* v___x_789_; 
if (v_isShared_734_ == 0)
{
lean_ctor_set(v___x_733_, 0, v___x_787_);
v___x_789_ = v___x_733_;
goto v_reusejp_788_;
}
else
{
lean_object* v_reuseFailAlloc_790_; 
v_reuseFailAlloc_790_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_790_, 0, v___x_787_);
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
}
}
}
}
}
}
else
{
lean_object* v_a_798_; lean_object* v___x_800_; uint8_t v_isShared_801_; uint8_t v_isSharedCheck_805_; 
lean_del_object(v___x_728_);
lean_dec(v_snd_726_);
lean_dec(v_fst_725_);
lean_del_object(v___x_713_);
lean_dec_ref(v_proof_711_);
lean_dec_ref(v_d_710_);
lean_dec_ref(v_n_709_);
lean_dec_ref(v_inst_708_);
lean_dec_ref(v_s_u03b1_545_);
lean_dec_ref(v_pb_544_);
lean_dec_ref(v_nb_543_);
lean_dec_ref(v_b_542_);
lean_dec_ref(v_a_541_);
lean_dec_ref(v_f_540_);
lean_dec_ref(v_00_u03b1_538_);
lean_dec(v_u_537_);
v_a_798_ = lean_ctor_get(v___x_730_, 0);
v_isSharedCheck_805_ = !lean_is_exclusive(v___x_730_);
if (v_isSharedCheck_805_ == 0)
{
v___x_800_ = v___x_730_;
v_isShared_801_ = v_isSharedCheck_805_;
goto v_resetjp_799_;
}
else
{
lean_inc(v_a_798_);
lean_dec(v___x_730_);
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
}
}
}
}
else
{
lean_object* v_a_808_; lean_object* v___x_810_; uint8_t v_isShared_811_; uint8_t v_isSharedCheck_815_; 
lean_del_object(v___x_713_);
lean_dec_ref(v_proof_711_);
lean_dec_ref(v_d_710_);
lean_dec_ref(v_n_709_);
lean_dec_ref(v_inst_708_);
lean_dec_ref(v_s_u03b1_545_);
lean_dec_ref(v_pb_544_);
lean_dec_ref(v_nb_543_);
lean_dec_ref(v_b_542_);
lean_dec_ref(v_a_541_);
lean_dec_ref(v_f_540_);
lean_dec_ref(v_00_u03b1_538_);
lean_dec(v_u_537_);
v_a_808_ = lean_ctor_get(v___x_715_, 0);
v_isSharedCheck_815_ = !lean_is_exclusive(v___x_715_);
if (v_isSharedCheck_815_ == 0)
{
v___x_810_ = v___x_715_;
v_isShared_811_ = v_isSharedCheck_815_;
goto v_resetjp_809_;
}
else
{
lean_inc(v_a_808_);
lean_dec(v___x_715_);
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
}
default: 
{
lean_object* v_q_818_; lean_object* v_inst_819_; lean_object* v_n_820_; lean_object* v_d_821_; lean_object* v_proof_822_; lean_object* v_num_823_; lean_object* v___x_824_; lean_object* v___x_825_; lean_object* v___x_826_; lean_object* v___x_827_; 
lean_dec_ref(v_s_u03b1_545_);
v_q_818_ = lean_ctor_get(v_ra_546_, 1);
lean_inc_ref(v_q_818_);
v_inst_819_ = lean_ctor_get(v_ra_546_, 0);
lean_inc_ref(v_inst_819_);
v_n_820_ = lean_ctor_get(v_ra_546_, 2);
lean_inc_ref(v_n_820_);
v_d_821_ = lean_ctor_get(v_ra_546_, 3);
lean_inc_ref(v_d_821_);
v_proof_822_ = lean_ctor_get(v_ra_546_, 4);
lean_inc_ref(v_proof_822_);
lean_dec_ref_known(v_ra_546_, 5);
v_num_823_ = lean_ctor_get(v_q_818_, 0);
lean_inc(v_num_823_);
lean_dec_ref(v_q_818_);
v___x_824_ = lean_box(0);
v___x_825_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__8, &lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__8_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__8);
v___x_826_ = l_Lean_Expr_app___override(v___x_825_, v_n_820_);
lean_inc_ref(v_nb_543_);
v___x_827_ = lp_mathlib_Mathlib_Meta_NormNum_evalIntPow(v_num_823_, v___x_826_, v_nb_543_, v_a_547_, v_a_548_);
lean_dec(v_num_823_);
if (lean_obj_tag(v___x_827_) == 0)
{
lean_object* v_a_828_; lean_object* v___x_830_; uint8_t v_isShared_831_; uint8_t v_isSharedCheck_920_; 
v_a_828_ = lean_ctor_get(v___x_827_, 0);
v_isSharedCheck_920_ = !lean_is_exclusive(v___x_827_);
if (v_isSharedCheck_920_ == 0)
{
v___x_830_ = v___x_827_;
v_isShared_831_ = v_isSharedCheck_920_;
goto v_resetjp_829_;
}
else
{
lean_inc(v_a_828_);
lean_dec(v___x_827_);
v___x_830_ = lean_box(0);
v_isShared_831_ = v_isSharedCheck_920_;
goto v_resetjp_829_;
}
v_resetjp_829_:
{
if (lean_obj_tag(v_a_828_) == 0)
{
lean_object* v___x_832_; lean_object* v___x_834_; 
lean_dec_ref(v___x_826_);
lean_dec_ref(v_proof_822_);
lean_dec_ref(v_d_821_);
lean_dec_ref(v_inst_819_);
lean_dec_ref(v_pb_544_);
lean_dec_ref(v_nb_543_);
lean_dec_ref(v_b_542_);
lean_dec_ref(v_a_541_);
lean_dec_ref(v_f_540_);
lean_dec_ref(v_e_539_);
lean_dec_ref(v_00_u03b1_538_);
lean_dec(v_u_537_);
v___x_832_ = lean_box(0);
if (v_isShared_831_ == 0)
{
lean_ctor_set(v___x_830_, 0, v___x_832_);
v___x_834_ = v___x_830_;
goto v_reusejp_833_;
}
else
{
lean_object* v_reuseFailAlloc_835_; 
v_reuseFailAlloc_835_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_835_, 0, v___x_832_);
v___x_834_ = v_reuseFailAlloc_835_;
goto v_reusejp_833_;
}
v_reusejp_833_:
{
return v___x_834_;
}
}
else
{
lean_object* v_val_836_; lean_object* v_snd_837_; lean_object* v_fst_838_; lean_object* v_fst_839_; lean_object* v_snd_840_; lean_object* v___x_842_; uint8_t v_isShared_843_; uint8_t v_isSharedCheck_919_; 
lean_del_object(v___x_830_);
v_val_836_ = lean_ctor_get(v_a_828_, 0);
lean_inc(v_val_836_);
lean_dec_ref_known(v_a_828_, 1);
v_snd_837_ = lean_ctor_get(v_val_836_, 1);
lean_inc(v_snd_837_);
v_fst_838_ = lean_ctor_get(v_val_836_, 0);
lean_inc(v_fst_838_);
lean_dec(v_val_836_);
v_fst_839_ = lean_ctor_get(v_snd_837_, 0);
v_snd_840_ = lean_ctor_get(v_snd_837_, 1);
v_isSharedCheck_919_ = !lean_is_exclusive(v_snd_837_);
if (v_isSharedCheck_919_ == 0)
{
v___x_842_ = v_snd_837_;
v_isShared_843_ = v_isSharedCheck_919_;
goto v_resetjp_841_;
}
else
{
lean_inc(v_snd_840_);
lean_inc(v_fst_839_);
lean_dec(v_snd_837_);
v___x_842_ = lean_box(0);
v_isShared_843_ = v_isSharedCheck_919_;
goto v_resetjp_841_;
}
v_resetjp_841_:
{
lean_object* v___x_844_; 
lean_inc_ref(v_nb_543_);
lean_inc_ref(v_d_821_);
v___x_844_ = lp_mathlib_Mathlib_Meta_NormNum_evalNatPow(v_d_821_, v_nb_543_, v_a_547_, v_a_548_);
if (lean_obj_tag(v___x_844_) == 0)
{
lean_object* v_a_845_; lean_object* v___x_847_; uint8_t v_isShared_848_; uint8_t v_isSharedCheck_910_; 
v_a_845_ = lean_ctor_get(v___x_844_, 0);
v_isSharedCheck_910_ = !lean_is_exclusive(v___x_844_);
if (v_isSharedCheck_910_ == 0)
{
v___x_847_ = v___x_844_;
v_isShared_848_ = v_isSharedCheck_910_;
goto v_resetjp_846_;
}
else
{
lean_inc(v_a_845_);
lean_dec(v___x_844_);
v___x_847_ = lean_box(0);
v_isShared_848_ = v_isSharedCheck_910_;
goto v_resetjp_846_;
}
v_resetjp_846_:
{
if (lean_obj_tag(v_a_845_) == 0)
{
lean_object* v___x_849_; lean_object* v___x_851_; 
lean_del_object(v___x_842_);
lean_dec(v_snd_840_);
lean_dec(v_fst_839_);
lean_dec(v_fst_838_);
lean_dec_ref(v___x_826_);
lean_dec_ref(v_proof_822_);
lean_dec_ref(v_d_821_);
lean_dec_ref(v_inst_819_);
lean_dec_ref(v_pb_544_);
lean_dec_ref(v_nb_543_);
lean_dec_ref(v_b_542_);
lean_dec_ref(v_a_541_);
lean_dec_ref(v_f_540_);
lean_dec_ref(v_e_539_);
lean_dec_ref(v_00_u03b1_538_);
lean_dec(v_u_537_);
v___x_849_ = lean_box(0);
if (v_isShared_848_ == 0)
{
lean_ctor_set(v___x_847_, 0, v___x_849_);
v___x_851_ = v___x_847_;
goto v_reusejp_850_;
}
else
{
lean_object* v_reuseFailAlloc_852_; 
v_reuseFailAlloc_852_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_852_, 0, v___x_849_);
v___x_851_ = v_reuseFailAlloc_852_;
goto v_reusejp_850_;
}
v_reusejp_850_:
{
return v___x_851_;
}
}
else
{
lean_object* v_val_853_; lean_object* v___x_855_; uint8_t v_isShared_856_; uint8_t v_isSharedCheck_909_; 
v_val_853_ = lean_ctor_get(v_a_845_, 0);
v_isSharedCheck_909_ = !lean_is_exclusive(v_a_845_);
if (v_isSharedCheck_909_ == 0)
{
v___x_855_ = v_a_845_;
v_isShared_856_ = v_isSharedCheck_909_;
goto v_resetjp_854_;
}
else
{
lean_inc(v_val_853_);
lean_dec(v_a_845_);
v___x_855_ = lean_box(0);
v_isShared_856_ = v_isSharedCheck_909_;
goto v_resetjp_854_;
}
v_resetjp_854_:
{
lean_object* v_fst_857_; lean_object* v_snd_858_; lean_object* v___x_860_; uint8_t v_isShared_861_; uint8_t v_isSharedCheck_908_; 
v_fst_857_ = lean_ctor_get(v_val_853_, 0);
v_snd_858_ = lean_ctor_get(v_val_853_, 1);
v_isSharedCheck_908_ = !lean_is_exclusive(v_val_853_);
if (v_isSharedCheck_908_ == 0)
{
v___x_860_ = v_val_853_;
v_isShared_861_ = v_isSharedCheck_908_;
goto v_resetjp_859_;
}
else
{
lean_inc(v_snd_858_);
lean_inc(v_fst_857_);
lean_dec(v_val_853_);
v___x_860_ = lean_box(0);
v_isShared_861_ = v_isSharedCheck_908_;
goto v_resetjp_859_;
}
v_resetjp_859_:
{
lean_object* v___x_862_; lean_object* v___x_863_; lean_object* v___x_865_; 
v___x_862_ = lp_batteries_Lean_Expr_natLit_x21(v_fst_857_);
v___x_863_ = l_mkRat(v_fst_838_, v___x_862_);
lean_inc(v_u_537_);
if (v_isShared_861_ == 0)
{
lean_ctor_set_tag(v___x_860_, 1);
lean_ctor_set(v___x_860_, 1, v___x_824_);
lean_ctor_set(v___x_860_, 0, v_u_537_);
v___x_865_ = v___x_860_;
goto v_reusejp_864_;
}
else
{
lean_object* v_reuseFailAlloc_907_; 
v_reuseFailAlloc_907_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_907_, 0, v_u_537_);
lean_ctor_set(v_reuseFailAlloc_907_, 1, v___x_824_);
v___x_865_ = v_reuseFailAlloc_907_;
goto v_reusejp_864_;
}
v_reusejp_864_:
{
lean_object* v___x_866_; lean_object* v___x_867_; lean_object* v___x_868_; lean_object* v___x_869_; lean_object* v___x_870_; lean_object* v___x_871_; lean_object* v___x_872_; lean_object* v___x_873_; lean_object* v___x_874_; lean_object* v___x_875_; lean_object* v___x_876_; lean_object* v___x_877_; lean_object* v___x_878_; lean_object* v___x_879_; lean_object* v___x_880_; lean_object* v___x_881_; lean_object* v___x_882_; lean_object* v___x_883_; lean_object* v___x_885_; 
v___x_866_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__8));
lean_inc_ref(v___x_865_);
v___x_867_ = l_Lean_Expr_const___override(v___x_866_, v___x_865_);
lean_inc_ref_n(v_00_u03b1_538_, 2);
v___x_868_ = l_Lean_Expr_app___override(v___x_867_, v_00_u03b1_538_);
lean_inc_ref(v_inst_819_);
v___x_869_ = l_Lean_Expr_app___override(v___x_868_, v_inst_819_);
v___x_870_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___closed__10));
v___x_871_ = l_Lean_Expr_const___override(v___x_870_, v___x_865_);
v___x_872_ = l_Lean_Expr_app___override(v___x_871_, v_00_u03b1_538_);
v___x_873_ = l_Lean_Expr_app___override(v___x_872_, v___x_869_);
lean_inc_ref(v_f_540_);
v___x_874_ = l_Lean_Expr_app___override(v___x_873_, v_f_540_);
v___x_875_ = l_Lean_Expr_app___override(v___x_874_, v_a_541_);
v___x_876_ = l_Lean_Expr_app___override(v___x_875_, v___x_826_);
lean_inc(v_fst_839_);
v___x_877_ = l_Lean_Expr_app___override(v___x_876_, v_fst_839_);
v___x_878_ = l_Lean_Expr_app___override(v___x_877_, v_d_821_);
v___x_879_ = l_Lean_Expr_app___override(v___x_878_, v_b_542_);
v___x_880_ = l_Lean_Expr_app___override(v___x_879_, v_nb_543_);
lean_inc(v_fst_857_);
v___x_881_ = l_Lean_Expr_app___override(v___x_880_, v_fst_857_);
v___x_882_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__2));
lean_inc(v_u_537_);
v___x_883_ = l_Lean_Level_succ___override(v_u_537_);
if (v_isShared_843_ == 0)
{
lean_ctor_set_tag(v___x_842_, 1);
lean_ctor_set(v___x_842_, 1, v___x_824_);
lean_ctor_set(v___x_842_, 0, v___x_883_);
v___x_885_ = v___x_842_;
goto v_reusejp_884_;
}
else
{
lean_object* v_reuseFailAlloc_906_; 
v_reuseFailAlloc_906_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_906_, 0, v___x_883_);
lean_ctor_set(v_reuseFailAlloc_906_, 1, v___x_824_);
v___x_885_ = v_reuseFailAlloc_906_;
goto v_reusejp_884_;
}
v_reusejp_884_:
{
lean_object* v___x_886_; lean_object* v___x_887_; lean_object* v___x_888_; uint8_t v___x_889_; lean_object* v___x_890_; lean_object* v___x_891_; lean_object* v___x_892_; lean_object* v___x_893_; lean_object* v___x_894_; lean_object* v___x_895_; lean_object* v___x_896_; lean_object* v___x_897_; lean_object* v___x_898_; lean_object* v___x_899_; lean_object* v___x_901_; 
v___x_886_ = l_Lean_Expr_const___override(v___x_882_, v___x_885_);
v___x_887_ = lean_box(0);
v___x_888_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__7, &lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__7);
v___x_889_ = 0;
lean_inc_ref_n(v_00_u03b1_538_, 2);
v___x_890_ = l_Lean_Expr_forallE___override(v___x_887_, v___x_888_, v_00_u03b1_538_, v___x_889_);
v___x_891_ = l_Lean_Expr_forallE___override(v___x_887_, v_00_u03b1_538_, v___x_890_, v___x_889_);
v___x_892_ = l_Lean_Expr_app___override(v___x_886_, v___x_891_);
v___x_893_ = l_Lean_Expr_app___override(v___x_892_, v_f_540_);
v___x_894_ = l_Lean_Expr_app___override(v___x_881_, v___x_893_);
v___x_895_ = l_Lean_Expr_app___override(v___x_894_, v_proof_822_);
v___x_896_ = l_Lean_Expr_app___override(v___x_895_, v_pb_544_);
v___x_897_ = l_Lean_Expr_app___override(v___x_896_, v_snd_840_);
v___x_898_ = l_Lean_Expr_app___override(v___x_897_, v_snd_858_);
v___x_899_ = lp_mathlib_Mathlib_Meta_NormNum_Result_isRat(v_u_537_, v_00_u03b1_538_, v_e_539_, v_inst_819_, v___x_863_, v_fst_839_, v_fst_857_, v___x_898_);
if (v_isShared_856_ == 0)
{
lean_ctor_set(v___x_855_, 0, v___x_899_);
v___x_901_ = v___x_855_;
goto v_reusejp_900_;
}
else
{
lean_object* v_reuseFailAlloc_905_; 
v_reuseFailAlloc_905_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_905_, 0, v___x_899_);
v___x_901_ = v_reuseFailAlloc_905_;
goto v_reusejp_900_;
}
v_reusejp_900_:
{
lean_object* v___x_903_; 
if (v_isShared_848_ == 0)
{
lean_ctor_set(v___x_847_, 0, v___x_901_);
v___x_903_ = v___x_847_;
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
}
}
}
}
}
}
else
{
lean_object* v_a_911_; lean_object* v___x_913_; uint8_t v_isShared_914_; uint8_t v_isSharedCheck_918_; 
lean_del_object(v___x_842_);
lean_dec(v_snd_840_);
lean_dec(v_fst_839_);
lean_dec(v_fst_838_);
lean_dec_ref(v___x_826_);
lean_dec_ref(v_proof_822_);
lean_dec_ref(v_d_821_);
lean_dec_ref(v_inst_819_);
lean_dec_ref(v_pb_544_);
lean_dec_ref(v_nb_543_);
lean_dec_ref(v_b_542_);
lean_dec_ref(v_a_541_);
lean_dec_ref(v_f_540_);
lean_dec_ref(v_e_539_);
lean_dec_ref(v_00_u03b1_538_);
lean_dec(v_u_537_);
v_a_911_ = lean_ctor_get(v___x_844_, 0);
v_isSharedCheck_918_ = !lean_is_exclusive(v___x_844_);
if (v_isSharedCheck_918_ == 0)
{
v___x_913_ = v___x_844_;
v_isShared_914_ = v_isSharedCheck_918_;
goto v_resetjp_912_;
}
else
{
lean_inc(v_a_911_);
lean_dec(v___x_844_);
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
}
}
}
}
else
{
lean_object* v_a_921_; lean_object* v___x_923_; uint8_t v_isShared_924_; uint8_t v_isSharedCheck_928_; 
lean_dec_ref(v___x_826_);
lean_dec_ref(v_proof_822_);
lean_dec_ref(v_d_821_);
lean_dec_ref(v_inst_819_);
lean_dec_ref(v_pb_544_);
lean_dec_ref(v_nb_543_);
lean_dec_ref(v_b_542_);
lean_dec_ref(v_a_541_);
lean_dec_ref(v_f_540_);
lean_dec_ref(v_e_539_);
lean_dec_ref(v_00_u03b1_538_);
lean_dec(v_u_537_);
v_a_921_ = lean_ctor_get(v___x_827_, 0);
v_isSharedCheck_928_ = !lean_is_exclusive(v___x_827_);
if (v_isSharedCheck_928_ == 0)
{
v___x_923_ = v___x_827_;
v_isShared_924_ = v_isSharedCheck_928_;
goto v_resetjp_922_;
}
else
{
lean_inc(v_a_921_);
lean_dec(v___x_827_);
v___x_923_ = lean_box(0);
v_isShared_924_ = v_isSharedCheck_928_;
goto v_resetjp_922_;
}
v_resetjp_922_:
{
lean_object* v___x_926_; 
if (v_isShared_924_ == 0)
{
v___x_926_ = v___x_923_;
goto v_reusejp_925_;
}
else
{
lean_object* v_reuseFailAlloc_927_; 
v_reuseFailAlloc_927_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_927_, 0, v_a_921_);
v___x_926_ = v_reuseFailAlloc_927_;
goto v_reusejp_925_;
}
v_reusejp_925_:
{
return v___x_926_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow_core___boxed(lean_object* v_u_929_, lean_object* v_00_u03b1_930_, lean_object* v_e_931_, lean_object* v_f_932_, lean_object* v_a_933_, lean_object* v_b_934_, lean_object* v_nb_935_, lean_object* v_pb_936_, lean_object* v_s_u03b1_937_, lean_object* v_ra_938_, lean_object* v_a_939_, lean_object* v_a_940_, lean_object* v_a_941_){
_start:
{
lean_object* v_res_942_; 
v_res_942_ = lp_mathlib_Mathlib_Meta_NormNum_evalPow_core(v_u_929_, v_00_u03b1_930_, v_e_931_, v_f_932_, v_a_933_, v_b_934_, v_nb_935_, v_pb_936_, v_s_u03b1_937_, v_ra_938_, v_a_939_, v_a_940_);
lean_dec(v_a_940_);
lean_dec_ref(v_a_939_);
return v_res_942_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalPow_spec__1___redArg(lean_object* v_k_943_, uint8_t v_allowLevelAssignments_944_, lean_object* v___y_945_, lean_object* v___y_946_, lean_object* v___y_947_, lean_object* v___y_948_){
_start:
{
lean_object* v___x_950_; 
v___x_950_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), v_allowLevelAssignments_944_, v_k_943_, v___y_945_, v___y_946_, v___y_947_, v___y_948_);
if (lean_obj_tag(v___x_950_) == 0)
{
lean_object* v_a_951_; lean_object* v___x_953_; uint8_t v_isShared_954_; uint8_t v_isSharedCheck_958_; 
v_a_951_ = lean_ctor_get(v___x_950_, 0);
v_isSharedCheck_958_ = !lean_is_exclusive(v___x_950_);
if (v_isSharedCheck_958_ == 0)
{
v___x_953_ = v___x_950_;
v_isShared_954_ = v_isSharedCheck_958_;
goto v_resetjp_952_;
}
else
{
lean_inc(v_a_951_);
lean_dec(v___x_950_);
v___x_953_ = lean_box(0);
v_isShared_954_ = v_isSharedCheck_958_;
goto v_resetjp_952_;
}
v_resetjp_952_:
{
lean_object* v___x_956_; 
if (v_isShared_954_ == 0)
{
v___x_956_ = v___x_953_;
goto v_reusejp_955_;
}
else
{
lean_object* v_reuseFailAlloc_957_; 
v_reuseFailAlloc_957_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_957_, 0, v_a_951_);
v___x_956_ = v_reuseFailAlloc_957_;
goto v_reusejp_955_;
}
v_reusejp_955_:
{
return v___x_956_;
}
}
}
else
{
lean_object* v_a_959_; lean_object* v___x_961_; uint8_t v_isShared_962_; uint8_t v_isSharedCheck_966_; 
v_a_959_ = lean_ctor_get(v___x_950_, 0);
v_isSharedCheck_966_ = !lean_is_exclusive(v___x_950_);
if (v_isSharedCheck_966_ == 0)
{
v___x_961_ = v___x_950_;
v_isShared_962_ = v_isSharedCheck_966_;
goto v_resetjp_960_;
}
else
{
lean_inc(v_a_959_);
lean_dec(v___x_950_);
v___x_961_ = lean_box(0);
v_isShared_962_ = v_isSharedCheck_966_;
goto v_resetjp_960_;
}
v_resetjp_960_:
{
lean_object* v___x_964_; 
if (v_isShared_962_ == 0)
{
v___x_964_ = v___x_961_;
goto v_reusejp_963_;
}
else
{
lean_object* v_reuseFailAlloc_965_; 
v_reuseFailAlloc_965_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_965_, 0, v_a_959_);
v___x_964_ = v_reuseFailAlloc_965_;
goto v_reusejp_963_;
}
v_reusejp_963_:
{
return v___x_964_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalPow_spec__1___redArg___boxed(lean_object* v_k_967_, lean_object* v_allowLevelAssignments_968_, lean_object* v___y_969_, lean_object* v___y_970_, lean_object* v___y_971_, lean_object* v___y_972_, lean_object* v___y_973_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_974_; lean_object* v_res_975_; 
v_allowLevelAssignments_boxed_974_ = lean_unbox(v_allowLevelAssignments_968_);
v_res_975_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalPow_spec__1___redArg(v_k_967_, v_allowLevelAssignments_boxed_974_, v___y_969_, v___y_970_, v___y_971_, v___y_972_);
lean_dec(v___y_972_);
lean_dec_ref(v___y_971_);
lean_dec(v___y_970_);
lean_dec_ref(v___y_969_);
return v_res_975_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalPow_spec__1(lean_object* v_00_u03b1_976_, lean_object* v_k_977_, uint8_t v_allowLevelAssignments_978_, lean_object* v___y_979_, lean_object* v___y_980_, lean_object* v___y_981_, lean_object* v___y_982_){
_start:
{
lean_object* v___x_984_; 
v___x_984_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalPow_spec__1___redArg(v_k_977_, v_allowLevelAssignments_978_, v___y_979_, v___y_980_, v___y_981_, v___y_982_);
return v___x_984_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalPow_spec__1___boxed(lean_object* v_00_u03b1_985_, lean_object* v_k_986_, lean_object* v_allowLevelAssignments_987_, lean_object* v___y_988_, lean_object* v___y_989_, lean_object* v___y_990_, lean_object* v___y_991_, lean_object* v___y_992_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_993_; lean_object* v_res_994_; 
v_allowLevelAssignments_boxed_993_ = lean_unbox(v_allowLevelAssignments_987_);
v_res_994_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalPow_spec__1(v_00_u03b1_985_, v_k_986_, v_allowLevelAssignments_boxed_993_, v___y_988_, v___y_989_, v___y_990_, v___y_991_);
lean_dec(v___y_991_);
lean_dec_ref(v___y_990_);
lean_dec(v___y_989_);
lean_dec_ref(v___y_988_);
return v_res_994_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__0(lean_object* v_fn_995_, lean_object* v___x_996_, lean_object* v___y_997_, lean_object* v___y_998_, lean_object* v___y_999_, lean_object* v___y_1000_){
_start:
{
lean_object* v___x_1002_; 
v___x_1002_ = l_Lean_Meta_isExprDefEq(v_fn_995_, v___x_996_, v___y_997_, v___y_998_, v___y_999_, v___y_1000_);
return v___x_1002_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__0___boxed(lean_object* v_fn_1003_, lean_object* v___x_1004_, lean_object* v___y_1005_, lean_object* v___y_1006_, lean_object* v___y_1007_, lean_object* v___y_1008_, lean_object* v___y_1009_){
_start:
{
lean_object* v_res_1010_; 
v_res_1010_ = lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__0(v_fn_1003_, v___x_1004_, v___y_1005_, v___y_1006_, v___y_1007_, v___y_1008_);
lean_dec(v___y_1008_);
lean_dec_ref(v___y_1007_);
lean_dec(v___y_1006_);
lean_dec_ref(v___y_1005_);
return v_res_1010_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_evalPow_spec__0_spec__0(lean_object* v_msgData_1011_, lean_object* v___y_1012_, lean_object* v___y_1013_, lean_object* v___y_1014_, lean_object* v___y_1015_){
_start:
{
lean_object* v___x_1017_; lean_object* v_env_1018_; lean_object* v___x_1019_; lean_object* v_mctx_1020_; lean_object* v_lctx_1021_; lean_object* v_options_1022_; lean_object* v___x_1023_; lean_object* v___x_1024_; lean_object* v___x_1025_; 
v___x_1017_ = lean_st_ref_get(v___y_1015_);
v_env_1018_ = lean_ctor_get(v___x_1017_, 0);
lean_inc_ref(v_env_1018_);
lean_dec(v___x_1017_);
v___x_1019_ = lean_st_ref_get(v___y_1013_);
v_mctx_1020_ = lean_ctor_get(v___x_1019_, 0);
lean_inc_ref(v_mctx_1020_);
lean_dec(v___x_1019_);
v_lctx_1021_ = lean_ctor_get(v___y_1012_, 2);
v_options_1022_ = lean_ctor_get(v___y_1014_, 2);
lean_inc_ref(v_options_1022_);
lean_inc_ref(v_lctx_1021_);
v___x_1023_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1023_, 0, v_env_1018_);
lean_ctor_set(v___x_1023_, 1, v_mctx_1020_);
lean_ctor_set(v___x_1023_, 2, v_lctx_1021_);
lean_ctor_set(v___x_1023_, 3, v_options_1022_);
v___x_1024_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1024_, 0, v___x_1023_);
lean_ctor_set(v___x_1024_, 1, v_msgData_1011_);
v___x_1025_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1025_, 0, v___x_1024_);
return v___x_1025_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_evalPow_spec__0_spec__0___boxed(lean_object* v_msgData_1026_, lean_object* v___y_1027_, lean_object* v___y_1028_, lean_object* v___y_1029_, lean_object* v___y_1030_, lean_object* v___y_1031_){
_start:
{
lean_object* v_res_1032_; 
v_res_1032_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_evalPow_spec__0_spec__0(v_msgData_1026_, v___y_1027_, v___y_1028_, v___y_1029_, v___y_1030_);
lean_dec(v___y_1030_);
lean_dec_ref(v___y_1029_);
lean_dec(v___y_1028_);
lean_dec_ref(v___y_1027_);
return v_res_1032_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalPow_spec__0___redArg(lean_object* v_msg_1033_, lean_object* v___y_1034_, lean_object* v___y_1035_, lean_object* v___y_1036_, lean_object* v___y_1037_){
_start:
{
lean_object* v_ref_1039_; lean_object* v___x_1040_; lean_object* v_a_1041_; lean_object* v___x_1043_; uint8_t v_isShared_1044_; uint8_t v_isSharedCheck_1049_; 
v_ref_1039_ = lean_ctor_get(v___y_1036_, 5);
v___x_1040_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_evalPow_spec__0_spec__0(v_msg_1033_, v___y_1034_, v___y_1035_, v___y_1036_, v___y_1037_);
v_a_1041_ = lean_ctor_get(v___x_1040_, 0);
v_isSharedCheck_1049_ = !lean_is_exclusive(v___x_1040_);
if (v_isSharedCheck_1049_ == 0)
{
v___x_1043_ = v___x_1040_;
v_isShared_1044_ = v_isSharedCheck_1049_;
goto v_resetjp_1042_;
}
else
{
lean_inc(v_a_1041_);
lean_dec(v___x_1040_);
v___x_1043_ = lean_box(0);
v_isShared_1044_ = v_isSharedCheck_1049_;
goto v_resetjp_1042_;
}
v_resetjp_1042_:
{
lean_object* v___x_1045_; lean_object* v___x_1047_; 
lean_inc(v_ref_1039_);
v___x_1045_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1045_, 0, v_ref_1039_);
lean_ctor_set(v___x_1045_, 1, v_a_1041_);
if (v_isShared_1044_ == 0)
{
lean_ctor_set_tag(v___x_1043_, 1);
lean_ctor_set(v___x_1043_, 0, v___x_1045_);
v___x_1047_ = v___x_1043_;
goto v_reusejp_1046_;
}
else
{
lean_object* v_reuseFailAlloc_1048_; 
v_reuseFailAlloc_1048_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1048_, 0, v___x_1045_);
v___x_1047_ = v_reuseFailAlloc_1048_;
goto v_reusejp_1046_;
}
v_reusejp_1046_:
{
return v___x_1047_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalPow_spec__0___redArg___boxed(lean_object* v_msg_1050_, lean_object* v___y_1051_, lean_object* v___y_1052_, lean_object* v___y_1053_, lean_object* v___y_1054_, lean_object* v___y_1055_){
_start:
{
lean_object* v_res_1056_; 
v_res_1056_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalPow_spec__0___redArg(v_msg_1050_, v___y_1051_, v___y_1052_, v___y_1053_, v___y_1054_);
lean_dec(v___y_1054_);
lean_dec_ref(v___y_1053_);
lean_dec(v___y_1052_);
lean_dec_ref(v___y_1051_);
return v_res_1056_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__1(void){
_start:
{
lean_object* v___x_1058_; lean_object* v___x_1059_; 
v___x_1058_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__0));
v___x_1059_ = l_Lean_stringToMessageData(v___x_1058_);
return v___x_1059_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1(lean_object* v_u_1086_, lean_object* v_00_u03b1_1087_, lean_object* v_e_1088_, lean_object* v___y_1089_, lean_object* v___y_1090_, lean_object* v___y_1091_, lean_object* v___y_1092_){
_start:
{
lean_object* v___x_1094_; 
lean_inc_ref(v_e_1088_);
v___x_1094_ = l_Lean_Meta_whnfR(v_e_1088_, v___y_1089_, v___y_1090_, v___y_1091_, v___y_1092_);
if (lean_obj_tag(v___x_1094_) == 0)
{
lean_object* v_a_1095_; lean_object* v___y_1097_; lean_object* v___y_1098_; lean_object* v___y_1099_; lean_object* v___y_1100_; 
v_a_1095_ = lean_ctor_get(v___x_1094_, 0);
lean_inc(v_a_1095_);
lean_dec_ref_known(v___x_1094_, 1);
if (lean_obj_tag(v_a_1095_) == 5)
{
lean_object* v_fn_1103_; 
v_fn_1103_ = lean_ctor_get(v_a_1095_, 0);
lean_inc_ref(v_fn_1103_);
if (lean_obj_tag(v_fn_1103_) == 5)
{
lean_object* v_arg_1104_; lean_object* v_fn_1105_; lean_object* v_arg_1106_; lean_object* v___x_1107_; lean_object* v___x_1108_; lean_object* v___x_1109_; lean_object* v___x_1110_; lean_object* v___x_1111_; 
v_arg_1104_ = lean_ctor_get(v_a_1095_, 1);
lean_inc_ref_n(v_arg_1104_, 2);
lean_dec_ref_known(v_a_1095_, 2);
v_fn_1105_ = lean_ctor_get(v_fn_1103_, 0);
lean_inc_ref(v_fn_1105_);
v_arg_1106_ = lean_ctor_get(v_fn_1103_, 1);
lean_inc_ref(v_arg_1106_);
lean_dec_ref_known(v_fn_1103_, 2);
v___x_1107_ = lean_box(0);
v___x_1108_ = lean_box(0);
v___x_1109_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__7, &lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__7);
v___x_1110_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__2));
v___x_1111_ = lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg(v___x_1107_, v___x_1109_, v_arg_1104_, v___y_1089_, v___y_1090_, v___y_1091_, v___y_1092_);
if (lean_obj_tag(v___x_1111_) == 0)
{
lean_object* v_a_1112_; lean_object* v_fst_1113_; lean_object* v_snd_1114_; lean_object* v___x_1116_; uint8_t v_isShared_1117_; uint8_t v_isSharedCheck_1217_; 
v_a_1112_ = lean_ctor_get(v___x_1111_, 0);
lean_inc(v_a_1112_);
lean_dec_ref_known(v___x_1111_, 1);
v_fst_1113_ = lean_ctor_get(v_a_1112_, 0);
v_snd_1114_ = lean_ctor_get(v_a_1112_, 1);
v_isSharedCheck_1217_ = !lean_is_exclusive(v_a_1112_);
if (v_isSharedCheck_1217_ == 0)
{
v___x_1116_ = v_a_1112_;
v_isShared_1117_ = v_isSharedCheck_1217_;
goto v_resetjp_1115_;
}
else
{
lean_inc(v_snd_1114_);
lean_inc(v_fst_1113_);
lean_dec(v_a_1112_);
v___x_1116_ = lean_box(0);
v_isShared_1117_ = v_isSharedCheck_1217_;
goto v_resetjp_1115_;
}
v_resetjp_1115_:
{
lean_object* v___x_1118_; 
lean_inc_ref(v_00_u03b1_1087_);
lean_inc(v_u_1086_);
v___x_1118_ = lp_mathlib_Mathlib_Meta_NormNum_inferSemiring(v_u_1086_, v_00_u03b1_1087_, v___y_1089_, v___y_1090_, v___y_1091_, v___y_1092_);
if (lean_obj_tag(v___x_1118_) == 0)
{
lean_object* v_a_1119_; uint8_t v___x_1120_; lean_object* v___x_1121_; 
v_a_1119_ = lean_ctor_get(v___x_1118_, 0);
lean_inc(v_a_1119_);
lean_dec_ref_known(v___x_1118_, 1);
v___x_1120_ = 0;
lean_inc_ref(v_arg_1106_);
lean_inc_ref(v_00_u03b1_1087_);
lean_inc(v_u_1086_);
v___x_1121_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v_u_1086_, v_00_u03b1_1087_, v_arg_1106_, v___x_1120_, v___y_1089_, v___y_1090_, v___y_1091_, v___y_1092_);
if (lean_obj_tag(v___x_1121_) == 0)
{
lean_object* v_a_1122_; lean_object* v___x_1144_; lean_object* v___x_1146_; 
v_a_1122_ = lean_ctor_get(v___x_1121_, 0);
lean_inc(v_a_1122_);
lean_dec_ref_known(v___x_1121_, 1);
v___x_1144_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__5));
lean_inc(v_u_1086_);
if (v_isShared_1117_ == 0)
{
lean_ctor_set_tag(v___x_1116_, 1);
lean_ctor_set(v___x_1116_, 1, v___x_1108_);
lean_ctor_set(v___x_1116_, 0, v_u_1086_);
v___x_1146_ = v___x_1116_;
goto v_reusejp_1145_;
}
else
{
lean_object* v_reuseFailAlloc_1208_; 
v_reuseFailAlloc_1208_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1208_, 0, v_u_1086_);
lean_ctor_set(v_reuseFailAlloc_1208_, 1, v___x_1108_);
v___x_1146_ = v_reuseFailAlloc_1208_;
goto v_reusejp_1145_;
}
v___jp_1123_:
{
lean_object* v___x_1124_; 
v___x_1124_ = lp_mathlib_Mathlib_Meta_NormNum_evalPow_core(v_u_1086_, v_00_u03b1_1087_, v_e_1088_, v_fn_1105_, v_arg_1106_, v_arg_1104_, v_fst_1113_, v_snd_1114_, v_a_1119_, v_a_1122_, v___y_1091_, v___y_1092_);
if (lean_obj_tag(v___x_1124_) == 0)
{
lean_object* v_a_1125_; lean_object* v___x_1127_; uint8_t v_isShared_1128_; uint8_t v_isSharedCheck_1135_; 
v_a_1125_ = lean_ctor_get(v___x_1124_, 0);
v_isSharedCheck_1135_ = !lean_is_exclusive(v___x_1124_);
if (v_isSharedCheck_1135_ == 0)
{
v___x_1127_ = v___x_1124_;
v_isShared_1128_ = v_isSharedCheck_1135_;
goto v_resetjp_1126_;
}
else
{
lean_inc(v_a_1125_);
lean_dec(v___x_1124_);
v___x_1127_ = lean_box(0);
v_isShared_1128_ = v_isSharedCheck_1135_;
goto v_resetjp_1126_;
}
v_resetjp_1126_:
{
if (lean_obj_tag(v_a_1125_) == 1)
{
lean_object* v_val_1129_; lean_object* v___x_1131_; 
v_val_1129_ = lean_ctor_get(v_a_1125_, 0);
lean_inc(v_val_1129_);
lean_dec_ref_known(v_a_1125_, 1);
if (v_isShared_1128_ == 0)
{
lean_ctor_set(v___x_1127_, 0, v_val_1129_);
v___x_1131_ = v___x_1127_;
goto v_reusejp_1130_;
}
else
{
lean_object* v_reuseFailAlloc_1132_; 
v_reuseFailAlloc_1132_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1132_, 0, v_val_1129_);
v___x_1131_ = v_reuseFailAlloc_1132_;
goto v_reusejp_1130_;
}
v_reusejp_1130_:
{
return v___x_1131_;
}
}
else
{
lean_object* v___x_1133_; lean_object* v___x_1134_; 
lean_del_object(v___x_1127_);
lean_dec(v_a_1125_);
v___x_1133_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__1);
v___x_1134_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalPow_spec__0___redArg(v___x_1133_, v___y_1089_, v___y_1090_, v___y_1091_, v___y_1092_);
return v___x_1134_;
}
}
}
else
{
lean_object* v_a_1136_; lean_object* v___x_1138_; uint8_t v_isShared_1139_; uint8_t v_isSharedCheck_1143_; 
v_a_1136_ = lean_ctor_get(v___x_1124_, 0);
v_isSharedCheck_1143_ = !lean_is_exclusive(v___x_1124_);
if (v_isSharedCheck_1143_ == 0)
{
v___x_1138_ = v___x_1124_;
v_isShared_1139_ = v_isSharedCheck_1143_;
goto v_resetjp_1137_;
}
else
{
lean_inc(v_a_1136_);
lean_dec(v___x_1124_);
v___x_1138_ = lean_box(0);
v_isShared_1139_ = v_isSharedCheck_1143_;
goto v_resetjp_1137_;
}
v_resetjp_1137_:
{
lean_object* v___x_1141_; 
if (v_isShared_1139_ == 0)
{
v___x_1141_ = v___x_1138_;
goto v_reusejp_1140_;
}
else
{
lean_object* v_reuseFailAlloc_1142_; 
v_reuseFailAlloc_1142_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1142_, 0, v_a_1136_);
v___x_1141_ = v_reuseFailAlloc_1142_;
goto v_reusejp_1140_;
}
v_reusejp_1140_:
{
return v___x_1141_;
}
}
}
}
v_reusejp_1145_:
{
lean_object* v___x_1147_; lean_object* v___x_1148_; lean_object* v___x_1149_; lean_object* v___x_1150_; lean_object* v___x_1151_; lean_object* v___x_1152_; lean_object* v___x_1153_; lean_object* v___x_1154_; lean_object* v___x_1155_; lean_object* v___x_1156_; lean_object* v___x_1157_; lean_object* v___x_1158_; lean_object* v___x_1159_; lean_object* v_keyedConfig_1160_; uint8_t v_trackZetaDelta_1161_; lean_object* v_zetaDeltaSet_1162_; lean_object* v_lctx_1163_; lean_object* v_localInstances_1164_; lean_object* v_defEqCtx_x3f_1165_; lean_object* v_synthPendingDepth_1166_; lean_object* v_customCanUnfoldPredicate_x3f_1167_; uint8_t v_univApprox_1168_; uint8_t v_inTypeClassResolution_1169_; uint8_t v_cacheInferType_1170_; lean_object* v___x_1171_; lean_object* v___x_1172_; lean_object* v___x_1173_; lean_object* v___x_1174_; lean_object* v___x_1175_; lean_object* v___x_1176_; lean_object* v___x_1177_; lean_object* v___x_1178_; lean_object* v___x_1179_; lean_object* v___x_1180_; lean_object* v___x_1181_; lean_object* v___x_1182_; lean_object* v___f_1183_; uint8_t v___x_1184_; lean_object* v___x_1185_; lean_object* v___x_1186_; lean_object* v___x_1187_; 
lean_inc_ref_n(v___x_1146_, 3);
v___x_1147_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1147_, 0, v___x_1107_);
lean_ctor_set(v___x_1147_, 1, v___x_1146_);
lean_inc_n(v_u_1086_, 2);
v___x_1148_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1148_, 0, v_u_1086_);
lean_ctor_set(v___x_1148_, 1, v___x_1147_);
v___x_1149_ = l_Lean_Expr_const___override(v___x_1144_, v___x_1148_);
lean_inc_ref_n(v_00_u03b1_1087_, 6);
v___x_1150_ = l_Lean_Expr_app___override(v___x_1149_, v_00_u03b1_1087_);
v___x_1151_ = l_Lean_Expr_app___override(v___x_1150_, v___x_1109_);
v___x_1152_ = l_Lean_Expr_app___override(v___x_1151_, v_00_u03b1_1087_);
v___x_1153_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__7));
v___x_1154_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1154_, 0, v_u_1086_);
lean_ctor_set(v___x_1154_, 1, v___x_1110_);
v___x_1155_ = l_Lean_Expr_const___override(v___x_1153_, v___x_1154_);
v___x_1156_ = l_Lean_Expr_app___override(v___x_1155_, v_00_u03b1_1087_);
v___x_1157_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__10));
v___x_1158_ = l_Lean_Expr_const___override(v___x_1157_, v___x_1146_);
v___x_1159_ = l_Lean_Expr_app___override(v___x_1158_, v_00_u03b1_1087_);
v_keyedConfig_1160_ = lean_ctor_get(v___y_1089_, 0);
v_trackZetaDelta_1161_ = lean_ctor_get_uint8(v___y_1089_, sizeof(void*)*7);
v_zetaDeltaSet_1162_ = lean_ctor_get(v___y_1089_, 1);
v_lctx_1163_ = lean_ctor_get(v___y_1089_, 2);
v_localInstances_1164_ = lean_ctor_get(v___y_1089_, 3);
v_defEqCtx_x3f_1165_ = lean_ctor_get(v___y_1089_, 4);
v_synthPendingDepth_1166_ = lean_ctor_get(v___y_1089_, 5);
v_customCanUnfoldPredicate_x3f_1167_ = lean_ctor_get(v___y_1089_, 6);
v_univApprox_1168_ = lean_ctor_get_uint8(v___y_1089_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1169_ = lean_ctor_get_uint8(v___y_1089_, sizeof(void*)*7 + 2);
v_cacheInferType_1170_ = lean_ctor_get_uint8(v___y_1089_, sizeof(void*)*7 + 3);
v___x_1171_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__13));
v___x_1172_ = l_Lean_Expr_const___override(v___x_1171_, v___x_1146_);
v___x_1173_ = l_Lean_Expr_app___override(v___x_1172_, v_00_u03b1_1087_);
v___x_1174_ = l_Lean_Expr_app___override(v___x_1156_, v___x_1109_);
v___x_1175_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__16));
v___x_1176_ = l_Lean_Expr_const___override(v___x_1175_, v___x_1146_);
v___x_1177_ = l_Lean_Expr_app___override(v___x_1176_, v_00_u03b1_1087_);
lean_inc(v_a_1119_);
v___x_1178_ = l_Lean_Expr_app___override(v___x_1177_, v_a_1119_);
v___x_1179_ = l_Lean_Expr_app___override(v___x_1173_, v___x_1178_);
v___x_1180_ = l_Lean_Expr_app___override(v___x_1159_, v___x_1179_);
v___x_1181_ = l_Lean_Expr_app___override(v___x_1174_, v___x_1180_);
v___x_1182_ = l_Lean_Expr_app___override(v___x_1152_, v___x_1181_);
lean_inc_ref(v_fn_1105_);
v___f_1183_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__0___boxed), 7, 2);
lean_closure_set(v___f_1183_, 0, v_fn_1105_);
lean_closure_set(v___f_1183_, 1, v___x_1182_);
v___x_1184_ = 1;
lean_inc_ref(v_keyedConfig_1160_);
v___x_1185_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1184_, v_keyedConfig_1160_);
lean_inc(v_customCanUnfoldPredicate_x3f_1167_);
lean_inc(v_synthPendingDepth_1166_);
lean_inc(v_defEqCtx_x3f_1165_);
lean_inc_ref(v_localInstances_1164_);
lean_inc_ref(v_lctx_1163_);
lean_inc(v_zetaDeltaSet_1162_);
v___x_1186_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_1186_, 0, v___x_1185_);
lean_ctor_set(v___x_1186_, 1, v_zetaDeltaSet_1162_);
lean_ctor_set(v___x_1186_, 2, v_lctx_1163_);
lean_ctor_set(v___x_1186_, 3, v_localInstances_1164_);
lean_ctor_set(v___x_1186_, 4, v_defEqCtx_x3f_1165_);
lean_ctor_set(v___x_1186_, 5, v_synthPendingDepth_1166_);
lean_ctor_set(v___x_1186_, 6, v_customCanUnfoldPredicate_x3f_1167_);
lean_ctor_set_uint8(v___x_1186_, sizeof(void*)*7, v_trackZetaDelta_1161_);
lean_ctor_set_uint8(v___x_1186_, sizeof(void*)*7 + 1, v_univApprox_1168_);
lean_ctor_set_uint8(v___x_1186_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1169_);
lean_ctor_set_uint8(v___x_1186_, sizeof(void*)*7 + 3, v_cacheInferType_1170_);
v___x_1187_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalPow_spec__1___redArg(v___f_1183_, v___x_1120_, v___x_1186_, v___y_1090_, v___y_1091_, v___y_1092_);
lean_dec_ref_known(v___x_1186_, 7);
if (lean_obj_tag(v___x_1187_) == 0)
{
lean_object* v_a_1188_; uint8_t v___x_1189_; 
v_a_1188_ = lean_ctor_get(v___x_1187_, 0);
lean_inc(v_a_1188_);
lean_dec_ref_known(v___x_1187_, 1);
v___x_1189_ = lean_unbox(v_a_1188_);
lean_dec(v_a_1188_);
if (v___x_1189_ == 0)
{
lean_object* v___x_1190_; lean_object* v___x_1191_; lean_object* v_a_1192_; lean_object* v___x_1194_; uint8_t v_isShared_1195_; uint8_t v_isSharedCheck_1199_; 
lean_dec(v_a_1122_);
lean_dec(v_a_1119_);
lean_dec(v_snd_1114_);
lean_dec(v_fst_1113_);
lean_dec_ref(v_arg_1106_);
lean_dec_ref(v_fn_1105_);
lean_dec_ref(v_arg_1104_);
lean_dec_ref(v_e_1088_);
lean_dec_ref(v_00_u03b1_1087_);
lean_dec(v_u_1086_);
v___x_1190_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__1);
v___x_1191_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalPow_spec__0___redArg(v___x_1190_, v___y_1089_, v___y_1090_, v___y_1091_, v___y_1092_);
v_a_1192_ = lean_ctor_get(v___x_1191_, 0);
v_isSharedCheck_1199_ = !lean_is_exclusive(v___x_1191_);
if (v_isSharedCheck_1199_ == 0)
{
v___x_1194_ = v___x_1191_;
v_isShared_1195_ = v_isSharedCheck_1199_;
goto v_resetjp_1193_;
}
else
{
lean_inc(v_a_1192_);
lean_dec(v___x_1191_);
v___x_1194_ = lean_box(0);
v_isShared_1195_ = v_isSharedCheck_1199_;
goto v_resetjp_1193_;
}
v_resetjp_1193_:
{
lean_object* v___x_1197_; 
if (v_isShared_1195_ == 0)
{
v___x_1197_ = v___x_1194_;
goto v_reusejp_1196_;
}
else
{
lean_object* v_reuseFailAlloc_1198_; 
v_reuseFailAlloc_1198_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1198_, 0, v_a_1192_);
v___x_1197_ = v_reuseFailAlloc_1198_;
goto v_reusejp_1196_;
}
v_reusejp_1196_:
{
return v___x_1197_;
}
}
}
else
{
goto v___jp_1123_;
}
}
else
{
lean_object* v_a_1200_; lean_object* v___x_1202_; uint8_t v_isShared_1203_; uint8_t v_isSharedCheck_1207_; 
lean_dec(v_a_1122_);
lean_dec(v_a_1119_);
lean_dec(v_snd_1114_);
lean_dec(v_fst_1113_);
lean_dec_ref(v_arg_1106_);
lean_dec_ref(v_fn_1105_);
lean_dec_ref(v_arg_1104_);
lean_dec_ref(v_e_1088_);
lean_dec_ref(v_00_u03b1_1087_);
lean_dec(v_u_1086_);
v_a_1200_ = lean_ctor_get(v___x_1187_, 0);
v_isSharedCheck_1207_ = !lean_is_exclusive(v___x_1187_);
if (v_isSharedCheck_1207_ == 0)
{
v___x_1202_ = v___x_1187_;
v_isShared_1203_ = v_isSharedCheck_1207_;
goto v_resetjp_1201_;
}
else
{
lean_inc(v_a_1200_);
lean_dec(v___x_1187_);
v___x_1202_ = lean_box(0);
v_isShared_1203_ = v_isSharedCheck_1207_;
goto v_resetjp_1201_;
}
v_resetjp_1201_:
{
lean_object* v___x_1205_; 
if (v_isShared_1203_ == 0)
{
v___x_1205_ = v___x_1202_;
goto v_reusejp_1204_;
}
else
{
lean_object* v_reuseFailAlloc_1206_; 
v_reuseFailAlloc_1206_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1206_, 0, v_a_1200_);
v___x_1205_ = v_reuseFailAlloc_1206_;
goto v_reusejp_1204_;
}
v_reusejp_1204_:
{
return v___x_1205_;
}
}
}
}
}
else
{
lean_dec(v_a_1119_);
lean_del_object(v___x_1116_);
lean_dec(v_snd_1114_);
lean_dec(v_fst_1113_);
lean_dec_ref(v_arg_1106_);
lean_dec_ref(v_fn_1105_);
lean_dec_ref(v_arg_1104_);
lean_dec_ref(v_e_1088_);
lean_dec_ref(v_00_u03b1_1087_);
lean_dec(v_u_1086_);
return v___x_1121_;
}
}
else
{
lean_object* v_a_1209_; lean_object* v___x_1211_; uint8_t v_isShared_1212_; uint8_t v_isSharedCheck_1216_; 
lean_del_object(v___x_1116_);
lean_dec(v_snd_1114_);
lean_dec(v_fst_1113_);
lean_dec_ref(v_arg_1106_);
lean_dec_ref(v_fn_1105_);
lean_dec_ref(v_arg_1104_);
lean_dec_ref(v_e_1088_);
lean_dec_ref(v_00_u03b1_1087_);
lean_dec(v_u_1086_);
v_a_1209_ = lean_ctor_get(v___x_1118_, 0);
v_isSharedCheck_1216_ = !lean_is_exclusive(v___x_1118_);
if (v_isSharedCheck_1216_ == 0)
{
v___x_1211_ = v___x_1118_;
v_isShared_1212_ = v_isSharedCheck_1216_;
goto v_resetjp_1210_;
}
else
{
lean_inc(v_a_1209_);
lean_dec(v___x_1118_);
v___x_1211_ = lean_box(0);
v_isShared_1212_ = v_isSharedCheck_1216_;
goto v_resetjp_1210_;
}
v_resetjp_1210_:
{
lean_object* v___x_1214_; 
if (v_isShared_1212_ == 0)
{
v___x_1214_ = v___x_1211_;
goto v_reusejp_1213_;
}
else
{
lean_object* v_reuseFailAlloc_1215_; 
v_reuseFailAlloc_1215_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1215_, 0, v_a_1209_);
v___x_1214_ = v_reuseFailAlloc_1215_;
goto v_reusejp_1213_;
}
v_reusejp_1213_:
{
return v___x_1214_;
}
}
}
}
}
else
{
lean_object* v_a_1218_; lean_object* v___x_1220_; uint8_t v_isShared_1221_; uint8_t v_isSharedCheck_1225_; 
lean_dec_ref(v_arg_1106_);
lean_dec_ref(v_fn_1105_);
lean_dec_ref(v_arg_1104_);
lean_dec_ref(v_e_1088_);
lean_dec_ref(v_00_u03b1_1087_);
lean_dec(v_u_1086_);
v_a_1218_ = lean_ctor_get(v___x_1111_, 0);
v_isSharedCheck_1225_ = !lean_is_exclusive(v___x_1111_);
if (v_isSharedCheck_1225_ == 0)
{
v___x_1220_ = v___x_1111_;
v_isShared_1221_ = v_isSharedCheck_1225_;
goto v_resetjp_1219_;
}
else
{
lean_inc(v_a_1218_);
lean_dec(v___x_1111_);
v___x_1220_ = lean_box(0);
v_isShared_1221_ = v_isSharedCheck_1225_;
goto v_resetjp_1219_;
}
v_resetjp_1219_:
{
lean_object* v___x_1223_; 
if (v_isShared_1221_ == 0)
{
v___x_1223_ = v___x_1220_;
goto v_reusejp_1222_;
}
else
{
lean_object* v_reuseFailAlloc_1224_; 
v_reuseFailAlloc_1224_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1224_, 0, v_a_1218_);
v___x_1223_ = v_reuseFailAlloc_1224_;
goto v_reusejp_1222_;
}
v_reusejp_1222_:
{
return v___x_1223_;
}
}
}
}
else
{
lean_dec_ref_known(v_a_1095_, 2);
lean_dec_ref(v_fn_1103_);
lean_dec_ref(v_e_1088_);
lean_dec_ref(v_00_u03b1_1087_);
lean_dec(v_u_1086_);
v___y_1097_ = v___y_1089_;
v___y_1098_ = v___y_1090_;
v___y_1099_ = v___y_1091_;
v___y_1100_ = v___y_1092_;
goto v___jp_1096_;
}
}
else
{
lean_dec(v_a_1095_);
lean_dec_ref(v_e_1088_);
lean_dec_ref(v_00_u03b1_1087_);
lean_dec(v_u_1086_);
v___y_1097_ = v___y_1089_;
v___y_1098_ = v___y_1090_;
v___y_1099_ = v___y_1091_;
v___y_1100_ = v___y_1092_;
goto v___jp_1096_;
}
v___jp_1096_:
{
lean_object* v___x_1101_; lean_object* v___x_1102_; 
v___x_1101_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__1);
v___x_1102_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalPow_spec__0___redArg(v___x_1101_, v___y_1097_, v___y_1098_, v___y_1099_, v___y_1100_);
return v___x_1102_;
}
}
else
{
lean_object* v_a_1226_; lean_object* v___x_1228_; uint8_t v_isShared_1229_; uint8_t v_isSharedCheck_1233_; 
lean_dec_ref(v_e_1088_);
lean_dec_ref(v_00_u03b1_1087_);
lean_dec(v_u_1086_);
v_a_1226_ = lean_ctor_get(v___x_1094_, 0);
v_isSharedCheck_1233_ = !lean_is_exclusive(v___x_1094_);
if (v_isSharedCheck_1233_ == 0)
{
v___x_1228_ = v___x_1094_;
v_isShared_1229_ = v_isSharedCheck_1233_;
goto v_resetjp_1227_;
}
else
{
lean_inc(v_a_1226_);
lean_dec(v___x_1094_);
v___x_1228_ = lean_box(0);
v_isShared_1229_ = v_isSharedCheck_1233_;
goto v_resetjp_1227_;
}
v_resetjp_1227_:
{
lean_object* v___x_1231_; 
if (v_isShared_1229_ == 0)
{
v___x_1231_ = v___x_1228_;
goto v_reusejp_1230_;
}
else
{
lean_object* v_reuseFailAlloc_1232_; 
v_reuseFailAlloc_1232_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1232_, 0, v_a_1226_);
v___x_1231_ = v_reuseFailAlloc_1232_;
goto v_reusejp_1230_;
}
v_reusejp_1230_:
{
return v___x_1231_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___boxed(lean_object* v_u_1234_, lean_object* v_00_u03b1_1235_, lean_object* v_e_1236_, lean_object* v___y_1237_, lean_object* v___y_1238_, lean_object* v___y_1239_, lean_object* v___y_1240_, lean_object* v___y_1241_){
_start:
{
lean_object* v_res_1242_; 
v_res_1242_ = lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1(v_u_1234_, v_00_u03b1_1235_, v_e_1236_, v___y_1237_, v___y_1238_, v___y_1239_, v___y_1240_);
lean_dec(v___y_1240_);
lean_dec_ref(v___y_1239_);
lean_dec(v___y_1238_);
lean_dec_ref(v___y_1237_);
return v_res_1242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalPow_spec__0(lean_object* v_00_u03b1_1255_, lean_object* v_msg_1256_, lean_object* v___y_1257_, lean_object* v___y_1258_, lean_object* v___y_1259_, lean_object* v___y_1260_){
_start:
{
lean_object* v___x_1262_; 
v___x_1262_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalPow_spec__0___redArg(v_msg_1256_, v___y_1257_, v___y_1258_, v___y_1259_, v___y_1260_);
return v___x_1262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalPow_spec__0___boxed(lean_object* v_00_u03b1_1263_, lean_object* v_msg_1264_, lean_object* v___y_1265_, lean_object* v___y_1266_, lean_object* v___y_1267_, lean_object* v___y_1268_, lean_object* v___y_1269_){
_start:
{
lean_object* v_res_1270_; 
v_res_1270_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalPow_spec__0(v_00_u03b1_1263_, v_msg_1264_, v___y_1265_, v___y_1266_, v___y_1267_, v___y_1268_);
lean_dec(v___y_1268_);
lean_dec_ref(v___y_1267_);
lean_dec(v___y_1266_);
lean_dec_ref(v___y_1265_);
return v_res_1270_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__3(void){
_start:
{
lean_object* v___x_1276_; lean_object* v___x_1277_; lean_object* v___x_1278_; 
v___x_1276_ = lean_box(0);
v___x_1277_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__2));
v___x_1278_ = l_Lean_Expr_const___override(v___x_1277_, v___x_1276_);
return v___x_1278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0(lean_object* v_u_1375_, lean_object* v_00_u03b1_1376_, lean_object* v_e_1377_, lean_object* v___y_1378_, lean_object* v___y_1379_, lean_object* v___y_1380_, lean_object* v___y_1381_){
_start:
{
lean_object* v___y_1384_; lean_object* v___y_1385_; lean_object* v___y_1386_; lean_object* v___y_1387_; lean_object* v___x_1390_; 
lean_inc_ref(v_e_1377_);
v___x_1390_ = l_Lean_Meta_whnfR(v_e_1377_, v___y_1378_, v___y_1379_, v___y_1380_, v___y_1381_);
if (lean_obj_tag(v___x_1390_) == 0)
{
lean_object* v_a_1391_; lean_object* v___y_1393_; lean_object* v___y_1394_; lean_object* v___y_1395_; lean_object* v___y_1396_; 
v_a_1391_ = lean_ctor_get(v___x_1390_, 0);
lean_inc(v_a_1391_);
lean_dec_ref_known(v___x_1390_, 1);
if (lean_obj_tag(v_a_1391_) == 5)
{
lean_object* v_fn_1399_; 
v_fn_1399_ = lean_ctor_get(v_a_1391_, 0);
lean_inc_ref(v_fn_1399_);
if (lean_obj_tag(v_fn_1399_) == 5)
{
lean_object* v_arg_1400_; lean_object* v_arg_1401_; lean_object* v___x_1402_; lean_object* v___x_1403_; lean_object* v___x_1404_; lean_object* v___x_1405_; lean_object* v___x_1406_; lean_object* v___x_1407_; lean_object* v___x_1408_; 
v_arg_1400_ = lean_ctor_get(v_a_1391_, 1);
lean_inc_ref(v_arg_1400_);
lean_dec_ref_known(v_a_1391_, 2);
v_arg_1401_ = lean_ctor_get(v_fn_1399_, 1);
lean_inc_ref(v_arg_1401_);
lean_dec_ref_known(v_fn_1399_, 2);
lean_inc_n(v_u_1375_, 2);
v___x_1402_ = l_Lean_Level_succ___override(v_u_1375_);
v___x_1403_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__1));
v___x_1404_ = lean_box(0);
v___x_1405_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1405_, 0, v_u_1375_);
lean_ctor_set(v___x_1405_, 1, v___x_1404_);
lean_inc_ref(v___x_1405_);
v___x_1406_ = l_Lean_Expr_const___override(v___x_1403_, v___x_1405_);
lean_inc_ref(v_00_u03b1_1376_);
v___x_1407_ = l_Lean_Expr_app___override(v___x_1406_, v_00_u03b1_1376_);
v___x_1408_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_1407_, v___y_1378_, v___y_1379_, v___y_1380_, v___y_1381_);
if (lean_obj_tag(v___x_1408_) == 0)
{
lean_object* v_a_1409_; lean_object* v___x_1410_; lean_object* v___x_1411_; uint8_t v___x_1412_; lean_object* v___x_1413_; 
v_a_1409_ = lean_ctor_get(v___x_1408_, 0);
lean_inc(v_a_1409_);
lean_dec_ref_known(v___x_1408_, 1);
v___x_1410_ = lean_box(0);
v___x_1411_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__3, &lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__3_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__3);
v___x_1412_ = 0;
lean_inc_ref(v_arg_1400_);
v___x_1413_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v___x_1410_, v___x_1411_, v_arg_1400_, v___x_1412_, v___y_1378_, v___y_1379_, v___y_1380_, v___y_1381_);
if (lean_obj_tag(v___x_1413_) == 0)
{
lean_object* v_a_1414_; 
v_a_1414_ = lean_ctor_get(v___x_1413_, 0);
lean_inc(v_a_1414_);
lean_dec_ref_known(v___x_1413_, 1);
switch(lean_obj_tag(v_a_1414_))
{
case 0:
{
lean_dec_ref_known(v_a_1414_, 1);
lean_dec(v_a_1409_);
lean_dec_ref_known(v___x_1405_, 2);
lean_dec(v___x_1402_);
lean_dec_ref(v_arg_1401_);
lean_dec_ref(v_arg_1400_);
lean_dec_ref(v_e_1377_);
lean_dec_ref(v_00_u03b1_1376_);
lean_dec(v_u_1375_);
v___y_1384_ = v___y_1378_;
v___y_1385_ = v___y_1379_;
v___y_1386_ = v___y_1380_;
v___y_1387_ = v___y_1381_;
goto v___jp_1383_;
}
case 1:
{
lean_object* v_lit_1415_; lean_object* v_proof_1416_; lean_object* v___x_1417_; lean_object* v___x_1418_; lean_object* v___x_1419_; lean_object* v___x_1420_; lean_object* v___x_1421_; lean_object* v___x_1422_; lean_object* v___x_1423_; lean_object* v___x_1424_; lean_object* v___x_1425_; lean_object* v___x_1426_; lean_object* v___x_1427_; lean_object* v___x_1428_; lean_object* v___x_1429_; lean_object* v___x_1430_; lean_object* v___x_1431_; lean_object* v___x_1432_; lean_object* v___x_1433_; lean_object* v___x_1434_; lean_object* v___x_1435_; lean_object* v___x_1436_; lean_object* v___x_1437_; lean_object* v___x_1438_; lean_object* v___x_1439_; lean_object* v___x_1440_; lean_object* v___x_1441_; lean_object* v___x_1442_; lean_object* v___x_1443_; lean_object* v___x_1444_; lean_object* v___x_1445_; lean_object* v___x_1446_; lean_object* v___x_1447_; lean_object* v___x_1448_; lean_object* v___x_1449_; lean_object* v___x_1450_; 
v_lit_1415_ = lean_ctor_get(v_a_1414_, 1);
lean_inc_ref(v_lit_1415_);
v_proof_1416_ = lean_ctor_get(v_a_1414_, 2);
lean_inc_ref(v_proof_1416_);
lean_dec_ref_known(v_a_1414_, 3);
v___x_1417_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__5));
lean_inc_ref_n(v___x_1405_, 5);
v___x_1418_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1418_, 0, v___x_1410_);
lean_ctor_set(v___x_1418_, 1, v___x_1405_);
lean_inc_n(v_u_1375_, 2);
v___x_1419_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1419_, 0, v_u_1375_);
lean_ctor_set(v___x_1419_, 1, v___x_1418_);
v___x_1420_ = l_Lean_Expr_const___override(v___x_1417_, v___x_1419_);
lean_inc_ref_n(v_00_u03b1_1376_, 8);
v___x_1421_ = l_Lean_Expr_app___override(v___x_1420_, v_00_u03b1_1376_);
lean_inc_ref(v___x_1421_);
v___x_1422_ = l_Lean_Expr_app___override(v___x_1421_, v___x_1411_);
v___x_1423_ = l_Lean_Expr_app___override(v___x_1422_, v_00_u03b1_1376_);
v___x_1424_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__7));
v___x_1425_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__2));
v___x_1426_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1426_, 0, v_u_1375_);
lean_ctor_set(v___x_1426_, 1, v___x_1425_);
v___x_1427_ = l_Lean_Expr_const___override(v___x_1424_, v___x_1426_);
v___x_1428_ = l_Lean_Expr_app___override(v___x_1427_, v_00_u03b1_1376_);
lean_inc_ref(v___x_1428_);
v___x_1429_ = l_Lean_Expr_app___override(v___x_1428_, v___x_1411_);
v___x_1430_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__5));
v___x_1431_ = l_Lean_Expr_const___override(v___x_1430_, v___x_1405_);
v___x_1432_ = l_Lean_Expr_app___override(v___x_1431_, v_00_u03b1_1376_);
v___x_1433_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__8));
v___x_1434_ = l_Lean_Expr_const___override(v___x_1433_, v___x_1405_);
v___x_1435_ = l_Lean_Expr_app___override(v___x_1434_, v_00_u03b1_1376_);
v___x_1436_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__11));
v___x_1437_ = l_Lean_Expr_const___override(v___x_1436_, v___x_1405_);
v___x_1438_ = l_Lean_Expr_app___override(v___x_1437_, v_00_u03b1_1376_);
v___x_1439_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__13));
v___x_1440_ = l_Lean_Expr_const___override(v___x_1439_, v___x_1405_);
v___x_1441_ = l_Lean_Expr_app___override(v___x_1440_, v_00_u03b1_1376_);
lean_inc(v_a_1409_);
v___x_1442_ = l_Lean_Expr_app___override(v___x_1441_, v_a_1409_);
v___x_1443_ = l_Lean_Expr_app___override(v___x_1438_, v___x_1442_);
v___x_1444_ = l_Lean_Expr_app___override(v___x_1435_, v___x_1443_);
v___x_1445_ = l_Lean_Expr_app___override(v___x_1432_, v___x_1444_);
v___x_1446_ = l_Lean_Expr_app___override(v___x_1429_, v___x_1445_);
v___x_1447_ = l_Lean_Expr_app___override(v___x_1423_, v___x_1446_);
lean_inc_ref(v_arg_1401_);
v___x_1448_ = l_Lean_Expr_app___override(v___x_1447_, v_arg_1401_);
lean_inc_ref(v_arg_1400_);
v___x_1449_ = l_Lean_Expr_app___override(v___x_1448_, v_arg_1400_);
lean_inc_ref(v_e_1377_);
v___x_1450_ = lp_Qq_Qq_QuotedDefEq_check___redArg(v___x_1402_, v_00_u03b1_1376_, v_e_1377_, v___x_1449_, v___y_1378_, v___y_1379_, v___y_1380_, v___y_1381_);
if (lean_obj_tag(v___x_1450_) == 0)
{
lean_object* v___x_1451_; lean_object* v___x_1452_; lean_object* v___x_1453_; lean_object* v___x_1454_; lean_object* v___x_1455_; lean_object* v___x_1456_; lean_object* v___x_1457_; lean_object* v___x_1458_; lean_object* v___x_1459_; lean_object* v___x_1460_; lean_object* v___x_1461_; lean_object* v___x_1462_; lean_object* v___x_1463_; lean_object* v___x_1464_; lean_object* v___x_1465_; lean_object* v___x_1466_; lean_object* v___x_1467_; lean_object* v___x_1468_; lean_object* v___x_1469_; lean_object* v___x_1470_; lean_object* v___x_1471_; lean_object* v___x_1472_; lean_object* v___x_1473_; lean_object* v___x_1474_; lean_object* v___x_1475_; 
lean_dec_ref_known(v___x_1450_, 1);
v___x_1451_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__7, &lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__7);
v___x_1452_ = l_Lean_Expr_app___override(v___x_1421_, v___x_1451_);
lean_inc_ref_n(v_00_u03b1_1376_, 6);
v___x_1453_ = l_Lean_Expr_app___override(v___x_1452_, v_00_u03b1_1376_);
v___x_1454_ = l_Lean_Expr_app___override(v___x_1428_, v___x_1451_);
v___x_1455_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__10));
lean_inc_ref_n(v___x_1405_, 4);
v___x_1456_ = l_Lean_Expr_const___override(v___x_1455_, v___x_1405_);
v___x_1457_ = l_Lean_Expr_app___override(v___x_1456_, v_00_u03b1_1376_);
v___x_1458_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__13));
v___x_1459_ = l_Lean_Expr_const___override(v___x_1458_, v___x_1405_);
v___x_1460_ = l_Lean_Expr_app___override(v___x_1459_, v_00_u03b1_1376_);
v___x_1461_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__16));
v___x_1462_ = l_Lean_Expr_const___override(v___x_1461_, v___x_1405_);
v___x_1463_ = l_Lean_Expr_app___override(v___x_1462_, v_00_u03b1_1376_);
v___x_1464_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__15));
v___x_1465_ = l_Lean_Expr_const___override(v___x_1464_, v___x_1405_);
v___x_1466_ = l_Lean_Expr_app___override(v___x_1465_, v_00_u03b1_1376_);
lean_inc(v_a_1409_);
v___x_1467_ = l_Lean_Expr_app___override(v___x_1466_, v_a_1409_);
v___x_1468_ = l_Lean_Expr_app___override(v___x_1463_, v___x_1467_);
v___x_1469_ = l_Lean_Expr_app___override(v___x_1460_, v___x_1468_);
v___x_1470_ = l_Lean_Expr_app___override(v___x_1457_, v___x_1469_);
v___x_1471_ = l_Lean_Expr_app___override(v___x_1454_, v___x_1470_);
v___x_1472_ = l_Lean_Expr_app___override(v___x_1453_, v___x_1471_);
lean_inc_ref(v_arg_1401_);
v___x_1473_ = l_Lean_Expr_app___override(v___x_1472_, v_arg_1401_);
lean_inc_ref(v_lit_1415_);
v___x_1474_ = l_Lean_Expr_app___override(v___x_1473_, v_lit_1415_);
lean_inc(v_u_1375_);
v___x_1475_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v_u_1375_, v_00_u03b1_1376_, v___x_1474_, v___x_1412_, v___y_1378_, v___y_1379_, v___y_1380_, v___y_1381_);
if (lean_obj_tag(v___x_1475_) == 0)
{
lean_object* v_a_1476_; lean_object* v___x_1478_; uint8_t v_isShared_1479_; uint8_t v_isSharedCheck_1591_; 
v_a_1476_ = lean_ctor_get(v___x_1475_, 0);
v_isSharedCheck_1591_ = !lean_is_exclusive(v___x_1475_);
if (v_isSharedCheck_1591_ == 0)
{
v___x_1478_ = v___x_1475_;
v_isShared_1479_ = v_isSharedCheck_1591_;
goto v_resetjp_1477_;
}
else
{
lean_inc(v_a_1476_);
lean_dec(v___x_1475_);
v___x_1478_ = lean_box(0);
v_isShared_1479_ = v_isSharedCheck_1591_;
goto v_resetjp_1477_;
}
v_resetjp_1477_:
{
switch(lean_obj_tag(v_a_1476_))
{
case 0:
{
lean_dec_ref_known(v_a_1476_, 1);
lean_del_object(v___x_1478_);
lean_dec_ref(v_proof_1416_);
lean_dec_ref(v_lit_1415_);
lean_dec(v_a_1409_);
lean_dec_ref_known(v___x_1405_, 2);
lean_dec_ref(v_arg_1401_);
lean_dec_ref(v_arg_1400_);
lean_dec_ref(v_e_1377_);
lean_dec_ref(v_00_u03b1_1376_);
lean_dec(v_u_1375_);
v___y_1384_ = v___y_1378_;
v___y_1385_ = v___y_1379_;
v___y_1386_ = v___y_1380_;
v___y_1387_ = v___y_1381_;
goto v___jp_1383_;
}
case 1:
{
lean_object* v_inst_1480_; lean_object* v_lit_1481_; lean_object* v_proof_1482_; lean_object* v___x_1484_; uint8_t v_isShared_1485_; uint8_t v_isSharedCheck_1502_; 
lean_dec_ref(v_e_1377_);
lean_dec(v_u_1375_);
v_inst_1480_ = lean_ctor_get(v_a_1476_, 0);
v_lit_1481_ = lean_ctor_get(v_a_1476_, 1);
v_proof_1482_ = lean_ctor_get(v_a_1476_, 2);
v_isSharedCheck_1502_ = !lean_is_exclusive(v_a_1476_);
if (v_isSharedCheck_1502_ == 0)
{
v___x_1484_ = v_a_1476_;
v_isShared_1485_ = v_isSharedCheck_1502_;
goto v_resetjp_1483_;
}
else
{
lean_inc(v_proof_1482_);
lean_inc(v_lit_1481_);
lean_inc(v_inst_1480_);
lean_dec(v_a_1476_);
v___x_1484_ = lean_box(0);
v_isShared_1485_ = v_isSharedCheck_1502_;
goto v_resetjp_1483_;
}
v_resetjp_1483_:
{
lean_object* v___x_1486_; lean_object* v___x_1487_; lean_object* v___x_1488_; lean_object* v___x_1489_; lean_object* v___x_1490_; lean_object* v___x_1491_; lean_object* v___x_1492_; lean_object* v___x_1493_; lean_object* v___x_1494_; lean_object* v___x_1495_; lean_object* v___x_1497_; 
v___x_1486_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__17));
v___x_1487_ = l_Lean_Expr_const___override(v___x_1486_, v___x_1405_);
v___x_1488_ = l_Lean_Expr_app___override(v___x_1487_, v_00_u03b1_1376_);
v___x_1489_ = l_Lean_Expr_app___override(v___x_1488_, v_a_1409_);
v___x_1490_ = l_Lean_Expr_app___override(v___x_1489_, v_arg_1401_);
v___x_1491_ = l_Lean_Expr_app___override(v___x_1490_, v_arg_1400_);
v___x_1492_ = l_Lean_Expr_app___override(v___x_1491_, v_lit_1415_);
lean_inc_ref(v_lit_1481_);
v___x_1493_ = l_Lean_Expr_app___override(v___x_1492_, v_lit_1481_);
v___x_1494_ = l_Lean_Expr_app___override(v___x_1493_, v_proof_1416_);
v___x_1495_ = l_Lean_Expr_app___override(v___x_1494_, v_proof_1482_);
if (v_isShared_1485_ == 0)
{
lean_ctor_set(v___x_1484_, 2, v___x_1495_);
v___x_1497_ = v___x_1484_;
goto v_reusejp_1496_;
}
else
{
lean_object* v_reuseFailAlloc_1501_; 
v_reuseFailAlloc_1501_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1501_, 0, v_inst_1480_);
lean_ctor_set(v_reuseFailAlloc_1501_, 1, v_lit_1481_);
lean_ctor_set(v_reuseFailAlloc_1501_, 2, v___x_1495_);
v___x_1497_ = v_reuseFailAlloc_1501_;
goto v_reusejp_1496_;
}
v_reusejp_1496_:
{
lean_object* v___x_1499_; 
if (v_isShared_1479_ == 0)
{
lean_ctor_set(v___x_1478_, 0, v___x_1497_);
v___x_1499_ = v___x_1478_;
goto v_reusejp_1498_;
}
else
{
lean_object* v_reuseFailAlloc_1500_; 
v_reuseFailAlloc_1500_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1500_, 0, v___x_1497_);
v___x_1499_ = v_reuseFailAlloc_1500_;
goto v_reusejp_1498_;
}
v_reusejp_1498_:
{
return v___x_1499_;
}
}
}
}
case 2:
{
lean_object* v_inst_1503_; lean_object* v_lit_1504_; lean_object* v_proof_1505_; lean_object* v___x_1507_; uint8_t v_isShared_1508_; uint8_t v_isSharedCheck_1542_; 
lean_del_object(v___x_1478_);
lean_dec(v_a_1409_);
lean_dec_ref(v_e_1377_);
lean_dec(v_u_1375_);
v_inst_1503_ = lean_ctor_get(v_a_1476_, 0);
v_lit_1504_ = lean_ctor_get(v_a_1476_, 1);
v_proof_1505_ = lean_ctor_get(v_a_1476_, 2);
v_isSharedCheck_1542_ = !lean_is_exclusive(v_a_1476_);
if (v_isSharedCheck_1542_ == 0)
{
v___x_1507_ = v_a_1476_;
v_isShared_1508_ = v_isSharedCheck_1542_;
goto v_resetjp_1506_;
}
else
{
lean_inc(v_proof_1505_);
lean_inc(v_lit_1504_);
lean_inc(v_inst_1503_);
lean_dec(v_a_1476_);
v___x_1507_ = lean_box(0);
v_isShared_1508_ = v_isSharedCheck_1542_;
goto v_resetjp_1506_;
}
v_resetjp_1506_:
{
lean_object* v___x_1509_; lean_object* v___x_1510_; lean_object* v___x_1511_; lean_object* v___x_1512_; 
v___x_1509_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__18));
lean_inc_ref(v___x_1405_);
v___x_1510_ = l_Lean_Expr_const___override(v___x_1509_, v___x_1405_);
lean_inc_ref(v_00_u03b1_1376_);
v___x_1511_ = l_Lean_Expr_app___override(v___x_1510_, v_00_u03b1_1376_);
v___x_1512_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_1511_, v___y_1378_, v___y_1379_, v___y_1380_, v___y_1381_);
if (lean_obj_tag(v___x_1512_) == 0)
{
lean_object* v_a_1513_; lean_object* v___x_1515_; uint8_t v_isShared_1516_; uint8_t v_isSharedCheck_1533_; 
v_a_1513_ = lean_ctor_get(v___x_1512_, 0);
v_isSharedCheck_1533_ = !lean_is_exclusive(v___x_1512_);
if (v_isSharedCheck_1533_ == 0)
{
v___x_1515_ = v___x_1512_;
v_isShared_1516_ = v_isSharedCheck_1533_;
goto v_resetjp_1514_;
}
else
{
lean_inc(v_a_1513_);
lean_dec(v___x_1512_);
v___x_1515_ = lean_box(0);
v_isShared_1516_ = v_isSharedCheck_1533_;
goto v_resetjp_1514_;
}
v_resetjp_1514_:
{
lean_object* v___x_1517_; lean_object* v___x_1518_; lean_object* v___x_1519_; lean_object* v___x_1520_; lean_object* v___x_1521_; lean_object* v___x_1522_; lean_object* v___x_1523_; lean_object* v___x_1524_; lean_object* v___x_1525_; lean_object* v___x_1526_; lean_object* v___x_1528_; 
v___x_1517_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__20));
v___x_1518_ = l_Lean_Expr_const___override(v___x_1517_, v___x_1405_);
v___x_1519_ = l_Lean_Expr_app___override(v___x_1518_, v_00_u03b1_1376_);
v___x_1520_ = l_Lean_Expr_app___override(v___x_1519_, v_a_1513_);
v___x_1521_ = l_Lean_Expr_app___override(v___x_1520_, v_arg_1401_);
v___x_1522_ = l_Lean_Expr_app___override(v___x_1521_, v_arg_1400_);
v___x_1523_ = l_Lean_Expr_app___override(v___x_1522_, v_lit_1415_);
lean_inc_ref(v_lit_1504_);
v___x_1524_ = l_Lean_Expr_app___override(v___x_1523_, v_lit_1504_);
v___x_1525_ = l_Lean_Expr_app___override(v___x_1524_, v_proof_1416_);
v___x_1526_ = l_Lean_Expr_app___override(v___x_1525_, v_proof_1505_);
if (v_isShared_1508_ == 0)
{
lean_ctor_set(v___x_1507_, 2, v___x_1526_);
v___x_1528_ = v___x_1507_;
goto v_reusejp_1527_;
}
else
{
lean_object* v_reuseFailAlloc_1532_; 
v_reuseFailAlloc_1532_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1532_, 0, v_inst_1503_);
lean_ctor_set(v_reuseFailAlloc_1532_, 1, v_lit_1504_);
lean_ctor_set(v_reuseFailAlloc_1532_, 2, v___x_1526_);
v___x_1528_ = v_reuseFailAlloc_1532_;
goto v_reusejp_1527_;
}
v_reusejp_1527_:
{
lean_object* v___x_1530_; 
if (v_isShared_1516_ == 0)
{
lean_ctor_set(v___x_1515_, 0, v___x_1528_);
v___x_1530_ = v___x_1515_;
goto v_reusejp_1529_;
}
else
{
lean_object* v_reuseFailAlloc_1531_; 
v_reuseFailAlloc_1531_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1531_, 0, v___x_1528_);
v___x_1530_ = v_reuseFailAlloc_1531_;
goto v_reusejp_1529_;
}
v_reusejp_1529_:
{
return v___x_1530_;
}
}
}
}
else
{
lean_object* v_a_1534_; lean_object* v___x_1536_; uint8_t v_isShared_1537_; uint8_t v_isSharedCheck_1541_; 
lean_del_object(v___x_1507_);
lean_dec_ref(v_proof_1505_);
lean_dec_ref(v_lit_1504_);
lean_dec_ref(v_inst_1503_);
lean_dec_ref(v_proof_1416_);
lean_dec_ref(v_lit_1415_);
lean_dec_ref_known(v___x_1405_, 2);
lean_dec_ref(v_arg_1401_);
lean_dec_ref(v_arg_1400_);
lean_dec_ref(v_00_u03b1_1376_);
v_a_1534_ = lean_ctor_get(v___x_1512_, 0);
v_isSharedCheck_1541_ = !lean_is_exclusive(v___x_1512_);
if (v_isSharedCheck_1541_ == 0)
{
v___x_1536_ = v___x_1512_;
v_isShared_1537_ = v_isSharedCheck_1541_;
goto v_resetjp_1535_;
}
else
{
lean_inc(v_a_1534_);
lean_dec(v___x_1512_);
v___x_1536_ = lean_box(0);
v_isShared_1537_ = v_isSharedCheck_1541_;
goto v_resetjp_1535_;
}
v_resetjp_1535_:
{
lean_object* v___x_1539_; 
if (v_isShared_1537_ == 0)
{
v___x_1539_ = v___x_1536_;
goto v_reusejp_1538_;
}
else
{
lean_object* v_reuseFailAlloc_1540_; 
v_reuseFailAlloc_1540_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1540_, 0, v_a_1534_);
v___x_1539_ = v_reuseFailAlloc_1540_;
goto v_reusejp_1538_;
}
v_reusejp_1538_:
{
return v___x_1539_;
}
}
}
}
}
case 3:
{
lean_object* v_inst_1543_; lean_object* v_q_1544_; lean_object* v_n_1545_; lean_object* v_d_1546_; lean_object* v_proof_1547_; lean_object* v___x_1549_; uint8_t v_isShared_1550_; uint8_t v_isSharedCheck_1568_; 
lean_dec_ref(v_e_1377_);
lean_dec(v_u_1375_);
v_inst_1543_ = lean_ctor_get(v_a_1476_, 0);
v_q_1544_ = lean_ctor_get(v_a_1476_, 1);
v_n_1545_ = lean_ctor_get(v_a_1476_, 2);
v_d_1546_ = lean_ctor_get(v_a_1476_, 3);
v_proof_1547_ = lean_ctor_get(v_a_1476_, 4);
v_isSharedCheck_1568_ = !lean_is_exclusive(v_a_1476_);
if (v_isSharedCheck_1568_ == 0)
{
v___x_1549_ = v_a_1476_;
v_isShared_1550_ = v_isSharedCheck_1568_;
goto v_resetjp_1548_;
}
else
{
lean_inc(v_proof_1547_);
lean_inc(v_d_1546_);
lean_inc(v_n_1545_);
lean_inc(v_q_1544_);
lean_inc(v_inst_1543_);
lean_dec(v_a_1476_);
v___x_1549_ = lean_box(0);
v_isShared_1550_ = v_isSharedCheck_1568_;
goto v_resetjp_1548_;
}
v_resetjp_1548_:
{
lean_object* v___x_1551_; lean_object* v___x_1552_; lean_object* v___x_1553_; lean_object* v___x_1554_; lean_object* v___x_1555_; lean_object* v___x_1556_; lean_object* v___x_1557_; lean_object* v___x_1558_; lean_object* v___x_1559_; lean_object* v___x_1560_; lean_object* v___x_1561_; lean_object* v___x_1563_; 
v___x_1551_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__22));
v___x_1552_ = l_Lean_Expr_const___override(v___x_1551_, v___x_1405_);
v___x_1553_ = l_Lean_Expr_app___override(v___x_1552_, v_00_u03b1_1376_);
v___x_1554_ = l_Lean_Expr_app___override(v___x_1553_, v_a_1409_);
v___x_1555_ = l_Lean_Expr_app___override(v___x_1554_, v_arg_1401_);
v___x_1556_ = l_Lean_Expr_app___override(v___x_1555_, v_arg_1400_);
v___x_1557_ = l_Lean_Expr_app___override(v___x_1556_, v_lit_1415_);
lean_inc_ref(v_n_1545_);
v___x_1558_ = l_Lean_Expr_app___override(v___x_1557_, v_n_1545_);
lean_inc_ref(v_d_1546_);
v___x_1559_ = l_Lean_Expr_app___override(v___x_1558_, v_d_1546_);
v___x_1560_ = l_Lean_Expr_app___override(v___x_1559_, v_proof_1416_);
v___x_1561_ = l_Lean_Expr_app___override(v___x_1560_, v_proof_1547_);
if (v_isShared_1550_ == 0)
{
lean_ctor_set(v___x_1549_, 4, v___x_1561_);
v___x_1563_ = v___x_1549_;
goto v_reusejp_1562_;
}
else
{
lean_object* v_reuseFailAlloc_1567_; 
v_reuseFailAlloc_1567_ = lean_alloc_ctor(3, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1567_, 0, v_inst_1543_);
lean_ctor_set(v_reuseFailAlloc_1567_, 1, v_q_1544_);
lean_ctor_set(v_reuseFailAlloc_1567_, 2, v_n_1545_);
lean_ctor_set(v_reuseFailAlloc_1567_, 3, v_d_1546_);
lean_ctor_set(v_reuseFailAlloc_1567_, 4, v___x_1561_);
v___x_1563_ = v_reuseFailAlloc_1567_;
goto v_reusejp_1562_;
}
v_reusejp_1562_:
{
lean_object* v___x_1565_; 
if (v_isShared_1479_ == 0)
{
lean_ctor_set(v___x_1478_, 0, v___x_1563_);
v___x_1565_ = v___x_1478_;
goto v_reusejp_1564_;
}
else
{
lean_object* v_reuseFailAlloc_1566_; 
v_reuseFailAlloc_1566_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1566_, 0, v___x_1563_);
v___x_1565_ = v_reuseFailAlloc_1566_;
goto v_reusejp_1564_;
}
v_reusejp_1564_:
{
return v___x_1565_;
}
}
}
}
default: 
{
lean_object* v_inst_1569_; lean_object* v_q_1570_; lean_object* v_n_1571_; lean_object* v_d_1572_; lean_object* v_proof_1573_; lean_object* v___x_1574_; lean_object* v___x_1575_; lean_object* v___x_1576_; lean_object* v___x_1577_; lean_object* v___x_1578_; lean_object* v___x_1579_; lean_object* v___x_1580_; lean_object* v___x_1581_; lean_object* v___x_1582_; lean_object* v___x_1583_; lean_object* v___x_1584_; lean_object* v___x_1585_; lean_object* v___x_1586_; lean_object* v___x_1587_; lean_object* v___x_1589_; 
lean_dec(v_a_1409_);
v_inst_1569_ = lean_ctor_get(v_a_1476_, 0);
lean_inc_ref_n(v_inst_1569_, 2);
v_q_1570_ = lean_ctor_get(v_a_1476_, 1);
lean_inc_ref(v_q_1570_);
v_n_1571_ = lean_ctor_get(v_a_1476_, 2);
lean_inc_ref_n(v_n_1571_, 2);
v_d_1572_ = lean_ctor_get(v_a_1476_, 3);
lean_inc_ref_n(v_d_1572_, 2);
v_proof_1573_ = lean_ctor_get(v_a_1476_, 4);
lean_inc_ref(v_proof_1573_);
lean_dec_ref_known(v_a_1476_, 5);
v___x_1574_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__8, &lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__8_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__8);
v___x_1575_ = l_Lean_Expr_app___override(v___x_1574_, v_n_1571_);
v___x_1576_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__24));
v___x_1577_ = l_Lean_Expr_const___override(v___x_1576_, v___x_1405_);
lean_inc_ref(v_00_u03b1_1376_);
v___x_1578_ = l_Lean_Expr_app___override(v___x_1577_, v_00_u03b1_1376_);
v___x_1579_ = l_Lean_Expr_app___override(v___x_1578_, v_inst_1569_);
v___x_1580_ = l_Lean_Expr_app___override(v___x_1579_, v_arg_1401_);
v___x_1581_ = l_Lean_Expr_app___override(v___x_1580_, v_arg_1400_);
v___x_1582_ = l_Lean_Expr_app___override(v___x_1581_, v_lit_1415_);
v___x_1583_ = l_Lean_Expr_app___override(v___x_1582_, v___x_1575_);
v___x_1584_ = l_Lean_Expr_app___override(v___x_1583_, v_d_1572_);
v___x_1585_ = l_Lean_Expr_app___override(v___x_1584_, v_proof_1416_);
v___x_1586_ = l_Lean_Expr_app___override(v___x_1585_, v_proof_1573_);
v___x_1587_ = lp_mathlib_Mathlib_Meta_NormNum_Result_isRat(v_u_1375_, v_00_u03b1_1376_, v_e_1377_, v_inst_1569_, v_q_1570_, v_n_1571_, v_d_1572_, v___x_1586_);
if (v_isShared_1479_ == 0)
{
lean_ctor_set(v___x_1478_, 0, v___x_1587_);
v___x_1589_ = v___x_1478_;
goto v_reusejp_1588_;
}
else
{
lean_object* v_reuseFailAlloc_1590_; 
v_reuseFailAlloc_1590_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1590_, 0, v___x_1587_);
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
}
else
{
lean_dec_ref(v_proof_1416_);
lean_dec_ref(v_lit_1415_);
lean_dec(v_a_1409_);
lean_dec_ref_known(v___x_1405_, 2);
lean_dec_ref(v_arg_1401_);
lean_dec_ref(v_arg_1400_);
lean_dec_ref(v_e_1377_);
lean_dec_ref(v_00_u03b1_1376_);
lean_dec(v_u_1375_);
return v___x_1475_;
}
}
else
{
lean_object* v_a_1592_; lean_object* v___x_1594_; uint8_t v_isShared_1595_; uint8_t v_isSharedCheck_1599_; 
lean_dec_ref(v___x_1428_);
lean_dec_ref(v___x_1421_);
lean_dec_ref(v_proof_1416_);
lean_dec_ref(v_lit_1415_);
lean_dec(v_a_1409_);
lean_dec_ref_known(v___x_1405_, 2);
lean_dec_ref(v_arg_1401_);
lean_dec_ref(v_arg_1400_);
lean_dec_ref(v_e_1377_);
lean_dec_ref(v_00_u03b1_1376_);
lean_dec(v_u_1375_);
v_a_1592_ = lean_ctor_get(v___x_1450_, 0);
v_isSharedCheck_1599_ = !lean_is_exclusive(v___x_1450_);
if (v_isSharedCheck_1599_ == 0)
{
v___x_1594_ = v___x_1450_;
v_isShared_1595_ = v_isSharedCheck_1599_;
goto v_resetjp_1593_;
}
else
{
lean_inc(v_a_1592_);
lean_dec(v___x_1450_);
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
case 2:
{
lean_object* v_lit_1600_; lean_object* v_proof_1601_; lean_object* v___x_1602_; lean_object* v___x_1603_; lean_object* v___x_1604_; lean_object* v___x_1605_; lean_object* v___x_1606_; lean_object* v___x_1607_; lean_object* v___x_1608_; lean_object* v___x_1609_; lean_object* v___x_1610_; lean_object* v___x_1611_; lean_object* v___x_1612_; lean_object* v___x_1613_; lean_object* v___x_1614_; lean_object* v___x_1615_; lean_object* v___x_1616_; lean_object* v___x_1617_; lean_object* v___x_1618_; lean_object* v___x_1619_; lean_object* v___x_1620_; lean_object* v___x_1621_; lean_object* v___x_1622_; lean_object* v___x_1623_; lean_object* v___x_1624_; lean_object* v___x_1625_; lean_object* v___x_1626_; lean_object* v___x_1627_; lean_object* v___x_1628_; lean_object* v___x_1629_; lean_object* v___x_1630_; lean_object* v___x_1631_; lean_object* v___x_1632_; lean_object* v___x_1633_; lean_object* v___x_1634_; lean_object* v___x_1635_; 
v_lit_1600_ = lean_ctor_get(v_a_1414_, 1);
lean_inc_ref(v_lit_1600_);
v_proof_1601_ = lean_ctor_get(v_a_1414_, 2);
lean_inc_ref(v_proof_1601_);
lean_dec_ref_known(v_a_1414_, 3);
v___x_1602_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__5));
lean_inc_ref_n(v___x_1405_, 5);
v___x_1603_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1603_, 0, v___x_1410_);
lean_ctor_set(v___x_1603_, 1, v___x_1405_);
lean_inc_n(v_u_1375_, 2);
v___x_1604_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1604_, 0, v_u_1375_);
lean_ctor_set(v___x_1604_, 1, v___x_1603_);
v___x_1605_ = l_Lean_Expr_const___override(v___x_1602_, v___x_1604_);
lean_inc_ref_n(v_00_u03b1_1376_, 8);
v___x_1606_ = l_Lean_Expr_app___override(v___x_1605_, v_00_u03b1_1376_);
lean_inc_ref(v___x_1606_);
v___x_1607_ = l_Lean_Expr_app___override(v___x_1606_, v___x_1411_);
v___x_1608_ = l_Lean_Expr_app___override(v___x_1607_, v_00_u03b1_1376_);
v___x_1609_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__7));
v___x_1610_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__2));
v___x_1611_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1611_, 0, v_u_1375_);
lean_ctor_set(v___x_1611_, 1, v___x_1610_);
v___x_1612_ = l_Lean_Expr_const___override(v___x_1609_, v___x_1611_);
v___x_1613_ = l_Lean_Expr_app___override(v___x_1612_, v_00_u03b1_1376_);
lean_inc_ref(v___x_1613_);
v___x_1614_ = l_Lean_Expr_app___override(v___x_1613_, v___x_1411_);
v___x_1615_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__5));
v___x_1616_ = l_Lean_Expr_const___override(v___x_1615_, v___x_1405_);
v___x_1617_ = l_Lean_Expr_app___override(v___x_1616_, v_00_u03b1_1376_);
v___x_1618_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__8));
v___x_1619_ = l_Lean_Expr_const___override(v___x_1618_, v___x_1405_);
v___x_1620_ = l_Lean_Expr_app___override(v___x_1619_, v_00_u03b1_1376_);
v___x_1621_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__11));
v___x_1622_ = l_Lean_Expr_const___override(v___x_1621_, v___x_1405_);
v___x_1623_ = l_Lean_Expr_app___override(v___x_1622_, v_00_u03b1_1376_);
v___x_1624_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__13));
v___x_1625_ = l_Lean_Expr_const___override(v___x_1624_, v___x_1405_);
v___x_1626_ = l_Lean_Expr_app___override(v___x_1625_, v_00_u03b1_1376_);
lean_inc(v_a_1409_);
v___x_1627_ = l_Lean_Expr_app___override(v___x_1626_, v_a_1409_);
lean_inc_ref(v___x_1627_);
v___x_1628_ = l_Lean_Expr_app___override(v___x_1623_, v___x_1627_);
v___x_1629_ = l_Lean_Expr_app___override(v___x_1620_, v___x_1628_);
v___x_1630_ = l_Lean_Expr_app___override(v___x_1617_, v___x_1629_);
v___x_1631_ = l_Lean_Expr_app___override(v___x_1614_, v___x_1630_);
v___x_1632_ = l_Lean_Expr_app___override(v___x_1608_, v___x_1631_);
lean_inc_ref(v_arg_1401_);
v___x_1633_ = l_Lean_Expr_app___override(v___x_1632_, v_arg_1401_);
lean_inc_ref(v_arg_1400_);
v___x_1634_ = l_Lean_Expr_app___override(v___x_1633_, v_arg_1400_);
lean_inc_ref(v_e_1377_);
v___x_1635_ = lp_Qq_Qq_QuotedDefEq_check___redArg(v___x_1402_, v_00_u03b1_1376_, v_e_1377_, v___x_1634_, v___y_1378_, v___y_1379_, v___y_1380_, v___y_1381_);
if (lean_obj_tag(v___x_1635_) == 0)
{
lean_object* v___x_1636_; lean_object* v___x_1637_; lean_object* v___x_1638_; lean_object* v___x_1639_; lean_object* v___x_1640_; lean_object* v___x_1641_; lean_object* v___x_1642_; lean_object* v___x_1643_; lean_object* v___x_1644_; lean_object* v___x_1645_; lean_object* v___x_1646_; lean_object* v___x_1647_; lean_object* v___x_1648_; lean_object* v___x_1649_; lean_object* v___x_1650_; lean_object* v___x_1651_; lean_object* v___x_1652_; lean_object* v___x_1653_; lean_object* v___x_1654_; lean_object* v___x_1655_; lean_object* v___x_1656_; lean_object* v___x_1657_; lean_object* v___x_1658_; lean_object* v___x_1659_; lean_object* v___x_1660_; lean_object* v___x_1661_; lean_object* v___x_1662_; lean_object* v___x_1663_; lean_object* v___x_1664_; lean_object* v___x_1665_; lean_object* v___x_1666_; lean_object* v___x_1667_; lean_object* v___x_1668_; lean_object* v___x_1669_; lean_object* v___x_1670_; lean_object* v___x_1671_; lean_object* v___x_1672_; lean_object* v___x_1673_; lean_object* v___x_1674_; lean_object* v___x_1675_; lean_object* v___x_1676_; lean_object* v___x_1677_; lean_object* v___x_1678_; lean_object* v___x_1679_; lean_object* v___x_1680_; lean_object* v___x_1681_; 
lean_dec_ref_known(v___x_1635_, 1);
v___x_1636_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__27));
lean_inc_ref_n(v___x_1405_, 9);
v___x_1637_ = l_Lean_Expr_const___override(v___x_1636_, v___x_1405_);
lean_inc_ref_n(v_00_u03b1_1376_, 11);
v___x_1638_ = l_Lean_Expr_app___override(v___x_1637_, v_00_u03b1_1376_);
v___x_1639_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__30));
v___x_1640_ = l_Lean_Expr_const___override(v___x_1639_, v___x_1405_);
v___x_1641_ = l_Lean_Expr_app___override(v___x_1640_, v_00_u03b1_1376_);
v___x_1642_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__33));
v___x_1643_ = l_Lean_Expr_const___override(v___x_1642_, v___x_1405_);
v___x_1644_ = l_Lean_Expr_app___override(v___x_1643_, v_00_u03b1_1376_);
v___x_1645_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__36));
v___x_1646_ = l_Lean_Expr_const___override(v___x_1645_, v___x_1405_);
v___x_1647_ = l_Lean_Expr_app___override(v___x_1646_, v_00_u03b1_1376_);
v___x_1648_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__38));
v___x_1649_ = l_Lean_Expr_const___override(v___x_1648_, v___x_1405_);
v___x_1650_ = l_Lean_Expr_app___override(v___x_1649_, v_00_u03b1_1376_);
v___x_1651_ = l_Lean_Expr_app___override(v___x_1650_, v___x_1627_);
v___x_1652_ = l_Lean_Expr_app___override(v___x_1647_, v___x_1651_);
v___x_1653_ = l_Lean_Expr_app___override(v___x_1644_, v___x_1652_);
v___x_1654_ = l_Lean_Expr_app___override(v___x_1641_, v___x_1653_);
v___x_1655_ = l_Lean_Expr_app___override(v___x_1638_, v___x_1654_);
v___x_1656_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__7, &lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Pow_0__Mathlib_Meta_NormNum_evalNatPow_go___redArg___closed__7);
v___x_1657_ = l_Lean_Expr_app___override(v___x_1606_, v___x_1656_);
v___x_1658_ = l_Lean_Expr_app___override(v___x_1657_, v_00_u03b1_1376_);
v___x_1659_ = l_Lean_Expr_app___override(v___x_1613_, v___x_1656_);
v___x_1660_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__10));
v___x_1661_ = l_Lean_Expr_const___override(v___x_1660_, v___x_1405_);
v___x_1662_ = l_Lean_Expr_app___override(v___x_1661_, v_00_u03b1_1376_);
v___x_1663_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__13));
v___x_1664_ = l_Lean_Expr_const___override(v___x_1663_, v___x_1405_);
v___x_1665_ = l_Lean_Expr_app___override(v___x_1664_, v_00_u03b1_1376_);
v___x_1666_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__16));
v___x_1667_ = l_Lean_Expr_const___override(v___x_1666_, v___x_1405_);
v___x_1668_ = l_Lean_Expr_app___override(v___x_1667_, v_00_u03b1_1376_);
v___x_1669_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__15));
v___x_1670_ = l_Lean_Expr_const___override(v___x_1669_, v___x_1405_);
v___x_1671_ = l_Lean_Expr_app___override(v___x_1670_, v_00_u03b1_1376_);
lean_inc(v_a_1409_);
v___x_1672_ = l_Lean_Expr_app___override(v___x_1671_, v_a_1409_);
v___x_1673_ = l_Lean_Expr_app___override(v___x_1668_, v___x_1672_);
v___x_1674_ = l_Lean_Expr_app___override(v___x_1665_, v___x_1673_);
v___x_1675_ = l_Lean_Expr_app___override(v___x_1662_, v___x_1674_);
v___x_1676_ = l_Lean_Expr_app___override(v___x_1659_, v___x_1675_);
v___x_1677_ = l_Lean_Expr_app___override(v___x_1658_, v___x_1676_);
lean_inc_ref(v_arg_1401_);
v___x_1678_ = l_Lean_Expr_app___override(v___x_1677_, v_arg_1401_);
lean_inc_ref(v_lit_1600_);
v___x_1679_ = l_Lean_Expr_app___override(v___x_1678_, v_lit_1600_);
v___x_1680_ = l_Lean_Expr_app___override(v___x_1655_, v___x_1679_);
lean_inc(v_u_1375_);
v___x_1681_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v_u_1375_, v_00_u03b1_1376_, v___x_1680_, v___x_1412_, v___y_1378_, v___y_1379_, v___y_1380_, v___y_1381_);
if (lean_obj_tag(v___x_1681_) == 0)
{
lean_object* v_a_1682_; lean_object* v___x_1684_; uint8_t v_isShared_1685_; uint8_t v_isSharedCheck_1797_; 
v_a_1682_ = lean_ctor_get(v___x_1681_, 0);
v_isSharedCheck_1797_ = !lean_is_exclusive(v___x_1681_);
if (v_isSharedCheck_1797_ == 0)
{
v___x_1684_ = v___x_1681_;
v_isShared_1685_ = v_isSharedCheck_1797_;
goto v_resetjp_1683_;
}
else
{
lean_inc(v_a_1682_);
lean_dec(v___x_1681_);
v___x_1684_ = lean_box(0);
v_isShared_1685_ = v_isSharedCheck_1797_;
goto v_resetjp_1683_;
}
v_resetjp_1683_:
{
switch(lean_obj_tag(v_a_1682_))
{
case 0:
{
lean_dec_ref_known(v_a_1682_, 1);
lean_del_object(v___x_1684_);
lean_dec_ref(v_proof_1601_);
lean_dec_ref(v_lit_1600_);
lean_dec(v_a_1409_);
lean_dec_ref_known(v___x_1405_, 2);
lean_dec_ref(v_arg_1401_);
lean_dec_ref(v_arg_1400_);
lean_dec_ref(v_e_1377_);
lean_dec_ref(v_00_u03b1_1376_);
lean_dec(v_u_1375_);
v___y_1384_ = v___y_1378_;
v___y_1385_ = v___y_1379_;
v___y_1386_ = v___y_1380_;
v___y_1387_ = v___y_1381_;
goto v___jp_1383_;
}
case 1:
{
lean_object* v_inst_1686_; lean_object* v_lit_1687_; lean_object* v_proof_1688_; lean_object* v___x_1690_; uint8_t v_isShared_1691_; uint8_t v_isSharedCheck_1708_; 
lean_dec_ref(v_e_1377_);
lean_dec(v_u_1375_);
v_inst_1686_ = lean_ctor_get(v_a_1682_, 0);
v_lit_1687_ = lean_ctor_get(v_a_1682_, 1);
v_proof_1688_ = lean_ctor_get(v_a_1682_, 2);
v_isSharedCheck_1708_ = !lean_is_exclusive(v_a_1682_);
if (v_isSharedCheck_1708_ == 0)
{
v___x_1690_ = v_a_1682_;
v_isShared_1691_ = v_isSharedCheck_1708_;
goto v_resetjp_1689_;
}
else
{
lean_inc(v_proof_1688_);
lean_inc(v_lit_1687_);
lean_inc(v_inst_1686_);
lean_dec(v_a_1682_);
v___x_1690_ = lean_box(0);
v_isShared_1691_ = v_isSharedCheck_1708_;
goto v_resetjp_1689_;
}
v_resetjp_1689_:
{
lean_object* v___x_1692_; lean_object* v___x_1693_; lean_object* v___x_1694_; lean_object* v___x_1695_; lean_object* v___x_1696_; lean_object* v___x_1697_; lean_object* v___x_1698_; lean_object* v___x_1699_; lean_object* v___x_1700_; lean_object* v___x_1701_; lean_object* v___x_1703_; 
v___x_1692_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__40));
v___x_1693_ = l_Lean_Expr_const___override(v___x_1692_, v___x_1405_);
v___x_1694_ = l_Lean_Expr_app___override(v___x_1693_, v_00_u03b1_1376_);
v___x_1695_ = l_Lean_Expr_app___override(v___x_1694_, v_a_1409_);
v___x_1696_ = l_Lean_Expr_app___override(v___x_1695_, v_arg_1401_);
v___x_1697_ = l_Lean_Expr_app___override(v___x_1696_, v_arg_1400_);
v___x_1698_ = l_Lean_Expr_app___override(v___x_1697_, v_lit_1600_);
lean_inc_ref(v_lit_1687_);
v___x_1699_ = l_Lean_Expr_app___override(v___x_1698_, v_lit_1687_);
v___x_1700_ = l_Lean_Expr_app___override(v___x_1699_, v_proof_1601_);
v___x_1701_ = l_Lean_Expr_app___override(v___x_1700_, v_proof_1688_);
if (v_isShared_1691_ == 0)
{
lean_ctor_set(v___x_1690_, 2, v___x_1701_);
v___x_1703_ = v___x_1690_;
goto v_reusejp_1702_;
}
else
{
lean_object* v_reuseFailAlloc_1707_; 
v_reuseFailAlloc_1707_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1707_, 0, v_inst_1686_);
lean_ctor_set(v_reuseFailAlloc_1707_, 1, v_lit_1687_);
lean_ctor_set(v_reuseFailAlloc_1707_, 2, v___x_1701_);
v___x_1703_ = v_reuseFailAlloc_1707_;
goto v_reusejp_1702_;
}
v_reusejp_1702_:
{
lean_object* v___x_1705_; 
if (v_isShared_1685_ == 0)
{
lean_ctor_set(v___x_1684_, 0, v___x_1703_);
v___x_1705_ = v___x_1684_;
goto v_reusejp_1704_;
}
else
{
lean_object* v_reuseFailAlloc_1706_; 
v_reuseFailAlloc_1706_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1706_, 0, v___x_1703_);
v___x_1705_ = v_reuseFailAlloc_1706_;
goto v_reusejp_1704_;
}
v_reusejp_1704_:
{
return v___x_1705_;
}
}
}
}
case 2:
{
lean_object* v_inst_1709_; lean_object* v_lit_1710_; lean_object* v_proof_1711_; lean_object* v___x_1713_; uint8_t v_isShared_1714_; uint8_t v_isSharedCheck_1748_; 
lean_del_object(v___x_1684_);
lean_dec(v_a_1409_);
lean_dec_ref(v_e_1377_);
lean_dec(v_u_1375_);
v_inst_1709_ = lean_ctor_get(v_a_1682_, 0);
v_lit_1710_ = lean_ctor_get(v_a_1682_, 1);
v_proof_1711_ = lean_ctor_get(v_a_1682_, 2);
v_isSharedCheck_1748_ = !lean_is_exclusive(v_a_1682_);
if (v_isSharedCheck_1748_ == 0)
{
v___x_1713_ = v_a_1682_;
v_isShared_1714_ = v_isSharedCheck_1748_;
goto v_resetjp_1712_;
}
else
{
lean_inc(v_proof_1711_);
lean_inc(v_lit_1710_);
lean_inc(v_inst_1709_);
lean_dec(v_a_1682_);
v___x_1713_ = lean_box(0);
v_isShared_1714_ = v_isSharedCheck_1748_;
goto v_resetjp_1712_;
}
v_resetjp_1712_:
{
lean_object* v___x_1715_; lean_object* v___x_1716_; lean_object* v___x_1717_; lean_object* v___x_1718_; 
v___x_1715_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__18));
lean_inc_ref(v___x_1405_);
v___x_1716_ = l_Lean_Expr_const___override(v___x_1715_, v___x_1405_);
lean_inc_ref(v_00_u03b1_1376_);
v___x_1717_ = l_Lean_Expr_app___override(v___x_1716_, v_00_u03b1_1376_);
v___x_1718_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_1717_, v___y_1378_, v___y_1379_, v___y_1380_, v___y_1381_);
if (lean_obj_tag(v___x_1718_) == 0)
{
lean_object* v_a_1719_; lean_object* v___x_1721_; uint8_t v_isShared_1722_; uint8_t v_isSharedCheck_1739_; 
v_a_1719_ = lean_ctor_get(v___x_1718_, 0);
v_isSharedCheck_1739_ = !lean_is_exclusive(v___x_1718_);
if (v_isSharedCheck_1739_ == 0)
{
v___x_1721_ = v___x_1718_;
v_isShared_1722_ = v_isSharedCheck_1739_;
goto v_resetjp_1720_;
}
else
{
lean_inc(v_a_1719_);
lean_dec(v___x_1718_);
v___x_1721_ = lean_box(0);
v_isShared_1722_ = v_isSharedCheck_1739_;
goto v_resetjp_1720_;
}
v_resetjp_1720_:
{
lean_object* v___x_1723_; lean_object* v___x_1724_; lean_object* v___x_1725_; lean_object* v___x_1726_; lean_object* v___x_1727_; lean_object* v___x_1728_; lean_object* v___x_1729_; lean_object* v___x_1730_; lean_object* v___x_1731_; lean_object* v___x_1732_; lean_object* v___x_1734_; 
v___x_1723_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__42));
v___x_1724_ = l_Lean_Expr_const___override(v___x_1723_, v___x_1405_);
v___x_1725_ = l_Lean_Expr_app___override(v___x_1724_, v_00_u03b1_1376_);
v___x_1726_ = l_Lean_Expr_app___override(v___x_1725_, v_a_1719_);
v___x_1727_ = l_Lean_Expr_app___override(v___x_1726_, v_arg_1401_);
v___x_1728_ = l_Lean_Expr_app___override(v___x_1727_, v_arg_1400_);
v___x_1729_ = l_Lean_Expr_app___override(v___x_1728_, v_lit_1600_);
lean_inc_ref(v_lit_1710_);
v___x_1730_ = l_Lean_Expr_app___override(v___x_1729_, v_lit_1710_);
v___x_1731_ = l_Lean_Expr_app___override(v___x_1730_, v_proof_1601_);
v___x_1732_ = l_Lean_Expr_app___override(v___x_1731_, v_proof_1711_);
if (v_isShared_1714_ == 0)
{
lean_ctor_set(v___x_1713_, 2, v___x_1732_);
v___x_1734_ = v___x_1713_;
goto v_reusejp_1733_;
}
else
{
lean_object* v_reuseFailAlloc_1738_; 
v_reuseFailAlloc_1738_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1738_, 0, v_inst_1709_);
lean_ctor_set(v_reuseFailAlloc_1738_, 1, v_lit_1710_);
lean_ctor_set(v_reuseFailAlloc_1738_, 2, v___x_1732_);
v___x_1734_ = v_reuseFailAlloc_1738_;
goto v_reusejp_1733_;
}
v_reusejp_1733_:
{
lean_object* v___x_1736_; 
if (v_isShared_1722_ == 0)
{
lean_ctor_set(v___x_1721_, 0, v___x_1734_);
v___x_1736_ = v___x_1721_;
goto v_reusejp_1735_;
}
else
{
lean_object* v_reuseFailAlloc_1737_; 
v_reuseFailAlloc_1737_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1737_, 0, v___x_1734_);
v___x_1736_ = v_reuseFailAlloc_1737_;
goto v_reusejp_1735_;
}
v_reusejp_1735_:
{
return v___x_1736_;
}
}
}
}
else
{
lean_object* v_a_1740_; lean_object* v___x_1742_; uint8_t v_isShared_1743_; uint8_t v_isSharedCheck_1747_; 
lean_del_object(v___x_1713_);
lean_dec_ref(v_proof_1711_);
lean_dec_ref(v_lit_1710_);
lean_dec_ref(v_inst_1709_);
lean_dec_ref(v_proof_1601_);
lean_dec_ref(v_lit_1600_);
lean_dec_ref_known(v___x_1405_, 2);
lean_dec_ref(v_arg_1401_);
lean_dec_ref(v_arg_1400_);
lean_dec_ref(v_00_u03b1_1376_);
v_a_1740_ = lean_ctor_get(v___x_1718_, 0);
v_isSharedCheck_1747_ = !lean_is_exclusive(v___x_1718_);
if (v_isSharedCheck_1747_ == 0)
{
v___x_1742_ = v___x_1718_;
v_isShared_1743_ = v_isSharedCheck_1747_;
goto v_resetjp_1741_;
}
else
{
lean_inc(v_a_1740_);
lean_dec(v___x_1718_);
v___x_1742_ = lean_box(0);
v_isShared_1743_ = v_isSharedCheck_1747_;
goto v_resetjp_1741_;
}
v_resetjp_1741_:
{
lean_object* v___x_1745_; 
if (v_isShared_1743_ == 0)
{
v___x_1745_ = v___x_1742_;
goto v_reusejp_1744_;
}
else
{
lean_object* v_reuseFailAlloc_1746_; 
v_reuseFailAlloc_1746_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1746_, 0, v_a_1740_);
v___x_1745_ = v_reuseFailAlloc_1746_;
goto v_reusejp_1744_;
}
v_reusejp_1744_:
{
return v___x_1745_;
}
}
}
}
}
case 3:
{
lean_object* v_inst_1749_; lean_object* v_q_1750_; lean_object* v_n_1751_; lean_object* v_d_1752_; lean_object* v_proof_1753_; lean_object* v___x_1755_; uint8_t v_isShared_1756_; uint8_t v_isSharedCheck_1774_; 
lean_dec_ref(v_e_1377_);
lean_dec(v_u_1375_);
v_inst_1749_ = lean_ctor_get(v_a_1682_, 0);
v_q_1750_ = lean_ctor_get(v_a_1682_, 1);
v_n_1751_ = lean_ctor_get(v_a_1682_, 2);
v_d_1752_ = lean_ctor_get(v_a_1682_, 3);
v_proof_1753_ = lean_ctor_get(v_a_1682_, 4);
v_isSharedCheck_1774_ = !lean_is_exclusive(v_a_1682_);
if (v_isSharedCheck_1774_ == 0)
{
v___x_1755_ = v_a_1682_;
v_isShared_1756_ = v_isSharedCheck_1774_;
goto v_resetjp_1754_;
}
else
{
lean_inc(v_proof_1753_);
lean_inc(v_d_1752_);
lean_inc(v_n_1751_);
lean_inc(v_q_1750_);
lean_inc(v_inst_1749_);
lean_dec(v_a_1682_);
v___x_1755_ = lean_box(0);
v_isShared_1756_ = v_isSharedCheck_1774_;
goto v_resetjp_1754_;
}
v_resetjp_1754_:
{
lean_object* v___x_1757_; lean_object* v___x_1758_; lean_object* v___x_1759_; lean_object* v___x_1760_; lean_object* v___x_1761_; lean_object* v___x_1762_; lean_object* v___x_1763_; lean_object* v___x_1764_; lean_object* v___x_1765_; lean_object* v___x_1766_; lean_object* v___x_1767_; lean_object* v___x_1769_; 
v___x_1757_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__44));
v___x_1758_ = l_Lean_Expr_const___override(v___x_1757_, v___x_1405_);
v___x_1759_ = l_Lean_Expr_app___override(v___x_1758_, v_00_u03b1_1376_);
v___x_1760_ = l_Lean_Expr_app___override(v___x_1759_, v_a_1409_);
v___x_1761_ = l_Lean_Expr_app___override(v___x_1760_, v_arg_1401_);
v___x_1762_ = l_Lean_Expr_app___override(v___x_1761_, v_arg_1400_);
v___x_1763_ = l_Lean_Expr_app___override(v___x_1762_, v_lit_1600_);
lean_inc_ref(v_n_1751_);
v___x_1764_ = l_Lean_Expr_app___override(v___x_1763_, v_n_1751_);
lean_inc_ref(v_d_1752_);
v___x_1765_ = l_Lean_Expr_app___override(v___x_1764_, v_d_1752_);
v___x_1766_ = l_Lean_Expr_app___override(v___x_1765_, v_proof_1601_);
v___x_1767_ = l_Lean_Expr_app___override(v___x_1766_, v_proof_1753_);
if (v_isShared_1756_ == 0)
{
lean_ctor_set(v___x_1755_, 4, v___x_1767_);
v___x_1769_ = v___x_1755_;
goto v_reusejp_1768_;
}
else
{
lean_object* v_reuseFailAlloc_1773_; 
v_reuseFailAlloc_1773_ = lean_alloc_ctor(3, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1773_, 0, v_inst_1749_);
lean_ctor_set(v_reuseFailAlloc_1773_, 1, v_q_1750_);
lean_ctor_set(v_reuseFailAlloc_1773_, 2, v_n_1751_);
lean_ctor_set(v_reuseFailAlloc_1773_, 3, v_d_1752_);
lean_ctor_set(v_reuseFailAlloc_1773_, 4, v___x_1767_);
v___x_1769_ = v_reuseFailAlloc_1773_;
goto v_reusejp_1768_;
}
v_reusejp_1768_:
{
lean_object* v___x_1771_; 
if (v_isShared_1685_ == 0)
{
lean_ctor_set(v___x_1684_, 0, v___x_1769_);
v___x_1771_ = v___x_1684_;
goto v_reusejp_1770_;
}
else
{
lean_object* v_reuseFailAlloc_1772_; 
v_reuseFailAlloc_1772_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1772_, 0, v___x_1769_);
v___x_1771_ = v_reuseFailAlloc_1772_;
goto v_reusejp_1770_;
}
v_reusejp_1770_:
{
return v___x_1771_;
}
}
}
}
default: 
{
lean_object* v_inst_1775_; lean_object* v_q_1776_; lean_object* v_n_1777_; lean_object* v_d_1778_; lean_object* v_proof_1779_; lean_object* v___x_1780_; lean_object* v___x_1781_; lean_object* v___x_1782_; lean_object* v___x_1783_; lean_object* v___x_1784_; lean_object* v___x_1785_; lean_object* v___x_1786_; lean_object* v___x_1787_; lean_object* v___x_1788_; lean_object* v___x_1789_; lean_object* v___x_1790_; lean_object* v___x_1791_; lean_object* v___x_1792_; lean_object* v___x_1793_; lean_object* v___x_1795_; 
lean_dec(v_a_1409_);
v_inst_1775_ = lean_ctor_get(v_a_1682_, 0);
lean_inc_ref_n(v_inst_1775_, 2);
v_q_1776_ = lean_ctor_get(v_a_1682_, 1);
lean_inc_ref(v_q_1776_);
v_n_1777_ = lean_ctor_get(v_a_1682_, 2);
lean_inc_ref(v_n_1777_);
v_d_1778_ = lean_ctor_get(v_a_1682_, 3);
lean_inc_ref_n(v_d_1778_, 2);
v_proof_1779_ = lean_ctor_get(v_a_1682_, 4);
lean_inc_ref(v_proof_1779_);
lean_dec_ref_known(v_a_1682_, 5);
v___x_1780_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__8, &lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__8_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntPow___closed__8);
v___x_1781_ = l_Lean_Expr_app___override(v___x_1780_, v_n_1777_);
v___x_1782_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___closed__46));
v___x_1783_ = l_Lean_Expr_const___override(v___x_1782_, v___x_1405_);
lean_inc_ref(v_00_u03b1_1376_);
v___x_1784_ = l_Lean_Expr_app___override(v___x_1783_, v_00_u03b1_1376_);
v___x_1785_ = l_Lean_Expr_app___override(v___x_1784_, v_inst_1775_);
v___x_1786_ = l_Lean_Expr_app___override(v___x_1785_, v_arg_1401_);
v___x_1787_ = l_Lean_Expr_app___override(v___x_1786_, v_arg_1400_);
v___x_1788_ = l_Lean_Expr_app___override(v___x_1787_, v_lit_1600_);
lean_inc_ref(v___x_1781_);
v___x_1789_ = l_Lean_Expr_app___override(v___x_1788_, v___x_1781_);
v___x_1790_ = l_Lean_Expr_app___override(v___x_1789_, v_d_1778_);
v___x_1791_ = l_Lean_Expr_app___override(v___x_1790_, v_proof_1601_);
v___x_1792_ = l_Lean_Expr_app___override(v___x_1791_, v_proof_1779_);
v___x_1793_ = lp_mathlib_Mathlib_Meta_NormNum_Result_isRat(v_u_1375_, v_00_u03b1_1376_, v_e_1377_, v_inst_1775_, v_q_1776_, v___x_1781_, v_d_1778_, v___x_1792_);
if (v_isShared_1685_ == 0)
{
lean_ctor_set(v___x_1684_, 0, v___x_1793_);
v___x_1795_ = v___x_1684_;
goto v_reusejp_1794_;
}
else
{
lean_object* v_reuseFailAlloc_1796_; 
v_reuseFailAlloc_1796_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1796_, 0, v___x_1793_);
v___x_1795_ = v_reuseFailAlloc_1796_;
goto v_reusejp_1794_;
}
v_reusejp_1794_:
{
return v___x_1795_;
}
}
}
}
}
else
{
lean_dec_ref(v_proof_1601_);
lean_dec_ref(v_lit_1600_);
lean_dec(v_a_1409_);
lean_dec_ref_known(v___x_1405_, 2);
lean_dec_ref(v_arg_1401_);
lean_dec_ref(v_arg_1400_);
lean_dec_ref(v_e_1377_);
lean_dec_ref(v_00_u03b1_1376_);
lean_dec(v_u_1375_);
return v___x_1681_;
}
}
else
{
lean_object* v_a_1798_; lean_object* v___x_1800_; uint8_t v_isShared_1801_; uint8_t v_isSharedCheck_1805_; 
lean_dec_ref(v___x_1627_);
lean_dec_ref(v___x_1613_);
lean_dec_ref(v___x_1606_);
lean_dec_ref(v_proof_1601_);
lean_dec_ref(v_lit_1600_);
lean_dec(v_a_1409_);
lean_dec_ref_known(v___x_1405_, 2);
lean_dec_ref(v_arg_1401_);
lean_dec_ref(v_arg_1400_);
lean_dec_ref(v_e_1377_);
lean_dec_ref(v_00_u03b1_1376_);
lean_dec(v_u_1375_);
v_a_1798_ = lean_ctor_get(v___x_1635_, 0);
v_isSharedCheck_1805_ = !lean_is_exclusive(v___x_1635_);
if (v_isSharedCheck_1805_ == 0)
{
v___x_1800_ = v___x_1635_;
v_isShared_1801_ = v_isSharedCheck_1805_;
goto v_resetjp_1799_;
}
else
{
lean_inc(v_a_1798_);
lean_dec(v___x_1635_);
v___x_1800_ = lean_box(0);
v_isShared_1801_ = v_isSharedCheck_1805_;
goto v_resetjp_1799_;
}
v_resetjp_1799_:
{
lean_object* v___x_1803_; 
if (v_isShared_1801_ == 0)
{
v___x_1803_ = v___x_1800_;
goto v_reusejp_1802_;
}
else
{
lean_object* v_reuseFailAlloc_1804_; 
v_reuseFailAlloc_1804_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1804_, 0, v_a_1798_);
v___x_1803_ = v_reuseFailAlloc_1804_;
goto v_reusejp_1802_;
}
v_reusejp_1802_:
{
return v___x_1803_;
}
}
}
}
default: 
{
lean_object* v___x_1806_; lean_object* v___x_1807_; 
lean_dec(v_a_1414_);
lean_dec(v_a_1409_);
lean_dec_ref_known(v___x_1405_, 2);
lean_dec(v___x_1402_);
lean_dec_ref(v_arg_1401_);
lean_dec_ref(v_arg_1400_);
lean_dec_ref(v_e_1377_);
lean_dec_ref(v_00_u03b1_1376_);
lean_dec(v_u_1375_);
v___x_1806_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__1);
v___x_1807_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalPow_spec__0___redArg(v___x_1806_, v___y_1378_, v___y_1379_, v___y_1380_, v___y_1381_);
return v___x_1807_;
}
}
}
else
{
lean_dec(v_a_1409_);
lean_dec_ref_known(v___x_1405_, 2);
lean_dec(v___x_1402_);
lean_dec_ref(v_arg_1401_);
lean_dec_ref(v_arg_1400_);
lean_dec_ref(v_e_1377_);
lean_dec_ref(v_00_u03b1_1376_);
lean_dec(v_u_1375_);
return v___x_1413_;
}
}
else
{
lean_object* v_a_1808_; lean_object* v___x_1810_; uint8_t v_isShared_1811_; uint8_t v_isSharedCheck_1815_; 
lean_dec_ref_known(v___x_1405_, 2);
lean_dec(v___x_1402_);
lean_dec_ref(v_arg_1401_);
lean_dec_ref(v_arg_1400_);
lean_dec_ref(v_e_1377_);
lean_dec_ref(v_00_u03b1_1376_);
lean_dec(v_u_1375_);
v_a_1808_ = lean_ctor_get(v___x_1408_, 0);
v_isSharedCheck_1815_ = !lean_is_exclusive(v___x_1408_);
if (v_isSharedCheck_1815_ == 0)
{
v___x_1810_ = v___x_1408_;
v_isShared_1811_ = v_isSharedCheck_1815_;
goto v_resetjp_1809_;
}
else
{
lean_inc(v_a_1808_);
lean_dec(v___x_1408_);
v___x_1810_ = lean_box(0);
v_isShared_1811_ = v_isSharedCheck_1815_;
goto v_resetjp_1809_;
}
v_resetjp_1809_:
{
lean_object* v___x_1813_; 
if (v_isShared_1811_ == 0)
{
v___x_1813_ = v___x_1810_;
goto v_reusejp_1812_;
}
else
{
lean_object* v_reuseFailAlloc_1814_; 
v_reuseFailAlloc_1814_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1814_, 0, v_a_1808_);
v___x_1813_ = v_reuseFailAlloc_1814_;
goto v_reusejp_1812_;
}
v_reusejp_1812_:
{
return v___x_1813_;
}
}
}
}
else
{
lean_dec_ref_known(v_a_1391_, 2);
lean_dec_ref(v_fn_1399_);
lean_dec_ref(v_e_1377_);
lean_dec_ref(v_00_u03b1_1376_);
lean_dec(v_u_1375_);
v___y_1393_ = v___y_1378_;
v___y_1394_ = v___y_1379_;
v___y_1395_ = v___y_1380_;
v___y_1396_ = v___y_1381_;
goto v___jp_1392_;
}
}
else
{
lean_dec(v_a_1391_);
lean_dec_ref(v_e_1377_);
lean_dec_ref(v_00_u03b1_1376_);
lean_dec(v_u_1375_);
v___y_1393_ = v___y_1378_;
v___y_1394_ = v___y_1379_;
v___y_1395_ = v___y_1380_;
v___y_1396_ = v___y_1381_;
goto v___jp_1392_;
}
v___jp_1392_:
{
lean_object* v___x_1397_; lean_object* v___x_1398_; 
v___x_1397_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__1);
v___x_1398_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalPow_spec__0___redArg(v___x_1397_, v___y_1393_, v___y_1394_, v___y_1395_, v___y_1396_);
return v___x_1398_;
}
}
else
{
lean_object* v_a_1816_; lean_object* v___x_1818_; uint8_t v_isShared_1819_; uint8_t v_isSharedCheck_1823_; 
lean_dec_ref(v_e_1377_);
lean_dec_ref(v_00_u03b1_1376_);
lean_dec(v_u_1375_);
v_a_1816_ = lean_ctor_get(v___x_1390_, 0);
v_isSharedCheck_1823_ = !lean_is_exclusive(v___x_1390_);
if (v_isSharedCheck_1823_ == 0)
{
v___x_1818_ = v___x_1390_;
v_isShared_1819_ = v_isSharedCheck_1823_;
goto v_resetjp_1817_;
}
else
{
lean_inc(v_a_1816_);
lean_dec(v___x_1390_);
v___x_1818_ = lean_box(0);
v_isShared_1819_ = v_isSharedCheck_1823_;
goto v_resetjp_1817_;
}
v_resetjp_1817_:
{
lean_object* v___x_1821_; 
if (v_isShared_1819_ == 0)
{
v___x_1821_ = v___x_1818_;
goto v_reusejp_1820_;
}
else
{
lean_object* v_reuseFailAlloc_1822_; 
v_reuseFailAlloc_1822_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1822_, 0, v_a_1816_);
v___x_1821_ = v_reuseFailAlloc_1822_;
goto v_reusejp_1820_;
}
v_reusejp_1820_:
{
return v___x_1821_;
}
}
}
v___jp_1383_:
{
lean_object* v___x_1388_; lean_object* v___x_1389_; 
v___x_1388_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalPow___lam__1___closed__1);
v___x_1389_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalPow_spec__0___redArg(v___x_1388_, v___y_1384_, v___y_1385_, v___y_1386_, v___y_1387_);
return v___x_1389_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0___boxed(lean_object* v_u_1824_, lean_object* v_00_u03b1_1825_, lean_object* v_e_1826_, lean_object* v___y_1827_, lean_object* v___y_1828_, lean_object* v___y_1829_, lean_object* v___y_1830_, lean_object* v___y_1831_){
_start:
{
lean_object* v_res_1832_; 
v_res_1832_ = lp_mathlib_Mathlib_Meta_NormNum_evalZPow___lam__0(v_u_1824_, v_00_u03b1_1825_, v_e_1826_, v___y_1827_, v___y_1828_, v___y_1829_, v___y_1830_);
lean_dec(v___y_1830_);
lean_dec_ref(v___y_1829_);
lean_dec(v___y_1828_);
lean_dec_ref(v___y_1827_);
return v_res_1832_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Lemmas(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_Qq(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Pow(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_Qq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_NormNum_Pow(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Int_Cast_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_NormNum_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_Qq(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_NormNum_Pow(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Cast_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_NormNum_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_Qq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Pow(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_NormNum_Pow(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_NormNum_Pow(builtin);
}
#ifdef __cplusplus
}
#endif
