// Lean compiler output
// Module: Mathlib.Tactic.NormNum.PowMod
// Imports: public import Init public meta import Init public import Mathlib.Tactic.NormNum.Pow
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
lean_object* l_Lean_Level_succ___override(lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_lit___override(lean_object*);
lean_object* lp_batteries_Lean_Expr_natLit_x21(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* l_Lean_mkRawNatLit(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lean_nat_land(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
lean_object* lean_nat_log2(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__2;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__3;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__4;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__6_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__7;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__8;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "mod"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__10_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__9_value),LEAN_SCALAR_PTR_LITERAL(244, 133, 16, 0, 168, 19, 182, 179)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__10_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__11;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "pow"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__13_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(155, 64, 52, 77, 166, 227, 131, 174)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__13_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__14;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__15_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__16_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "NormNum"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__17_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "IsNatPowModT"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__18_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trans"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__19_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__15_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__20_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__20_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__20_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__20_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__17_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__20_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__20_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__18_value),LEAN_SCALAR_PTR_LITERAL(15, 253, 173, 117, 65, 147, 155, 156)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__20_value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__19_value),LEAN_SCALAR_PTR_LITERAL(29, 246, 209, 7, 48, 77, 184, 123)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__20_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__21;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "bit1"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__22_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__23_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__15_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__23_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__23_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__23_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__23_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__17_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__23_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__23_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__18_value),LEAN_SCALAR_PTR_LITERAL(15, 253, 173, 117, 65, 147, 155, 156)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__23_value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__22_value),LEAN_SCALAR_PTR_LITERAL(108, 153, 75, 73, 44, 23, 192, 107)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__23 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__23_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__24;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "bit0"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__25 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__25_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__26_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__15_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__26_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__26_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__26_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__26_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__17_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__26_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__26_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__18_value),LEAN_SCALAR_PTR_LITERAL(15, 253, 173, 117, 65, 147, 155, 156)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__26_value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__25_value),LEAN_SCALAR_PTR_LITERAL(100, 82, 39, 206, 209, 32, 80, 217)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__26 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__26_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__27;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "run"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__15_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__3_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__17_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__3_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__3_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__18_value),LEAN_SCALAR_PTR_LITERAL(15, 253, 173, 117, 65, 147, 155, 156)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__3_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__2_value),LEAN_SCALAR_PTR_LITERAL(209, 89, 151, 173, 242, 77, 63, 226)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__4;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "natPow_one_natMod"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__15_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__6_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__6_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__17_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__6_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__5_value),LEAN_SCALAR_PTR_LITERAL(76, 92, 179, 57, 100, 253, 181, 157)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__7;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "natPow_zero_natMod_succ_succ"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__15_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__9_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__9_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__9_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__17_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__9_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__8_value),LEAN_SCALAR_PTR_LITERAL(136, 88, 240, 211, 34, 155, 228, 250)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__9_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__10;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__11_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__12;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "natPow_zero_natMod_one"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__15_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__14_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__14_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__14_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__17_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__14_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__13_value),LEAN_SCALAR_PTR_LITERAL(63, 133, 236, 3, 18, 166, 197, 53)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__14_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__15;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "natPow_zero_natMod_zero"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__15_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__17_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__17_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__17_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__17_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__17_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__17_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__16_value),LEAN_SCALAR_PTR_LITERAL(99, 58, 154, 113, 166, 10, 119, 252)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__17_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__18;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__2(void){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; 
v___x_4_ = lean_box(0);
v___x_5_ = l_Lean_Level_succ___override(v___x_4_);
return v___x_5_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__3(void){
_start:
{
lean_object* v___x_6_; lean_object* v___x_7_; lean_object* v___x_8_; 
v___x_6_ = lean_box(0);
v___x_7_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__2, &lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__2);
v___x_8_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_8_, 0, v___x_7_);
lean_ctor_set(v___x_8_, 1, v___x_6_);
return v___x_8_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__4(void){
_start:
{
lean_object* v___x_9_; lean_object* v___x_10_; lean_object* v___x_11_; 
v___x_9_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__3, &lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__3);
v___x_10_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__1));
v___x_11_ = l_Lean_Expr_const___override(v___x_10_, v___x_9_);
return v___x_11_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__7(void){
_start:
{
lean_object* v___x_15_; lean_object* v___x_16_; lean_object* v___x_17_; 
v___x_15_ = lean_box(0);
v___x_16_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__6));
v___x_17_ = l_Lean_Expr_const___override(v___x_16_, v___x_15_);
return v___x_17_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__8(void){
_start:
{
lean_object* v___x_18_; lean_object* v___x_19_; lean_object* v___x_20_; 
v___x_18_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__7, &lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__7);
v___x_19_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__4, &lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__4);
v___x_20_ = l_Lean_Expr_app___override(v___x_19_, v___x_18_);
return v___x_20_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__11(void){
_start:
{
lean_object* v___x_25_; lean_object* v___x_26_; lean_object* v___x_27_; 
v___x_25_ = lean_box(0);
v___x_26_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__10));
v___x_27_ = l_Lean_Expr_const___override(v___x_26_, v___x_25_);
return v___x_27_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__14(void){
_start:
{
lean_object* v___x_32_; lean_object* v___x_33_; lean_object* v___x_34_; 
v___x_32_ = lean_box(0);
v___x_33_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__13));
v___x_34_ = l_Lean_Expr_const___override(v___x_33_, v___x_32_);
return v___x_34_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__21(void){
_start:
{
lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; 
v___x_46_ = lean_box(0);
v___x_47_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__20));
v___x_48_ = l_Lean_Expr_const___override(v___x_47_, v___x_46_);
return v___x_48_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__24(void){
_start:
{
lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; 
v___x_56_ = lean_box(0);
v___x_57_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__23));
v___x_58_ = l_Lean_Expr_const___override(v___x_57_, v___x_56_);
return v___x_58_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__27(void){
_start:
{
lean_object* v___x_66_; lean_object* v___x_67_; lean_object* v___x_68_; 
v___x_66_ = lean_box(0);
v___x_67_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__26));
v___x_68_ = l_Lean_Expr_const___override(v___x_67_, v___x_66_);
return v___x_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg(lean_object* v_depth_69_, lean_object* v_a_70_, lean_object* v_m_71_, lean_object* v_b_u2080_72_, lean_object* v_c_u2080_73_, lean_object* v_b_74_, lean_object* v_p_75_){
_start:
{
lean_object* v_b_x27_76_; lean_object* v___x_77_; uint8_t v___x_78_; 
v_b_x27_76_ = lp_batteries_Lean_Expr_natLit_x21(v_b_74_);
v___x_77_ = lean_unsigned_to_nat(1u);
v___x_78_ = lean_nat_dec_le(v_depth_69_, v___x_77_);
if (v___x_78_ == 0)
{
lean_object* v_d_79_; lean_object* v___x_80_; lean_object* v_hi_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v_fst_84_; lean_object* v_snd_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v_fst_96_; lean_object* v_snd_97_; lean_object* v___x_99_; uint8_t v_isShared_100_; uint8_t v_isSharedCheck_114_; 
v_d_79_ = lean_nat_shiftr(v_depth_69_, v___x_77_);
v___x_80_ = lean_nat_shiftr(v_b_x27_76_, v_d_79_);
lean_dec(v_b_x27_76_);
v_hi_81_ = l_Lean_mkRawNatLit(v___x_80_);
v___x_82_ = lean_nat_sub(v_depth_69_, v_d_79_);
lean_inc_ref(v_p_75_);
lean_inc_ref_n(v_hi_81_, 3);
lean_inc_ref_n(v_m_71_, 3);
lean_inc_ref_n(v_a_70_, 3);
v___x_83_ = lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg(v___x_82_, v_a_70_, v_m_71_, v_b_u2080_72_, v_c_u2080_73_, v_hi_81_, v_p_75_);
lean_dec(v___x_82_);
v_fst_84_ = lean_ctor_get(v___x_83_, 0);
lean_inc_n(v_fst_84_, 3);
v_snd_85_ = lean_ctor_get(v___x_83_, 1);
lean_inc(v_snd_85_);
lean_dec_ref(v___x_83_);
v___x_86_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__8, &lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__8_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__8);
v___x_87_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__11, &lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__11_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__11);
v___x_88_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__14, &lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__14_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__14);
v___x_89_ = l_Lean_Expr_app___override(v___x_88_, v_a_70_);
v___x_90_ = l_Lean_Expr_app___override(v___x_89_, v_hi_81_);
v___x_91_ = l_Lean_Expr_app___override(v___x_87_, v___x_90_);
v___x_92_ = l_Lean_Expr_app___override(v___x_91_, v_m_71_);
v___x_93_ = l_Lean_Expr_app___override(v___x_86_, v___x_92_);
v___x_94_ = l_Lean_Expr_app___override(v___x_93_, v_fst_84_);
lean_inc_ref(v_b_74_);
v___x_95_ = lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg(v_d_79_, v_a_70_, v_m_71_, v_hi_81_, v_fst_84_, v_b_74_, v___x_94_);
lean_dec(v_d_79_);
v_fst_96_ = lean_ctor_get(v___x_95_, 0);
v_snd_97_ = lean_ctor_get(v___x_95_, 1);
v_isSharedCheck_114_ = !lean_is_exclusive(v___x_95_);
if (v_isSharedCheck_114_ == 0)
{
v___x_99_ = v___x_95_;
v_isShared_100_ = v_isSharedCheck_114_;
goto v_resetjp_98_;
}
else
{
lean_inc(v_snd_97_);
lean_inc(v_fst_96_);
lean_dec(v___x_95_);
v___x_99_ = lean_box(0);
v_isShared_100_ = v_isSharedCheck_114_;
goto v_resetjp_98_;
}
v_resetjp_98_:
{
lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_112_; 
v___x_101_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__21, &lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__21_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__21);
v___x_102_ = l_Lean_Expr_app___override(v___x_101_, v_p_75_);
v___x_103_ = l_Lean_Expr_app___override(v___x_102_, v_a_70_);
v___x_104_ = l_Lean_Expr_app___override(v___x_103_, v_hi_81_);
v___x_105_ = l_Lean_Expr_app___override(v___x_104_, v_m_71_);
v___x_106_ = l_Lean_Expr_app___override(v___x_105_, v_fst_84_);
v___x_107_ = l_Lean_Expr_app___override(v___x_106_, v_b_74_);
lean_inc(v_fst_96_);
v___x_108_ = l_Lean_Expr_app___override(v___x_107_, v_fst_96_);
v___x_109_ = l_Lean_Expr_app___override(v___x_108_, v_snd_85_);
v___x_110_ = l_Lean_Expr_app___override(v___x_109_, v_snd_97_);
if (v_isShared_100_ == 0)
{
lean_ctor_set(v___x_99_, 1, v___x_110_);
v___x_112_ = v___x_99_;
goto v_reusejp_111_;
}
else
{
lean_object* v_reuseFailAlloc_113_; 
v_reuseFailAlloc_113_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_113_, 0, v_fst_96_);
lean_ctor_set(v_reuseFailAlloc_113_, 1, v___x_110_);
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
lean_object* v_m_x27_115_; lean_object* v_c_u2080_x27_116_; lean_object* v___x_117_; lean_object* v___x_118_; uint8_t v___x_119_; 
lean_dec_ref(v_p_75_);
lean_dec_ref(v_b_74_);
v_m_x27_115_ = lp_batteries_Lean_Expr_natLit_x21(v_m_71_);
v_c_u2080_x27_116_ = lp_batteries_Lean_Expr_natLit_x21(v_c_u2080_73_);
v___x_117_ = lean_nat_land(v_b_x27_76_, v___x_77_);
lean_dec(v_b_x27_76_);
v___x_118_ = lean_unsigned_to_nat(0u);
v___x_119_ = lean_nat_dec_eq(v___x_117_, v___x_118_);
lean_dec(v___x_117_);
if (v___x_119_ == 0)
{
lean_object* v_a_x27_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v_c_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; 
v_a_x27_120_ = lp_batteries_Lean_Expr_natLit_x21(v_a_70_);
v___x_121_ = lean_nat_mul(v_c_u2080_x27_116_, v_a_x27_120_);
lean_dec(v_a_x27_120_);
v___x_122_ = lean_nat_mod(v___x_121_, v_m_x27_115_);
lean_dec(v___x_121_);
v___x_123_ = lean_nat_mul(v_c_u2080_x27_116_, v___x_122_);
lean_dec(v___x_122_);
lean_dec(v_c_u2080_x27_116_);
v___x_124_ = lean_nat_mod(v___x_123_, v_m_x27_115_);
lean_dec(v_m_x27_115_);
lean_dec(v___x_123_);
v_c_125_ = l_Lean_mkRawNatLit(v___x_124_);
v___x_126_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__24, &lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__24_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__24);
v___x_127_ = l_Lean_Expr_app___override(v___x_126_, v_a_70_);
v___x_128_ = l_Lean_Expr_app___override(v___x_127_, v_b_u2080_72_);
v___x_129_ = l_Lean_Expr_app___override(v___x_128_, v_m_71_);
v___x_130_ = l_Lean_Expr_app___override(v___x_129_, v_c_u2080_73_);
v___x_131_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_131_, 0, v_c_125_);
lean_ctor_set(v___x_131_, 1, v___x_130_);
return v___x_131_;
}
else
{
lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v_c_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; 
v___x_132_ = lean_nat_mul(v_c_u2080_x27_116_, v_c_u2080_x27_116_);
lean_dec(v_c_u2080_x27_116_);
v___x_133_ = lean_nat_mod(v___x_132_, v_m_x27_115_);
lean_dec(v_m_x27_115_);
lean_dec(v___x_132_);
v_c_134_ = l_Lean_mkRawNatLit(v___x_133_);
v___x_135_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__27, &lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__27_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__27);
v___x_136_ = l_Lean_Expr_app___override(v___x_135_, v_a_70_);
v___x_137_ = l_Lean_Expr_app___override(v___x_136_, v_b_u2080_72_);
v___x_138_ = l_Lean_Expr_app___override(v___x_137_, v_m_71_);
v___x_139_ = l_Lean_Expr_app___override(v___x_138_, v_c_u2080_73_);
v___x_140_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_140_, 0, v_c_134_);
lean_ctor_set(v___x_140_, 1, v___x_139_);
return v___x_140_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___boxed(lean_object* v_depth_141_, lean_object* v_a_142_, lean_object* v_m_143_, lean_object* v_b_u2080_144_, lean_object* v_c_u2080_145_, lean_object* v_b_146_, lean_object* v_p_147_){
_start:
{
lean_object* v_res_148_; 
v_res_148_ = lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg(v_depth_141_, v_a_142_, v_m_143_, v_b_u2080_144_, v_c_u2080_145_, v_b_146_, v_p_147_);
lean_dec(v_depth_141_);
return v_res_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go(lean_object* v_depth_149_, lean_object* v_a_150_, lean_object* v_m_151_, lean_object* v_b_u2080_152_, lean_object* v_c_u2080_153_, lean_object* v_b_154_, lean_object* v_p_155_, lean_object* v_hp_156_){
_start:
{
lean_object* v___x_157_; 
v___x_157_ = lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg(v_depth_149_, v_a_150_, v_m_151_, v_b_u2080_152_, v_c_u2080_153_, v_b_154_, v_p_155_);
return v___x_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___boxed(lean_object* v_depth_158_, lean_object* v_a_159_, lean_object* v_m_160_, lean_object* v_b_u2080_161_, lean_object* v_c_u2080_162_, lean_object* v_b_163_, lean_object* v_p_164_, lean_object* v_hp_165_){
_start:
{
lean_object* v_res_166_; 
v_res_166_ = lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go(v_depth_158_, v_a_159_, v_m_160_, v_b_u2080_161_, v_c_u2080_162_, v_b_163_, v_p_164_, v_hp_165_);
lean_dec(v_depth_158_);
return v_res_166_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__1(void){
_start:
{
lean_object* v___x_169_; lean_object* v___x_170_; 
v___x_169_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__0));
v___x_170_ = l_Lean_Expr_lit___override(v___x_169_);
return v___x_170_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__4(void){
_start:
{
lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; 
v___x_178_ = lean_box(0);
v___x_179_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__3));
v___x_180_ = l_Lean_Expr_const___override(v___x_179_, v___x_178_);
return v___x_180_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__7(void){
_start:
{
lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; 
v___x_187_ = lean_box(0);
v___x_188_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__6));
v___x_189_ = l_Lean_Expr_const___override(v___x_188_, v___x_187_);
return v___x_189_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__10(void){
_start:
{
lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; 
v___x_196_ = lean_box(0);
v___x_197_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__9));
v___x_198_ = l_Lean_Expr_const___override(v___x_197_, v___x_196_);
return v___x_198_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__12(void){
_start:
{
lean_object* v___x_201_; lean_object* v___x_202_; 
v___x_201_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__11));
v___x_202_ = l_Lean_Expr_lit___override(v___x_201_);
return v___x_202_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__15(void){
_start:
{
lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; 
v___x_209_ = lean_box(0);
v___x_210_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__14));
v___x_211_ = l_Lean_Expr_const___override(v___x_210_, v___x_209_);
return v___x_211_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__18(void){
_start:
{
lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; 
v___x_218_ = lean_box(0);
v___x_219_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__17));
v___x_220_ = l_Lean_Expr_const___override(v___x_219_, v___x_218_);
return v___x_220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod(lean_object* v_a_221_, lean_object* v_b_222_, lean_object* v_m_223_){
_start:
{
lean_object* v___x_224_; lean_object* v___x_225_; uint8_t v___x_226_; 
v___x_224_ = lp_batteries_Lean_Expr_natLit_x21(v_b_222_);
v___x_225_ = lean_unsigned_to_nat(0u);
v___x_226_ = lean_nat_dec_eq(v___x_224_, v___x_225_);
if (v___x_226_ == 0)
{
lean_object* v___x_227_; uint8_t v___x_228_; 
v___x_227_ = lean_unsigned_to_nat(1u);
v___x_228_ = lean_nat_dec_eq(v___x_224_, v___x_227_);
if (v___x_228_ == 0)
{
lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v_c_u2080_232_; lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v_fst_245_; lean_object* v_snd_246_; lean_object* v___x_248_; uint8_t v_isShared_249_; uint8_t v_isSharedCheck_259_; 
v___x_229_ = lp_batteries_Lean_Expr_natLit_x21(v_a_221_);
v___x_230_ = lp_batteries_Lean_Expr_natLit_x21(v_m_223_);
v___x_231_ = lean_nat_mod(v___x_229_, v___x_230_);
lean_dec(v___x_230_);
lean_dec(v___x_229_);
v_c_u2080_232_ = l_Lean_mkRawNatLit(v___x_231_);
v___x_233_ = lean_nat_log2(v___x_224_);
lean_dec(v___x_224_);
v___x_234_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__1);
v___x_235_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__8, &lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__8_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__8);
v___x_236_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__11, &lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__11_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__11);
v___x_237_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__14, &lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__14_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg___closed__14);
lean_inc_ref_n(v_a_221_, 2);
v___x_238_ = l_Lean_Expr_app___override(v___x_237_, v_a_221_);
v___x_239_ = l_Lean_Expr_app___override(v___x_238_, v___x_234_);
v___x_240_ = l_Lean_Expr_app___override(v___x_236_, v___x_239_);
lean_inc_ref_n(v_m_223_, 2);
v___x_241_ = l_Lean_Expr_app___override(v___x_240_, v_m_223_);
v___x_242_ = l_Lean_Expr_app___override(v___x_235_, v___x_241_);
lean_inc_ref(v_c_u2080_232_);
v___x_243_ = l_Lean_Expr_app___override(v___x_242_, v_c_u2080_232_);
lean_inc_ref(v_b_222_);
v___x_244_ = lp_mathlib___private_Mathlib_Tactic_NormNum_PowMod_0__Mathlib_Meta_NormNum_evalNatPowMod_go___redArg(v___x_233_, v_a_221_, v_m_223_, v___x_234_, v_c_u2080_232_, v_b_222_, v___x_243_);
lean_dec(v___x_233_);
v_fst_245_ = lean_ctor_get(v___x_244_, 0);
v_snd_246_ = lean_ctor_get(v___x_244_, 1);
v_isSharedCheck_259_ = !lean_is_exclusive(v___x_244_);
if (v_isSharedCheck_259_ == 0)
{
v___x_248_ = v___x_244_;
v_isShared_249_ = v_isSharedCheck_259_;
goto v_resetjp_247_;
}
else
{
lean_inc(v_snd_246_);
lean_inc(v_fst_245_);
lean_dec(v___x_244_);
v___x_248_ = lean_box(0);
v_isShared_249_ = v_isSharedCheck_259_;
goto v_resetjp_247_;
}
v_resetjp_247_:
{
lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_257_; 
v___x_250_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__4, &lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__4_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__4);
v___x_251_ = l_Lean_Expr_app___override(v___x_250_, v_a_221_);
v___x_252_ = l_Lean_Expr_app___override(v___x_251_, v_m_223_);
v___x_253_ = l_Lean_Expr_app___override(v___x_252_, v_b_222_);
lean_inc(v_fst_245_);
v___x_254_ = l_Lean_Expr_app___override(v___x_253_, v_fst_245_);
v___x_255_ = l_Lean_Expr_app___override(v___x_254_, v_snd_246_);
if (v_isShared_249_ == 0)
{
lean_ctor_set(v___x_248_, 1, v___x_255_);
v___x_257_ = v___x_248_;
goto v_reusejp_256_;
}
else
{
lean_object* v_reuseFailAlloc_258_; 
v_reuseFailAlloc_258_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_258_, 0, v_fst_245_);
lean_ctor_set(v_reuseFailAlloc_258_, 1, v___x_255_);
v___x_257_ = v_reuseFailAlloc_258_;
goto v_reusejp_256_;
}
v_reusejp_256_:
{
return v___x_257_;
}
}
}
else
{
lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v_c_263_; lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; 
lean_dec(v___x_224_);
lean_dec_ref(v_b_222_);
v___x_260_ = lp_batteries_Lean_Expr_natLit_x21(v_a_221_);
v___x_261_ = lp_batteries_Lean_Expr_natLit_x21(v_m_223_);
v___x_262_ = lean_nat_mod(v___x_260_, v___x_261_);
lean_dec(v___x_261_);
lean_dec(v___x_260_);
v_c_263_ = l_Lean_mkRawNatLit(v___x_262_);
v___x_264_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__7, &lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__7_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__7);
v___x_265_ = l_Lean_Expr_app___override(v___x_264_, v_a_221_);
v___x_266_ = l_Lean_Expr_app___override(v___x_265_, v_m_223_);
v___x_267_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_267_, 0, v_c_263_);
lean_ctor_set(v___x_267_, 1, v___x_266_);
return v___x_267_;
}
}
else
{
lean_object* v___x_268_; uint8_t v___x_269_; 
lean_dec(v___x_224_);
lean_dec_ref(v_b_222_);
v___x_268_ = lp_batteries_Lean_Expr_natLit_x21(v_m_223_);
lean_dec_ref(v_m_223_);
v___x_269_ = lean_nat_dec_eq(v___x_268_, v___x_225_);
if (v___x_269_ == 0)
{
lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v_m_x27_272_; lean_object* v___x_273_; uint8_t v___x_274_; 
v___x_270_ = lean_unsigned_to_nat(1u);
v___x_271_ = lean_nat_sub(v___x_268_, v___x_270_);
lean_dec(v___x_268_);
v_m_x27_272_ = l_Lean_mkRawNatLit(v___x_271_);
v___x_273_ = lp_batteries_Lean_Expr_natLit_x21(v_m_x27_272_);
lean_dec_ref(v_m_x27_272_);
v___x_274_ = lean_nat_dec_eq(v___x_273_, v___x_225_);
if (v___x_274_ == 0)
{
lean_object* v___x_275_; lean_object* v_m_x27_x27_276_; lean_object* v___x_277_; lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; 
v___x_275_ = lean_nat_sub(v___x_273_, v___x_270_);
lean_dec(v___x_273_);
v_m_x27_x27_276_ = l_Lean_mkRawNatLit(v___x_275_);
v___x_277_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__1);
v___x_278_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__10, &lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__10_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__10);
v___x_279_ = l_Lean_Expr_app___override(v___x_278_, v_a_221_);
v___x_280_ = l_Lean_Expr_app___override(v___x_279_, v_m_x27_x27_276_);
v___x_281_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_281_, 0, v___x_277_);
lean_ctor_set(v___x_281_, 1, v___x_280_);
return v___x_281_;
}
else
{
lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; lean_object* v___x_285_; 
lean_dec(v___x_273_);
v___x_282_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__12, &lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__12_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__12);
v___x_283_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__15, &lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__15_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__15);
v___x_284_ = l_Lean_Expr_app___override(v___x_283_, v_a_221_);
v___x_285_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_285_, 0, v___x_282_);
lean_ctor_set(v___x_285_, 1, v___x_284_);
return v___x_285_;
}
}
else
{
lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v___x_289_; 
lean_dec(v___x_268_);
v___x_286_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__1);
v___x_287_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__18, &lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__18_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod___closed__18);
v___x_288_ = l_Lean_Expr_app___override(v___x_287_, v_a_221_);
v___x_289_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_289_, 0, v___x_286_);
lean_ctor_set(v___x_289_, 1, v___x_288_);
return v___x_289_;
}
}
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Pow(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NormNum_PowMod(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Pow(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_NormNum_PowMod(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Tactic_NormNum_Pow(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_NormNum_PowMod(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_NormNum_Pow(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NormNum_PowMod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_NormNum_PowMod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_NormNum_PowMod(builtin);
}
#ifdef __cplusplus
}
#endif
