// Lean compiler output
// Module: Mathlib.Tactic.NormNum.Eq
// Imports: public import Init public meta import Init public import Mathlib.Tactic.NormNum.Inv
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
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Meta_whnfR(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Qq_inferTypeQ_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Level_succ___override(lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_derive(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_Lean_Expr_natLit_x21(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfAddMonoidWithOne_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t lean_int_dec_eq(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toInt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_instDecidableEqRat_decEq(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toNNRat_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_Qq_Qq_synthInstanceQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__1;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__2;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Bool"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__3_value),LEAN_SCALAR_PTR_LITERAL(250, 44, 198, 216, 184, 195, 199, 178)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__5;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "false"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__3_value),LEAN_SCALAR_PTR_LITERAL(250, 44, 198, 216, 184, 195, 199, 178)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__7_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(117, 151, 161, 190, 111, 237, 188, 218)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__7_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__8;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "refl"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__10_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__9_value),LEAN_SCALAR_PTR_LITERAL(72, 6, 107, 181, 0, 125, 21, 187)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__10_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__11;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__12;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__13;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__14_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__15_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "NormNum"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__16_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "isInt_eq_false"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__17_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__18_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__18_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__15_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__18_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__18_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__18_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__17_value),LEAN_SCALAR_PTR_LITERAL(90, 170, 206, 213, 126, 161, 148, 211)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__18_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__19_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "isInt_eq_true"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__21_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__22_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__22_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__22_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__15_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__22_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__22_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__22_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__21_value),LEAN_SCALAR_PTR_LITERAL(148, 89, 179, 154, 79, 181, 66, 26)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__22_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "isNNRat_eq_false"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__15_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__1_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(187, 120, 205, 210, 205, 122, 67, 148)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "DivisionSemiring"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "toSemiring"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 208, 64, 71, 63, 26, 215, 130)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__3_value),LEAN_SCALAR_PTR_LITERAL(241, 35, 131, 210, 203, 216, 146, 177)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "isNNRat_eq_true"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__6_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__15_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__6_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__6_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(71, 14, 12, 242, 236, 35, 176, 151)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "isRat_eq_false"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__15_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__1_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(5, 6, 242, 144, 92, 114, 186, 93)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "DivisionRing"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "toRing"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(34, 214, 17, 155, 7, 71, 232, 190)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__3_value),LEAN_SCALAR_PTR_LITERAL(196, 15, 37, 9, 106, 139, 236, 93)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "isRat_eq_true"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__6_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__15_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__6_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__6_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(234, 79, 220, 207, 47, 83, 31, 155)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalEq_spec__0___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalEq_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalEq_spec__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalEq_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "eq_of_false"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__15_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__2_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(156, 79, 170, 229, 94, 216, 39, 216)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "ne_of_false_of_true"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__5_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__15_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__5_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__5_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(190, 40, 133, 63, 98, 231, 118, 131)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__6;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "ne_of_true_of_false"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__8_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__15_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__8_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__8_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(244, 235, 97, 220, 172, 137, 196, 66)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__9;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "eq_of_true"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__11_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__15_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__11_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__11_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(33, 116, 170, 144, 66, 155, 149, 233)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__11_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__12;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "isNat_eq_false"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__14_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__15_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__14_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__14_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__14_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(6, 14, 210, 70, 228, 250, 149, 222)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "isNat_eq_true"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__16_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__15_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__16_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__16_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__16_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(112, 119, 25, 97, 161, 198, 135, 24)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(34, 214, 17, 155, 7, 71, 232, 190)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__17_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalEq___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "evalEq"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalEq___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__15_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___closed__2_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___closed__1_value),LEAN_SCALAR_PTR_LITERAL(12, 133, 142, 66, 241, 131, 57, 225)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalEq___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalEq___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___closed__2_value),LEAN_SCALAR_PTR_LITERAL(1, 1, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalEq___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___closed__3_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalEq = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalEq___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm_spec__0_spec__0(lean_object* v_msgData_1_, lean_object* v___y_2_, lean_object* v___y_3_, lean_object* v___y_4_, lean_object* v___y_5_){
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm_spec__0_spec__0___boxed(lean_object* v_msgData_16_, lean_object* v___y_17_, lean_object* v___y_18_, lean_object* v___y_19_, lean_object* v___y_20_, lean_object* v___y_21_){
_start:
{
lean_object* v_res_22_; 
v_res_22_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm_spec__0_spec__0(v_msgData_16_, v___y_17_, v___y_18_, v___y_19_, v___y_20_);
lean_dec(v___y_20_);
lean_dec_ref(v___y_19_);
lean_dec(v___y_18_);
lean_dec_ref(v___y_17_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm_spec__0___redArg(lean_object* v_msg_23_, lean_object* v___y_24_, lean_object* v___y_25_, lean_object* v___y_26_, lean_object* v___y_27_){
_start:
{
lean_object* v_ref_29_; lean_object* v___x_30_; lean_object* v_a_31_; lean_object* v___x_33_; uint8_t v_isShared_34_; uint8_t v_isSharedCheck_39_; 
v_ref_29_ = lean_ctor_get(v___y_26_, 5);
v___x_30_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm_spec__0_spec__0(v_msg_23_, v___y_24_, v___y_25_, v___y_26_, v___y_27_);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm_spec__0___redArg___boxed(lean_object* v_msg_40_, lean_object* v___y_41_, lean_object* v___y_42_, lean_object* v___y_43_, lean_object* v___y_44_, lean_object* v___y_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm_spec__0___redArg(v_msg_40_, v___y_41_, v___y_42_, v___y_43_, v___y_44_);
lean_dec(v___y_44_);
lean_dec_ref(v___y_43_);
lean_dec(v___y_42_);
lean_dec_ref(v___y_41_);
return v_res_46_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__1(void){
_start:
{
lean_object* v___x_48_; lean_object* v___x_49_; 
v___x_48_ = lean_box(0);
v___x_49_ = l_Lean_Level_succ___override(v___x_48_);
return v___x_49_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__2(void){
_start:
{
lean_object* v___x_50_; lean_object* v___x_51_; lean_object* v___x_52_; 
v___x_50_ = lean_box(0);
v___x_51_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__1, &lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__1);
v___x_52_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_52_, 0, v___x_51_);
lean_ctor_set(v___x_52_, 1, v___x_50_);
return v___x_52_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__5(void){
_start:
{
lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; 
v___x_56_ = lean_box(0);
v___x_57_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__4));
v___x_58_ = l_Lean_Expr_const___override(v___x_57_, v___x_56_);
return v___x_58_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__8(void){
_start:
{
lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; 
v___x_63_ = lean_box(0);
v___x_64_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__7));
v___x_65_ = l_Lean_Expr_const___override(v___x_64_, v___x_63_);
return v___x_65_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__11(void){
_start:
{
lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; 
v___x_70_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__2, &lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__2);
v___x_71_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__10));
v___x_72_ = l_Lean_Expr_const___override(v___x_71_, v___x_70_);
return v___x_72_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__12(void){
_start:
{
lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; 
v___x_73_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__5, &lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__5);
v___x_74_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__11, &lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__11_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__11);
v___x_75_ = l_Lean_Expr_app___override(v___x_74_, v___x_73_);
return v___x_75_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__13(void){
_start:
{
lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; 
v___x_76_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__8, &lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__8_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__8);
v___x_77_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__12, &lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__12_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__12);
v___x_78_ = l_Lean_Expr_app___override(v___x_77_, v___x_76_);
return v___x_78_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20(void){
_start:
{
lean_object* v___x_89_; lean_object* v___x_90_; 
v___x_89_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__19));
v___x_90_ = l_Lean_stringToMessageData(v___x_89_);
return v___x_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg(lean_object* v_u_97_, lean_object* v_00_u03b1_98_, lean_object* v_a_99_, lean_object* v_b_100_, lean_object* v_ra_101_, lean_object* v_rb_102_, lean_object* v_r_u03b1_103_, lean_object* v_a_104_, lean_object* v_a_105_, lean_object* v_a_106_, lean_object* v_a_107_){
_start:
{
lean_object* v___y_110_; lean_object* v___y_111_; lean_object* v___y_112_; lean_object* v_a_113_; lean_object* v_a_177_; lean_object* v___x_194_; 
lean_inc_ref(v_r_u03b1_103_);
lean_inc_ref(v_a_99_);
lean_inc_ref(v_00_u03b1_98_);
lean_inc(v_u_97_);
v___x_194_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toInt(v_u_97_, v_00_u03b1_98_, v_a_99_, v_r_u03b1_103_, v_ra_101_);
if (lean_obj_tag(v___x_194_) == 0)
{
lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v_a_197_; lean_object* v___x_199_; uint8_t v_isShared_200_; uint8_t v_isSharedCheck_204_; 
lean_dec_ref(v_r_u03b1_103_);
lean_dec_ref(v_rb_102_);
lean_dec_ref(v_b_100_);
lean_dec_ref(v_a_99_);
lean_dec_ref(v_00_u03b1_98_);
lean_dec(v_u_97_);
v___x_195_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20, &lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20);
v___x_196_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm_spec__0___redArg(v___x_195_, v_a_104_, v_a_105_, v_a_106_, v_a_107_);
v_a_197_ = lean_ctor_get(v___x_196_, 0);
v_isSharedCheck_204_ = !lean_is_exclusive(v___x_196_);
if (v_isSharedCheck_204_ == 0)
{
v___x_199_ = v___x_196_;
v_isShared_200_ = v_isSharedCheck_204_;
goto v_resetjp_198_;
}
else
{
lean_inc(v_a_197_);
lean_dec(v___x_196_);
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
else
{
lean_object* v_val_205_; 
v_val_205_ = lean_ctor_get(v___x_194_, 0);
lean_inc(v_val_205_);
lean_dec_ref_known(v___x_194_, 1);
v_a_177_ = v_val_205_;
goto v___jp_176_;
}
v___jp_109_:
{
lean_object* v_snd_114_; lean_object* v_fst_115_; lean_object* v_fst_116_; lean_object* v_snd_117_; lean_object* v___x_119_; uint8_t v_isShared_120_; uint8_t v_isSharedCheck_175_; 
v_snd_114_ = lean_ctor_get(v_a_113_, 1);
lean_inc(v_snd_114_);
v_fst_115_ = lean_ctor_get(v_a_113_, 0);
lean_inc(v_fst_115_);
lean_dec_ref(v_a_113_);
v_fst_116_ = lean_ctor_get(v_snd_114_, 0);
v_snd_117_ = lean_ctor_get(v_snd_114_, 1);
v_isSharedCheck_175_ = !lean_is_exclusive(v_snd_114_);
if (v_isSharedCheck_175_ == 0)
{
v___x_119_ = v_snd_114_;
v_isShared_120_ = v_isSharedCheck_175_;
goto v_resetjp_118_;
}
else
{
lean_inc(v_snd_117_);
lean_inc(v_fst_116_);
lean_dec(v_snd_114_);
v___x_119_ = lean_box(0);
v_isShared_120_ = v_isSharedCheck_175_;
goto v_resetjp_118_;
}
v_resetjp_118_:
{
uint8_t v___x_121_; 
v___x_121_ = lean_int_dec_eq(v___y_112_, v_fst_115_);
lean_dec(v_fst_115_);
lean_dec(v___y_112_);
if (v___x_121_ == 0)
{
lean_object* v___x_122_; 
lean_inc_ref(v_r_u03b1_103_);
lean_inc_ref(v_00_u03b1_98_);
lean_inc(v_u_97_);
v___x_122_ = lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfRing_x3f(v_u_97_, v_00_u03b1_98_, v_r_u03b1_103_, v_a_104_, v_a_105_, v_a_106_, v_a_107_);
if (lean_obj_tag(v___x_122_) == 0)
{
lean_object* v_a_123_; lean_object* v___x_125_; uint8_t v_isShared_126_; uint8_t v_isSharedCheck_151_; 
v_a_123_ = lean_ctor_get(v___x_122_, 0);
v_isSharedCheck_151_ = !lean_is_exclusive(v___x_122_);
if (v_isSharedCheck_151_ == 0)
{
v___x_125_ = v___x_122_;
v_isShared_126_ = v_isSharedCheck_151_;
goto v_resetjp_124_;
}
else
{
lean_inc(v_a_123_);
lean_dec(v___x_122_);
v___x_125_ = lean_box(0);
v_isShared_126_ = v_isSharedCheck_151_;
goto v_resetjp_124_;
}
v_resetjp_124_:
{
if (lean_obj_tag(v_a_123_) == 1)
{
lean_object* v_val_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_132_; 
v_val_127_ = lean_ctor_get(v_a_123_, 0);
lean_inc(v_val_127_);
lean_dec_ref_known(v_a_123_, 1);
v___x_128_ = lean_box(0);
v___x_129_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__13, &lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__13_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__13);
v___x_130_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__18));
if (v_isShared_120_ == 0)
{
lean_ctor_set_tag(v___x_119_, 1);
lean_ctor_set(v___x_119_, 1, v___x_128_);
lean_ctor_set(v___x_119_, 0, v_u_97_);
v___x_132_ = v___x_119_;
goto v_reusejp_131_;
}
else
{
lean_object* v_reuseFailAlloc_148_; 
v_reuseFailAlloc_148_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_148_, 0, v_u_97_);
lean_ctor_set(v_reuseFailAlloc_148_, 1, v___x_128_);
v___x_132_ = v_reuseFailAlloc_148_;
goto v_reusejp_131_;
}
v_reusejp_131_:
{
lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_146_; 
v___x_133_ = l_Lean_Expr_const___override(v___x_130_, v___x_132_);
v___x_134_ = l_Lean_Expr_app___override(v___x_133_, v_00_u03b1_98_);
v___x_135_ = l_Lean_Expr_app___override(v___x_134_, v_r_u03b1_103_);
v___x_136_ = l_Lean_Expr_app___override(v___x_135_, v_val_127_);
v___x_137_ = l_Lean_Expr_app___override(v___x_136_, v_a_99_);
v___x_138_ = l_Lean_Expr_app___override(v___x_137_, v_b_100_);
v___x_139_ = l_Lean_Expr_app___override(v___x_138_, v___y_110_);
v___x_140_ = l_Lean_Expr_app___override(v___x_139_, v_fst_116_);
v___x_141_ = l_Lean_Expr_app___override(v___x_140_, v___y_111_);
v___x_142_ = l_Lean_Expr_app___override(v___x_141_, v_snd_117_);
v___x_143_ = l_Lean_Expr_app___override(v___x_142_, v___x_129_);
v___x_144_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_144_, 0, v___x_143_);
lean_ctor_set_uint8(v___x_144_, sizeof(void*)*1, v___x_121_);
if (v_isShared_126_ == 0)
{
lean_ctor_set(v___x_125_, 0, v___x_144_);
v___x_146_ = v___x_125_;
goto v_reusejp_145_;
}
else
{
lean_object* v_reuseFailAlloc_147_; 
v_reuseFailAlloc_147_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_147_, 0, v___x_144_);
v___x_146_ = v_reuseFailAlloc_147_;
goto v_reusejp_145_;
}
v_reusejp_145_:
{
return v___x_146_;
}
}
}
else
{
lean_object* v___x_149_; lean_object* v___x_150_; 
lean_del_object(v___x_125_);
lean_dec(v_a_123_);
lean_del_object(v___x_119_);
lean_dec(v_snd_117_);
lean_dec(v_fst_116_);
lean_dec_ref(v___y_111_);
lean_dec_ref(v___y_110_);
lean_dec_ref(v_r_u03b1_103_);
lean_dec_ref(v_b_100_);
lean_dec_ref(v_a_99_);
lean_dec_ref(v_00_u03b1_98_);
lean_dec(v_u_97_);
v___x_149_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20, &lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20);
v___x_150_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm_spec__0___redArg(v___x_149_, v_a_104_, v_a_105_, v_a_106_, v_a_107_);
return v___x_150_;
}
}
}
else
{
lean_object* v_a_152_; lean_object* v___x_154_; uint8_t v_isShared_155_; uint8_t v_isSharedCheck_159_; 
lean_del_object(v___x_119_);
lean_dec(v_snd_117_);
lean_dec(v_fst_116_);
lean_dec_ref(v___y_111_);
lean_dec_ref(v___y_110_);
lean_dec_ref(v_r_u03b1_103_);
lean_dec_ref(v_b_100_);
lean_dec_ref(v_a_99_);
lean_dec_ref(v_00_u03b1_98_);
lean_dec(v_u_97_);
v_a_152_ = lean_ctor_get(v___x_122_, 0);
v_isSharedCheck_159_ = !lean_is_exclusive(v___x_122_);
if (v_isSharedCheck_159_ == 0)
{
v___x_154_ = v___x_122_;
v_isShared_155_ = v_isSharedCheck_159_;
goto v_resetjp_153_;
}
else
{
lean_inc(v_a_152_);
lean_dec(v___x_122_);
v___x_154_ = lean_box(0);
v_isShared_155_ = v_isSharedCheck_159_;
goto v_resetjp_153_;
}
v_resetjp_153_:
{
lean_object* v___x_157_; 
if (v_isShared_155_ == 0)
{
v___x_157_ = v___x_154_;
goto v_reusejp_156_;
}
else
{
lean_object* v_reuseFailAlloc_158_; 
v_reuseFailAlloc_158_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_158_, 0, v_a_152_);
v___x_157_ = v_reuseFailAlloc_158_;
goto v_reusejp_156_;
}
v_reusejp_156_:
{
return v___x_157_;
}
}
}
}
else
{
lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_163_; 
lean_dec(v_fst_116_);
v___x_160_ = lean_box(0);
v___x_161_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__22));
if (v_isShared_120_ == 0)
{
lean_ctor_set_tag(v___x_119_, 1);
lean_ctor_set(v___x_119_, 1, v___x_160_);
lean_ctor_set(v___x_119_, 0, v_u_97_);
v___x_163_ = v___x_119_;
goto v_reusejp_162_;
}
else
{
lean_object* v_reuseFailAlloc_174_; 
v_reuseFailAlloc_174_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_174_, 0, v_u_97_);
lean_ctor_set(v_reuseFailAlloc_174_, 1, v___x_160_);
v___x_163_ = v_reuseFailAlloc_174_;
goto v_reusejp_162_;
}
v_reusejp_162_:
{
lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; 
v___x_164_ = l_Lean_Expr_const___override(v___x_161_, v___x_163_);
v___x_165_ = l_Lean_Expr_app___override(v___x_164_, v_00_u03b1_98_);
v___x_166_ = l_Lean_Expr_app___override(v___x_165_, v_r_u03b1_103_);
v___x_167_ = l_Lean_Expr_app___override(v___x_166_, v_a_99_);
v___x_168_ = l_Lean_Expr_app___override(v___x_167_, v_b_100_);
v___x_169_ = l_Lean_Expr_app___override(v___x_168_, v___y_110_);
v___x_170_ = l_Lean_Expr_app___override(v___x_169_, v___y_111_);
v___x_171_ = l_Lean_Expr_app___override(v___x_170_, v_snd_117_);
v___x_172_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_172_, 0, v___x_171_);
lean_ctor_set_uint8(v___x_172_, sizeof(void*)*1, v___x_121_);
v___x_173_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_173_, 0, v___x_172_);
return v___x_173_;
}
}
}
}
v___jp_176_:
{
lean_object* v_snd_178_; lean_object* v_fst_179_; lean_object* v_fst_180_; lean_object* v_snd_181_; lean_object* v___x_182_; 
v_snd_178_ = lean_ctor_get(v_a_177_, 1);
lean_inc(v_snd_178_);
v_fst_179_ = lean_ctor_get(v_a_177_, 0);
lean_inc(v_fst_179_);
lean_dec_ref(v_a_177_);
v_fst_180_ = lean_ctor_get(v_snd_178_, 0);
lean_inc(v_fst_180_);
v_snd_181_ = lean_ctor_get(v_snd_178_, 1);
lean_inc(v_snd_181_);
lean_dec(v_snd_178_);
lean_inc_ref(v_r_u03b1_103_);
lean_inc_ref(v_b_100_);
lean_inc_ref(v_00_u03b1_98_);
lean_inc(v_u_97_);
v___x_182_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toInt(v_u_97_, v_00_u03b1_98_, v_b_100_, v_r_u03b1_103_, v_rb_102_);
if (lean_obj_tag(v___x_182_) == 0)
{
lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v_a_185_; lean_object* v___x_187_; uint8_t v_isShared_188_; uint8_t v_isSharedCheck_192_; 
lean_dec(v_snd_181_);
lean_dec(v_fst_180_);
lean_dec(v_fst_179_);
lean_dec_ref(v_r_u03b1_103_);
lean_dec_ref(v_b_100_);
lean_dec_ref(v_a_99_);
lean_dec_ref(v_00_u03b1_98_);
lean_dec(v_u_97_);
v___x_183_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20, &lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20);
v___x_184_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm_spec__0___redArg(v___x_183_, v_a_104_, v_a_105_, v_a_106_, v_a_107_);
v_a_185_ = lean_ctor_get(v___x_184_, 0);
v_isSharedCheck_192_ = !lean_is_exclusive(v___x_184_);
if (v_isSharedCheck_192_ == 0)
{
v___x_187_ = v___x_184_;
v_isShared_188_ = v_isSharedCheck_192_;
goto v_resetjp_186_;
}
else
{
lean_inc(v_a_185_);
lean_dec(v___x_184_);
v___x_187_ = lean_box(0);
v_isShared_188_ = v_isSharedCheck_192_;
goto v_resetjp_186_;
}
v_resetjp_186_:
{
lean_object* v___x_190_; 
if (v_isShared_188_ == 0)
{
v___x_190_ = v___x_187_;
goto v_reusejp_189_;
}
else
{
lean_object* v_reuseFailAlloc_191_; 
v_reuseFailAlloc_191_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_191_, 0, v_a_185_);
v___x_190_ = v_reuseFailAlloc_191_;
goto v_reusejp_189_;
}
v_reusejp_189_:
{
return v___x_190_;
}
}
}
else
{
lean_object* v_val_193_; 
v_val_193_ = lean_ctor_get(v___x_182_, 0);
lean_inc(v_val_193_);
lean_dec_ref_known(v___x_182_, 1);
v___y_110_ = v_fst_180_;
v___y_111_ = v_snd_181_;
v___y_112_ = v_fst_179_;
v_a_113_ = v_val_193_;
goto v___jp_109_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___boxed(lean_object* v_u_206_, lean_object* v_00_u03b1_207_, lean_object* v_a_208_, lean_object* v_b_209_, lean_object* v_ra_210_, lean_object* v_rb_211_, lean_object* v_r_u03b1_212_, lean_object* v_a_213_, lean_object* v_a_214_, lean_object* v_a_215_, lean_object* v_a_216_, lean_object* v_a_217_){
_start:
{
lean_object* v_res_218_; 
v_res_218_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg(v_u_206_, v_00_u03b1_207_, v_a_208_, v_b_209_, v_ra_210_, v_rb_211_, v_r_u03b1_212_, v_a_213_, v_a_214_, v_a_215_, v_a_216_);
lean_dec(v_a_216_);
lean_dec_ref(v_a_215_);
lean_dec(v_a_214_);
lean_dec_ref(v_a_213_);
return v_res_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm(lean_object* v_v_219_, lean_object* v_00_u03b2_220_, lean_object* v_e_221_, lean_object* v_u_222_, lean_object* v_00_u03b1_223_, lean_object* v_a_224_, lean_object* v_b_225_, lean_object* v_ra_226_, lean_object* v_rb_227_, lean_object* v_r_u03b1_228_, lean_object* v_a_229_, lean_object* v_a_230_, lean_object* v_a_231_, lean_object* v_a_232_){
_start:
{
lean_object* v___x_234_; 
v___x_234_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg(v_u_222_, v_00_u03b1_223_, v_a_224_, v_b_225_, v_ra_226_, v_rb_227_, v_r_u03b1_228_, v_a_229_, v_a_230_, v_a_231_, v_a_232_);
return v___x_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___boxed(lean_object* v_v_235_, lean_object* v_00_u03b2_236_, lean_object* v_e_237_, lean_object* v_u_238_, lean_object* v_00_u03b1_239_, lean_object* v_a_240_, lean_object* v_b_241_, lean_object* v_ra_242_, lean_object* v_rb_243_, lean_object* v_r_u03b1_244_, lean_object* v_a_245_, lean_object* v_a_246_, lean_object* v_a_247_, lean_object* v_a_248_, lean_object* v_a_249_){
_start:
{
lean_object* v_res_250_; 
v_res_250_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm(v_v_235_, v_00_u03b2_236_, v_e_237_, v_u_238_, v_00_u03b1_239_, v_a_240_, v_b_241_, v_ra_242_, v_rb_243_, v_r_u03b1_244_, v_a_245_, v_a_246_, v_a_247_, v_a_248_);
lean_dec(v_a_248_);
lean_dec_ref(v_a_247_);
lean_dec(v_a_246_);
lean_dec_ref(v_a_245_);
lean_dec_ref(v_e_237_);
lean_dec_ref(v_00_u03b2_236_);
lean_dec(v_v_235_);
return v_res_250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm_spec__0(lean_object* v_00_u03b1_251_, lean_object* v_msg_252_, lean_object* v___y_253_, lean_object* v___y_254_, lean_object* v___y_255_, lean_object* v___y_256_){
_start:
{
lean_object* v___x_258_; 
v___x_258_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm_spec__0___redArg(v_msg_252_, v___y_253_, v___y_254_, v___y_255_, v___y_256_);
return v___x_258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm_spec__0___boxed(lean_object* v_00_u03b1_259_, lean_object* v_msg_260_, lean_object* v___y_261_, lean_object* v___y_262_, lean_object* v___y_263_, lean_object* v___y_264_, lean_object* v___y_265_){
_start:
{
lean_object* v_res_266_; 
v_res_266_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm_spec__0(v_00_u03b1_259_, v_msg_260_, v___y_261_, v___y_262_, v___y_263_, v___y_264_);
lean_dec(v___y_264_);
lean_dec_ref(v___y_263_);
lean_dec(v___y_262_);
lean_dec_ref(v___y_261_);
return v_res_266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg(lean_object* v_u_284_, lean_object* v_00_u03b1_285_, lean_object* v_a_286_, lean_object* v_b_287_, lean_object* v_ra_288_, lean_object* v_rb_289_, lean_object* v_ds_u03b1_290_, lean_object* v_a_291_, lean_object* v_a_292_, lean_object* v_a_293_, lean_object* v_a_294_){
_start:
{
lean_object* v___y_297_; lean_object* v___y_298_; lean_object* v___y_299_; lean_object* v___y_300_; lean_object* v_a_301_; lean_object* v_a_378_; lean_object* v___x_397_; 
lean_inc_ref(v_ds_u03b1_290_);
lean_inc_ref(v_a_286_);
lean_inc_ref(v_00_u03b1_285_);
lean_inc(v_u_284_);
v___x_397_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toNNRat_x27(v_u_284_, v_00_u03b1_285_, v_a_286_, v_ds_u03b1_290_, v_ra_288_);
if (lean_obj_tag(v___x_397_) == 0)
{
lean_object* v___x_398_; lean_object* v___x_399_; lean_object* v_a_400_; lean_object* v___x_402_; uint8_t v_isShared_403_; uint8_t v_isSharedCheck_407_; 
lean_dec_ref(v_ds_u03b1_290_);
lean_dec_ref(v_rb_289_);
lean_dec_ref(v_b_287_);
lean_dec_ref(v_a_286_);
lean_dec_ref(v_00_u03b1_285_);
lean_dec(v_u_284_);
v___x_398_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20, &lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20);
v___x_399_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm_spec__0___redArg(v___x_398_, v_a_291_, v_a_292_, v_a_293_, v_a_294_);
v_a_400_ = lean_ctor_get(v___x_399_, 0);
v_isSharedCheck_407_ = !lean_is_exclusive(v___x_399_);
if (v_isSharedCheck_407_ == 0)
{
v___x_402_ = v___x_399_;
v_isShared_403_ = v_isSharedCheck_407_;
goto v_resetjp_401_;
}
else
{
lean_inc(v_a_400_);
lean_dec(v___x_399_);
v___x_402_ = lean_box(0);
v_isShared_403_ = v_isSharedCheck_407_;
goto v_resetjp_401_;
}
v_resetjp_401_:
{
lean_object* v___x_405_; 
if (v_isShared_403_ == 0)
{
v___x_405_ = v___x_402_;
goto v_reusejp_404_;
}
else
{
lean_object* v_reuseFailAlloc_406_; 
v_reuseFailAlloc_406_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_406_, 0, v_a_400_);
v___x_405_ = v_reuseFailAlloc_406_;
goto v_reusejp_404_;
}
v_reusejp_404_:
{
return v___x_405_;
}
}
}
else
{
lean_object* v_val_408_; 
v_val_408_ = lean_ctor_get(v___x_397_, 0);
lean_inc(v_val_408_);
lean_dec_ref_known(v___x_397_, 1);
v_a_378_ = v_val_408_;
goto v___jp_377_;
}
v___jp_296_:
{
lean_object* v_snd_302_; lean_object* v_snd_303_; lean_object* v_fst_304_; lean_object* v_fst_305_; lean_object* v_fst_306_; lean_object* v_snd_307_; lean_object* v___x_309_; uint8_t v_isShared_310_; uint8_t v_isSharedCheck_376_; 
v_snd_302_ = lean_ctor_get(v_a_301_, 1);
lean_inc(v_snd_302_);
v_snd_303_ = lean_ctor_get(v_snd_302_, 1);
lean_inc(v_snd_303_);
v_fst_304_ = lean_ctor_get(v_a_301_, 0);
lean_inc(v_fst_304_);
lean_dec_ref(v_a_301_);
v_fst_305_ = lean_ctor_get(v_snd_302_, 0);
lean_inc(v_fst_305_);
lean_dec(v_snd_302_);
v_fst_306_ = lean_ctor_get(v_snd_303_, 0);
v_snd_307_ = lean_ctor_get(v_snd_303_, 1);
v_isSharedCheck_376_ = !lean_is_exclusive(v_snd_303_);
if (v_isSharedCheck_376_ == 0)
{
v___x_309_ = v_snd_303_;
v_isShared_310_ = v_isSharedCheck_376_;
goto v_resetjp_308_;
}
else
{
lean_inc(v_snd_307_);
lean_inc(v_fst_306_);
lean_dec(v_snd_303_);
v___x_309_ = lean_box(0);
v_isShared_310_ = v_isSharedCheck_376_;
goto v_resetjp_308_;
}
v_resetjp_308_:
{
uint8_t v___x_311_; 
v___x_311_ = l_instDecidableEqRat_decEq(v___y_299_, v_fst_304_);
lean_dec(v_fst_304_);
lean_dec_ref(v___y_299_);
if (v___x_311_ == 0)
{
lean_object* v___x_312_; 
lean_inc_ref(v_ds_u03b1_290_);
lean_inc_ref(v_00_u03b1_285_);
lean_inc(v_u_284_);
v___x_312_ = lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionSemiring_x3f(v_u_284_, v_00_u03b1_285_, v_ds_u03b1_290_, v_a_291_, v_a_292_, v_a_293_, v_a_294_);
if (lean_obj_tag(v___x_312_) == 0)
{
lean_object* v_a_313_; lean_object* v___x_315_; uint8_t v_isShared_316_; uint8_t v_isSharedCheck_347_; 
v_a_313_ = lean_ctor_get(v___x_312_, 0);
v_isSharedCheck_347_ = !lean_is_exclusive(v___x_312_);
if (v_isSharedCheck_347_ == 0)
{
v___x_315_ = v___x_312_;
v_isShared_316_ = v_isSharedCheck_347_;
goto v_resetjp_314_;
}
else
{
lean_inc(v_a_313_);
lean_dec(v___x_312_);
v___x_315_ = lean_box(0);
v_isShared_316_ = v_isSharedCheck_347_;
goto v_resetjp_314_;
}
v_resetjp_314_:
{
if (lean_obj_tag(v_a_313_) == 1)
{
lean_object* v_val_317_; lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_322_; 
v_val_317_ = lean_ctor_get(v_a_313_, 0);
lean_inc(v_val_317_);
lean_dec_ref_known(v_a_313_, 1);
v___x_318_ = lean_box(0);
v___x_319_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__13, &lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__13_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__13);
v___x_320_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__1));
if (v_isShared_310_ == 0)
{
lean_ctor_set_tag(v___x_309_, 1);
lean_ctor_set(v___x_309_, 1, v___x_318_);
lean_ctor_set(v___x_309_, 0, v_u_284_);
v___x_322_ = v___x_309_;
goto v_reusejp_321_;
}
else
{
lean_object* v_reuseFailAlloc_344_; 
v_reuseFailAlloc_344_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_344_, 0, v_u_284_);
lean_ctor_set(v_reuseFailAlloc_344_, 1, v___x_318_);
v___x_322_ = v_reuseFailAlloc_344_;
goto v_reusejp_321_;
}
v_reusejp_321_:
{
lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___x_334_; lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v___x_340_; lean_object* v___x_342_; 
lean_inc_ref(v___x_322_);
v___x_323_ = l_Lean_Expr_const___override(v___x_320_, v___x_322_);
lean_inc_ref(v_00_u03b1_285_);
v___x_324_ = l_Lean_Expr_app___override(v___x_323_, v_00_u03b1_285_);
v___x_325_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__4));
v___x_326_ = l_Lean_Expr_const___override(v___x_325_, v___x_322_);
v___x_327_ = l_Lean_Expr_app___override(v___x_326_, v_00_u03b1_285_);
v___x_328_ = l_Lean_Expr_app___override(v___x_327_, v_ds_u03b1_290_);
v___x_329_ = l_Lean_Expr_app___override(v___x_324_, v___x_328_);
v___x_330_ = l_Lean_Expr_app___override(v___x_329_, v_val_317_);
v___x_331_ = l_Lean_Expr_app___override(v___x_330_, v_a_286_);
v___x_332_ = l_Lean_Expr_app___override(v___x_331_, v_b_287_);
v___x_333_ = l_Lean_Expr_app___override(v___x_332_, v___y_298_);
v___x_334_ = l_Lean_Expr_app___override(v___x_333_, v_fst_305_);
v___x_335_ = l_Lean_Expr_app___override(v___x_334_, v___y_297_);
v___x_336_ = l_Lean_Expr_app___override(v___x_335_, v_fst_306_);
v___x_337_ = l_Lean_Expr_app___override(v___x_336_, v___y_300_);
v___x_338_ = l_Lean_Expr_app___override(v___x_337_, v_snd_307_);
v___x_339_ = l_Lean_Expr_app___override(v___x_338_, v___x_319_);
v___x_340_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_340_, 0, v___x_339_);
lean_ctor_set_uint8(v___x_340_, sizeof(void*)*1, v___x_311_);
if (v_isShared_316_ == 0)
{
lean_ctor_set(v___x_315_, 0, v___x_340_);
v___x_342_ = v___x_315_;
goto v_reusejp_341_;
}
else
{
lean_object* v_reuseFailAlloc_343_; 
v_reuseFailAlloc_343_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_343_, 0, v___x_340_);
v___x_342_ = v_reuseFailAlloc_343_;
goto v_reusejp_341_;
}
v_reusejp_341_:
{
return v___x_342_;
}
}
}
else
{
lean_object* v___x_345_; lean_object* v___x_346_; 
lean_del_object(v___x_315_);
lean_dec(v_a_313_);
lean_del_object(v___x_309_);
lean_dec(v_snd_307_);
lean_dec(v_fst_306_);
lean_dec(v_fst_305_);
lean_dec_ref(v___y_300_);
lean_dec_ref(v___y_298_);
lean_dec_ref(v___y_297_);
lean_dec_ref(v_ds_u03b1_290_);
lean_dec_ref(v_b_287_);
lean_dec_ref(v_a_286_);
lean_dec_ref(v_00_u03b1_285_);
lean_dec(v_u_284_);
v___x_345_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20, &lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20);
v___x_346_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm_spec__0___redArg(v___x_345_, v_a_291_, v_a_292_, v_a_293_, v_a_294_);
return v___x_346_;
}
}
}
else
{
lean_object* v_a_348_; lean_object* v___x_350_; uint8_t v_isShared_351_; uint8_t v_isSharedCheck_355_; 
lean_del_object(v___x_309_);
lean_dec(v_snd_307_);
lean_dec(v_fst_306_);
lean_dec(v_fst_305_);
lean_dec_ref(v___y_300_);
lean_dec_ref(v___y_298_);
lean_dec_ref(v___y_297_);
lean_dec_ref(v_ds_u03b1_290_);
lean_dec_ref(v_b_287_);
lean_dec_ref(v_a_286_);
lean_dec_ref(v_00_u03b1_285_);
lean_dec(v_u_284_);
v_a_348_ = lean_ctor_get(v___x_312_, 0);
v_isSharedCheck_355_ = !lean_is_exclusive(v___x_312_);
if (v_isSharedCheck_355_ == 0)
{
v___x_350_ = v___x_312_;
v_isShared_351_ = v_isSharedCheck_355_;
goto v_resetjp_349_;
}
else
{
lean_inc(v_a_348_);
lean_dec(v___x_312_);
v___x_350_ = lean_box(0);
v_isShared_351_ = v_isSharedCheck_355_;
goto v_resetjp_349_;
}
v_resetjp_349_:
{
lean_object* v___x_353_; 
if (v_isShared_351_ == 0)
{
v___x_353_ = v___x_350_;
goto v_reusejp_352_;
}
else
{
lean_object* v_reuseFailAlloc_354_; 
v_reuseFailAlloc_354_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_354_, 0, v_a_348_);
v___x_353_ = v_reuseFailAlloc_354_;
goto v_reusejp_352_;
}
v_reusejp_352_:
{
return v___x_353_;
}
}
}
}
else
{
lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_359_; 
lean_dec(v_fst_306_);
lean_dec(v_fst_305_);
v___x_356_ = lean_box(0);
v___x_357_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__6));
if (v_isShared_310_ == 0)
{
lean_ctor_set_tag(v___x_309_, 1);
lean_ctor_set(v___x_309_, 1, v___x_356_);
lean_ctor_set(v___x_309_, 0, v_u_284_);
v___x_359_ = v___x_309_;
goto v_reusejp_358_;
}
else
{
lean_object* v_reuseFailAlloc_375_; 
v_reuseFailAlloc_375_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_375_, 0, v_u_284_);
lean_ctor_set(v_reuseFailAlloc_375_, 1, v___x_356_);
v___x_359_ = v_reuseFailAlloc_375_;
goto v_reusejp_358_;
}
v_reusejp_358_:
{
lean_object* v___x_360_; lean_object* v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; lean_object* v___x_364_; lean_object* v___x_365_; lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; 
lean_inc_ref(v___x_359_);
v___x_360_ = l_Lean_Expr_const___override(v___x_357_, v___x_359_);
lean_inc_ref(v_00_u03b1_285_);
v___x_361_ = l_Lean_Expr_app___override(v___x_360_, v_00_u03b1_285_);
v___x_362_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___closed__4));
v___x_363_ = l_Lean_Expr_const___override(v___x_362_, v___x_359_);
v___x_364_ = l_Lean_Expr_app___override(v___x_363_, v_00_u03b1_285_);
v___x_365_ = l_Lean_Expr_app___override(v___x_364_, v_ds_u03b1_290_);
v___x_366_ = l_Lean_Expr_app___override(v___x_361_, v___x_365_);
v___x_367_ = l_Lean_Expr_app___override(v___x_366_, v_a_286_);
v___x_368_ = l_Lean_Expr_app___override(v___x_367_, v_b_287_);
v___x_369_ = l_Lean_Expr_app___override(v___x_368_, v___y_298_);
v___x_370_ = l_Lean_Expr_app___override(v___x_369_, v___y_297_);
v___x_371_ = l_Lean_Expr_app___override(v___x_370_, v___y_300_);
v___x_372_ = l_Lean_Expr_app___override(v___x_371_, v_snd_307_);
v___x_373_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_373_, 0, v___x_372_);
lean_ctor_set_uint8(v___x_373_, sizeof(void*)*1, v___x_311_);
v___x_374_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_374_, 0, v___x_373_);
return v___x_374_;
}
}
}
}
v___jp_377_:
{
lean_object* v_snd_379_; lean_object* v_snd_380_; lean_object* v_fst_381_; lean_object* v_fst_382_; lean_object* v_fst_383_; lean_object* v_snd_384_; lean_object* v___x_385_; 
v_snd_379_ = lean_ctor_get(v_a_378_, 1);
lean_inc(v_snd_379_);
v_snd_380_ = lean_ctor_get(v_snd_379_, 1);
lean_inc(v_snd_380_);
v_fst_381_ = lean_ctor_get(v_a_378_, 0);
lean_inc(v_fst_381_);
lean_dec_ref(v_a_378_);
v_fst_382_ = lean_ctor_get(v_snd_379_, 0);
lean_inc(v_fst_382_);
lean_dec(v_snd_379_);
v_fst_383_ = lean_ctor_get(v_snd_380_, 0);
lean_inc(v_fst_383_);
v_snd_384_ = lean_ctor_get(v_snd_380_, 1);
lean_inc(v_snd_384_);
lean_dec(v_snd_380_);
lean_inc_ref(v_ds_u03b1_290_);
lean_inc_ref(v_b_287_);
lean_inc_ref(v_00_u03b1_285_);
lean_inc(v_u_284_);
v___x_385_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toNNRat_x27(v_u_284_, v_00_u03b1_285_, v_b_287_, v_ds_u03b1_290_, v_rb_289_);
if (lean_obj_tag(v___x_385_) == 0)
{
lean_object* v___x_386_; lean_object* v___x_387_; lean_object* v_a_388_; lean_object* v___x_390_; uint8_t v_isShared_391_; uint8_t v_isSharedCheck_395_; 
lean_dec(v_snd_384_);
lean_dec(v_fst_383_);
lean_dec(v_fst_382_);
lean_dec(v_fst_381_);
lean_dec_ref(v_ds_u03b1_290_);
lean_dec_ref(v_b_287_);
lean_dec_ref(v_a_286_);
lean_dec_ref(v_00_u03b1_285_);
lean_dec(v_u_284_);
v___x_386_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20, &lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20);
v___x_387_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm_spec__0___redArg(v___x_386_, v_a_291_, v_a_292_, v_a_293_, v_a_294_);
v_a_388_ = lean_ctor_get(v___x_387_, 0);
v_isSharedCheck_395_ = !lean_is_exclusive(v___x_387_);
if (v_isSharedCheck_395_ == 0)
{
v___x_390_ = v___x_387_;
v_isShared_391_ = v_isSharedCheck_395_;
goto v_resetjp_389_;
}
else
{
lean_inc(v_a_388_);
lean_dec(v___x_387_);
v___x_390_ = lean_box(0);
v_isShared_391_ = v_isSharedCheck_395_;
goto v_resetjp_389_;
}
v_resetjp_389_:
{
lean_object* v___x_393_; 
if (v_isShared_391_ == 0)
{
v___x_393_ = v___x_390_;
goto v_reusejp_392_;
}
else
{
lean_object* v_reuseFailAlloc_394_; 
v_reuseFailAlloc_394_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_394_, 0, v_a_388_);
v___x_393_ = v_reuseFailAlloc_394_;
goto v_reusejp_392_;
}
v_reusejp_392_:
{
return v___x_393_;
}
}
}
else
{
lean_object* v_val_396_; 
v_val_396_ = lean_ctor_get(v___x_385_, 0);
lean_inc(v_val_396_);
lean_dec_ref_known(v___x_385_, 1);
v___y_297_ = v_fst_383_;
v___y_298_ = v_fst_382_;
v___y_299_ = v_fst_381_;
v___y_300_ = v_snd_384_;
v_a_301_ = v_val_396_;
goto v___jp_296_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg___boxed(lean_object* v_u_409_, lean_object* v_00_u03b1_410_, lean_object* v_a_411_, lean_object* v_b_412_, lean_object* v_ra_413_, lean_object* v_rb_414_, lean_object* v_ds_u03b1_415_, lean_object* v_a_416_, lean_object* v_a_417_, lean_object* v_a_418_, lean_object* v_a_419_, lean_object* v_a_420_){
_start:
{
lean_object* v_res_421_; 
v_res_421_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg(v_u_409_, v_00_u03b1_410_, v_a_411_, v_b_412_, v_ra_413_, v_rb_414_, v_ds_u03b1_415_, v_a_416_, v_a_417_, v_a_418_, v_a_419_);
lean_dec(v_a_419_);
lean_dec_ref(v_a_418_);
lean_dec(v_a_417_);
lean_dec_ref(v_a_416_);
return v_res_421_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm(lean_object* v_v_422_, lean_object* v_00_u03b2_423_, lean_object* v_e_424_, lean_object* v_u_425_, lean_object* v_00_u03b1_426_, lean_object* v_a_427_, lean_object* v_b_428_, lean_object* v_ra_429_, lean_object* v_rb_430_, lean_object* v_ds_u03b1_431_, lean_object* v_a_432_, lean_object* v_a_433_, lean_object* v_a_434_, lean_object* v_a_435_){
_start:
{
lean_object* v___x_437_; 
v___x_437_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg(v_u_425_, v_00_u03b1_426_, v_a_427_, v_b_428_, v_ra_429_, v_rb_430_, v_ds_u03b1_431_, v_a_432_, v_a_433_, v_a_434_, v_a_435_);
return v___x_437_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___boxed(lean_object* v_v_438_, lean_object* v_00_u03b2_439_, lean_object* v_e_440_, lean_object* v_u_441_, lean_object* v_00_u03b1_442_, lean_object* v_a_443_, lean_object* v_b_444_, lean_object* v_ra_445_, lean_object* v_rb_446_, lean_object* v_ds_u03b1_447_, lean_object* v_a_448_, lean_object* v_a_449_, lean_object* v_a_450_, lean_object* v_a_451_, lean_object* v_a_452_){
_start:
{
lean_object* v_res_453_; 
v_res_453_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm(v_v_438_, v_00_u03b2_439_, v_e_440_, v_u_441_, v_00_u03b1_442_, v_a_443_, v_b_444_, v_ra_445_, v_rb_446_, v_ds_u03b1_447_, v_a_448_, v_a_449_, v_a_450_, v_a_451_);
lean_dec(v_a_451_);
lean_dec_ref(v_a_450_);
lean_dec(v_a_449_);
lean_dec_ref(v_a_448_);
lean_dec_ref(v_e_440_);
lean_dec_ref(v_00_u03b2_439_);
lean_dec(v_v_438_);
return v_res_453_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg(lean_object* v_u_471_, lean_object* v_00_u03b1_472_, lean_object* v_a_473_, lean_object* v_b_474_, lean_object* v_ra_475_, lean_object* v_rb_476_, lean_object* v_d_u03b1_477_, lean_object* v_a_478_, lean_object* v_a_479_, lean_object* v_a_480_, lean_object* v_a_481_){
_start:
{
lean_object* v___y_484_; lean_object* v___y_485_; lean_object* v___y_486_; lean_object* v___y_487_; lean_object* v_a_488_; lean_object* v_a_565_; lean_object* v___x_584_; 
lean_inc_ref(v_d_u03b1_477_);
lean_inc_ref(v_a_473_);
lean_inc_ref(v_00_u03b1_472_);
lean_inc(v_u_471_);
v___x_584_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27(v_u_471_, v_00_u03b1_472_, v_a_473_, v_d_u03b1_477_, v_ra_475_);
if (lean_obj_tag(v___x_584_) == 0)
{
lean_object* v___x_585_; lean_object* v___x_586_; lean_object* v_a_587_; lean_object* v___x_589_; uint8_t v_isShared_590_; uint8_t v_isSharedCheck_594_; 
lean_dec_ref(v_d_u03b1_477_);
lean_dec_ref(v_rb_476_);
lean_dec_ref(v_b_474_);
lean_dec_ref(v_a_473_);
lean_dec_ref(v_00_u03b1_472_);
lean_dec(v_u_471_);
v___x_585_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20, &lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20);
v___x_586_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm_spec__0___redArg(v___x_585_, v_a_478_, v_a_479_, v_a_480_, v_a_481_);
v_a_587_ = lean_ctor_get(v___x_586_, 0);
v_isSharedCheck_594_ = !lean_is_exclusive(v___x_586_);
if (v_isSharedCheck_594_ == 0)
{
v___x_589_ = v___x_586_;
v_isShared_590_ = v_isSharedCheck_594_;
goto v_resetjp_588_;
}
else
{
lean_inc(v_a_587_);
lean_dec(v___x_586_);
v___x_589_ = lean_box(0);
v_isShared_590_ = v_isSharedCheck_594_;
goto v_resetjp_588_;
}
v_resetjp_588_:
{
lean_object* v___x_592_; 
if (v_isShared_590_ == 0)
{
v___x_592_ = v___x_589_;
goto v_reusejp_591_;
}
else
{
lean_object* v_reuseFailAlloc_593_; 
v_reuseFailAlloc_593_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_593_, 0, v_a_587_);
v___x_592_ = v_reuseFailAlloc_593_;
goto v_reusejp_591_;
}
v_reusejp_591_:
{
return v___x_592_;
}
}
}
else
{
lean_object* v_val_595_; 
v_val_595_ = lean_ctor_get(v___x_584_, 0);
lean_inc(v_val_595_);
lean_dec_ref_known(v___x_584_, 1);
v_a_565_ = v_val_595_;
goto v___jp_564_;
}
v___jp_483_:
{
lean_object* v_snd_489_; lean_object* v_snd_490_; lean_object* v_fst_491_; lean_object* v_fst_492_; lean_object* v_fst_493_; lean_object* v_snd_494_; lean_object* v___x_496_; uint8_t v_isShared_497_; uint8_t v_isSharedCheck_563_; 
v_snd_489_ = lean_ctor_get(v_a_488_, 1);
lean_inc(v_snd_489_);
v_snd_490_ = lean_ctor_get(v_snd_489_, 1);
lean_inc(v_snd_490_);
v_fst_491_ = lean_ctor_get(v_a_488_, 0);
lean_inc(v_fst_491_);
lean_dec_ref(v_a_488_);
v_fst_492_ = lean_ctor_get(v_snd_489_, 0);
lean_inc(v_fst_492_);
lean_dec(v_snd_489_);
v_fst_493_ = lean_ctor_get(v_snd_490_, 0);
v_snd_494_ = lean_ctor_get(v_snd_490_, 1);
v_isSharedCheck_563_ = !lean_is_exclusive(v_snd_490_);
if (v_isSharedCheck_563_ == 0)
{
v___x_496_ = v_snd_490_;
v_isShared_497_ = v_isSharedCheck_563_;
goto v_resetjp_495_;
}
else
{
lean_inc(v_snd_494_);
lean_inc(v_fst_493_);
lean_dec(v_snd_490_);
v___x_496_ = lean_box(0);
v_isShared_497_ = v_isSharedCheck_563_;
goto v_resetjp_495_;
}
v_resetjp_495_:
{
uint8_t v___x_498_; 
v___x_498_ = l_instDecidableEqRat_decEq(v___y_485_, v_fst_491_);
lean_dec(v_fst_491_);
lean_dec_ref(v___y_485_);
if (v___x_498_ == 0)
{
lean_object* v___x_499_; 
lean_inc_ref(v_d_u03b1_477_);
lean_inc_ref(v_00_u03b1_472_);
lean_inc(v_u_471_);
v___x_499_ = lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfDivisionRing_x3f(v_u_471_, v_00_u03b1_472_, v_d_u03b1_477_, v_a_478_, v_a_479_, v_a_480_, v_a_481_);
if (lean_obj_tag(v___x_499_) == 0)
{
lean_object* v_a_500_; lean_object* v___x_502_; uint8_t v_isShared_503_; uint8_t v_isSharedCheck_534_; 
v_a_500_ = lean_ctor_get(v___x_499_, 0);
v_isSharedCheck_534_ = !lean_is_exclusive(v___x_499_);
if (v_isSharedCheck_534_ == 0)
{
v___x_502_ = v___x_499_;
v_isShared_503_ = v_isSharedCheck_534_;
goto v_resetjp_501_;
}
else
{
lean_inc(v_a_500_);
lean_dec(v___x_499_);
v___x_502_ = lean_box(0);
v_isShared_503_ = v_isSharedCheck_534_;
goto v_resetjp_501_;
}
v_resetjp_501_:
{
if (lean_obj_tag(v_a_500_) == 1)
{
lean_object* v_val_504_; lean_object* v___x_505_; lean_object* v___x_506_; lean_object* v___x_507_; lean_object* v___x_509_; 
v_val_504_ = lean_ctor_get(v_a_500_, 0);
lean_inc(v_val_504_);
lean_dec_ref_known(v_a_500_, 1);
v___x_505_ = lean_box(0);
v___x_506_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__13, &lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__13_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__13);
v___x_507_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__1));
if (v_isShared_497_ == 0)
{
lean_ctor_set_tag(v___x_496_, 1);
lean_ctor_set(v___x_496_, 1, v___x_505_);
lean_ctor_set(v___x_496_, 0, v_u_471_);
v___x_509_ = v___x_496_;
goto v_reusejp_508_;
}
else
{
lean_object* v_reuseFailAlloc_531_; 
v_reuseFailAlloc_531_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_531_, 0, v_u_471_);
lean_ctor_set(v_reuseFailAlloc_531_, 1, v___x_505_);
v___x_509_ = v_reuseFailAlloc_531_;
goto v_reusejp_508_;
}
v_reusejp_508_:
{
lean_object* v___x_510_; lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_513_; lean_object* v___x_514_; lean_object* v___x_515_; lean_object* v___x_516_; lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___x_520_; lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; lean_object* v___x_526_; lean_object* v___x_527_; lean_object* v___x_529_; 
lean_inc_ref(v___x_509_);
v___x_510_ = l_Lean_Expr_const___override(v___x_507_, v___x_509_);
lean_inc_ref(v_00_u03b1_472_);
v___x_511_ = l_Lean_Expr_app___override(v___x_510_, v_00_u03b1_472_);
v___x_512_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__4));
v___x_513_ = l_Lean_Expr_const___override(v___x_512_, v___x_509_);
v___x_514_ = l_Lean_Expr_app___override(v___x_513_, v_00_u03b1_472_);
v___x_515_ = l_Lean_Expr_app___override(v___x_514_, v_d_u03b1_477_);
v___x_516_ = l_Lean_Expr_app___override(v___x_511_, v___x_515_);
v___x_517_ = l_Lean_Expr_app___override(v___x_516_, v_val_504_);
v___x_518_ = l_Lean_Expr_app___override(v___x_517_, v_a_473_);
v___x_519_ = l_Lean_Expr_app___override(v___x_518_, v_b_474_);
v___x_520_ = l_Lean_Expr_app___override(v___x_519_, v___y_484_);
v___x_521_ = l_Lean_Expr_app___override(v___x_520_, v_fst_492_);
v___x_522_ = l_Lean_Expr_app___override(v___x_521_, v___y_487_);
v___x_523_ = l_Lean_Expr_app___override(v___x_522_, v_fst_493_);
v___x_524_ = l_Lean_Expr_app___override(v___x_523_, v___y_486_);
v___x_525_ = l_Lean_Expr_app___override(v___x_524_, v_snd_494_);
v___x_526_ = l_Lean_Expr_app___override(v___x_525_, v___x_506_);
v___x_527_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_527_, 0, v___x_526_);
lean_ctor_set_uint8(v___x_527_, sizeof(void*)*1, v___x_498_);
if (v_isShared_503_ == 0)
{
lean_ctor_set(v___x_502_, 0, v___x_527_);
v___x_529_ = v___x_502_;
goto v_reusejp_528_;
}
else
{
lean_object* v_reuseFailAlloc_530_; 
v_reuseFailAlloc_530_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_530_, 0, v___x_527_);
v___x_529_ = v_reuseFailAlloc_530_;
goto v_reusejp_528_;
}
v_reusejp_528_:
{
return v___x_529_;
}
}
}
else
{
lean_object* v___x_532_; lean_object* v___x_533_; 
lean_del_object(v___x_502_);
lean_dec(v_a_500_);
lean_del_object(v___x_496_);
lean_dec(v_snd_494_);
lean_dec(v_fst_493_);
lean_dec(v_fst_492_);
lean_dec_ref(v___y_487_);
lean_dec_ref(v___y_486_);
lean_dec_ref(v___y_484_);
lean_dec_ref(v_d_u03b1_477_);
lean_dec_ref(v_b_474_);
lean_dec_ref(v_a_473_);
lean_dec_ref(v_00_u03b1_472_);
lean_dec(v_u_471_);
v___x_532_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20, &lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20);
v___x_533_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm_spec__0___redArg(v___x_532_, v_a_478_, v_a_479_, v_a_480_, v_a_481_);
return v___x_533_;
}
}
}
else
{
lean_object* v_a_535_; lean_object* v___x_537_; uint8_t v_isShared_538_; uint8_t v_isSharedCheck_542_; 
lean_del_object(v___x_496_);
lean_dec(v_snd_494_);
lean_dec(v_fst_493_);
lean_dec(v_fst_492_);
lean_dec_ref(v___y_487_);
lean_dec_ref(v___y_486_);
lean_dec_ref(v___y_484_);
lean_dec_ref(v_d_u03b1_477_);
lean_dec_ref(v_b_474_);
lean_dec_ref(v_a_473_);
lean_dec_ref(v_00_u03b1_472_);
lean_dec(v_u_471_);
v_a_535_ = lean_ctor_get(v___x_499_, 0);
v_isSharedCheck_542_ = !lean_is_exclusive(v___x_499_);
if (v_isSharedCheck_542_ == 0)
{
v___x_537_ = v___x_499_;
v_isShared_538_ = v_isSharedCheck_542_;
goto v_resetjp_536_;
}
else
{
lean_inc(v_a_535_);
lean_dec(v___x_499_);
v___x_537_ = lean_box(0);
v_isShared_538_ = v_isSharedCheck_542_;
goto v_resetjp_536_;
}
v_resetjp_536_:
{
lean_object* v___x_540_; 
if (v_isShared_538_ == 0)
{
v___x_540_ = v___x_537_;
goto v_reusejp_539_;
}
else
{
lean_object* v_reuseFailAlloc_541_; 
v_reuseFailAlloc_541_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_541_, 0, v_a_535_);
v___x_540_ = v_reuseFailAlloc_541_;
goto v_reusejp_539_;
}
v_reusejp_539_:
{
return v___x_540_;
}
}
}
}
else
{
lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_546_; 
lean_dec(v_fst_493_);
lean_dec(v_fst_492_);
v___x_543_ = lean_box(0);
v___x_544_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__6));
if (v_isShared_497_ == 0)
{
lean_ctor_set_tag(v___x_496_, 1);
lean_ctor_set(v___x_496_, 1, v___x_543_);
lean_ctor_set(v___x_496_, 0, v_u_471_);
v___x_546_ = v___x_496_;
goto v_reusejp_545_;
}
else
{
lean_object* v_reuseFailAlloc_562_; 
v_reuseFailAlloc_562_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_562_, 0, v_u_471_);
lean_ctor_set(v_reuseFailAlloc_562_, 1, v___x_543_);
v___x_546_ = v_reuseFailAlloc_562_;
goto v_reusejp_545_;
}
v_reusejp_545_:
{
lean_object* v___x_547_; lean_object* v___x_548_; lean_object* v___x_549_; lean_object* v___x_550_; lean_object* v___x_551_; lean_object* v___x_552_; lean_object* v___x_553_; lean_object* v___x_554_; lean_object* v___x_555_; lean_object* v___x_556_; lean_object* v___x_557_; lean_object* v___x_558_; lean_object* v___x_559_; lean_object* v___x_560_; lean_object* v___x_561_; 
lean_inc_ref(v___x_546_);
v___x_547_ = l_Lean_Expr_const___override(v___x_544_, v___x_546_);
lean_inc_ref(v_00_u03b1_472_);
v___x_548_ = l_Lean_Expr_app___override(v___x_547_, v_00_u03b1_472_);
v___x_549_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___closed__4));
v___x_550_ = l_Lean_Expr_const___override(v___x_549_, v___x_546_);
v___x_551_ = l_Lean_Expr_app___override(v___x_550_, v_00_u03b1_472_);
v___x_552_ = l_Lean_Expr_app___override(v___x_551_, v_d_u03b1_477_);
v___x_553_ = l_Lean_Expr_app___override(v___x_548_, v___x_552_);
v___x_554_ = l_Lean_Expr_app___override(v___x_553_, v_a_473_);
v___x_555_ = l_Lean_Expr_app___override(v___x_554_, v_b_474_);
v___x_556_ = l_Lean_Expr_app___override(v___x_555_, v___y_484_);
v___x_557_ = l_Lean_Expr_app___override(v___x_556_, v___y_487_);
v___x_558_ = l_Lean_Expr_app___override(v___x_557_, v___y_486_);
v___x_559_ = l_Lean_Expr_app___override(v___x_558_, v_snd_494_);
v___x_560_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_560_, 0, v___x_559_);
lean_ctor_set_uint8(v___x_560_, sizeof(void*)*1, v___x_498_);
v___x_561_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_561_, 0, v___x_560_);
return v___x_561_;
}
}
}
}
v___jp_564_:
{
lean_object* v_snd_566_; lean_object* v_snd_567_; lean_object* v_fst_568_; lean_object* v_fst_569_; lean_object* v_fst_570_; lean_object* v_snd_571_; lean_object* v___x_572_; 
v_snd_566_ = lean_ctor_get(v_a_565_, 1);
lean_inc(v_snd_566_);
v_snd_567_ = lean_ctor_get(v_snd_566_, 1);
lean_inc(v_snd_567_);
v_fst_568_ = lean_ctor_get(v_a_565_, 0);
lean_inc(v_fst_568_);
lean_dec_ref(v_a_565_);
v_fst_569_ = lean_ctor_get(v_snd_566_, 0);
lean_inc(v_fst_569_);
lean_dec(v_snd_566_);
v_fst_570_ = lean_ctor_get(v_snd_567_, 0);
lean_inc(v_fst_570_);
v_snd_571_ = lean_ctor_get(v_snd_567_, 1);
lean_inc(v_snd_571_);
lean_dec(v_snd_567_);
lean_inc_ref(v_d_u03b1_477_);
lean_inc_ref(v_b_474_);
lean_inc_ref(v_00_u03b1_472_);
lean_inc(v_u_471_);
v___x_572_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27(v_u_471_, v_00_u03b1_472_, v_b_474_, v_d_u03b1_477_, v_rb_476_);
if (lean_obj_tag(v___x_572_) == 0)
{
lean_object* v___x_573_; lean_object* v___x_574_; lean_object* v_a_575_; lean_object* v___x_577_; uint8_t v_isShared_578_; uint8_t v_isSharedCheck_582_; 
lean_dec(v_snd_571_);
lean_dec(v_fst_570_);
lean_dec(v_fst_569_);
lean_dec(v_fst_568_);
lean_dec_ref(v_d_u03b1_477_);
lean_dec_ref(v_b_474_);
lean_dec_ref(v_a_473_);
lean_dec_ref(v_00_u03b1_472_);
lean_dec(v_u_471_);
v___x_573_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20, &lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20);
v___x_574_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm_spec__0___redArg(v___x_573_, v_a_478_, v_a_479_, v_a_480_, v_a_481_);
v_a_575_ = lean_ctor_get(v___x_574_, 0);
v_isSharedCheck_582_ = !lean_is_exclusive(v___x_574_);
if (v_isSharedCheck_582_ == 0)
{
v___x_577_ = v___x_574_;
v_isShared_578_ = v_isSharedCheck_582_;
goto v_resetjp_576_;
}
else
{
lean_inc(v_a_575_);
lean_dec(v___x_574_);
v___x_577_ = lean_box(0);
v_isShared_578_ = v_isSharedCheck_582_;
goto v_resetjp_576_;
}
v_resetjp_576_:
{
lean_object* v___x_580_; 
if (v_isShared_578_ == 0)
{
v___x_580_ = v___x_577_;
goto v_reusejp_579_;
}
else
{
lean_object* v_reuseFailAlloc_581_; 
v_reuseFailAlloc_581_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_581_, 0, v_a_575_);
v___x_580_ = v_reuseFailAlloc_581_;
goto v_reusejp_579_;
}
v_reusejp_579_:
{
return v___x_580_;
}
}
}
else
{
lean_object* v_val_583_; 
v_val_583_ = lean_ctor_get(v___x_572_, 0);
lean_inc(v_val_583_);
lean_dec_ref_known(v___x_572_, 1);
v___y_484_ = v_fst_569_;
v___y_485_ = v_fst_568_;
v___y_486_ = v_snd_571_;
v___y_487_ = v_fst_570_;
v_a_488_ = v_val_583_;
goto v___jp_483_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg___boxed(lean_object* v_u_596_, lean_object* v_00_u03b1_597_, lean_object* v_a_598_, lean_object* v_b_599_, lean_object* v_ra_600_, lean_object* v_rb_601_, lean_object* v_d_u03b1_602_, lean_object* v_a_603_, lean_object* v_a_604_, lean_object* v_a_605_, lean_object* v_a_606_, lean_object* v_a_607_){
_start:
{
lean_object* v_res_608_; 
v_res_608_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg(v_u_596_, v_00_u03b1_597_, v_a_598_, v_b_599_, v_ra_600_, v_rb_601_, v_d_u03b1_602_, v_a_603_, v_a_604_, v_a_605_, v_a_606_);
lean_dec(v_a_606_);
lean_dec_ref(v_a_605_);
lean_dec(v_a_604_);
lean_dec_ref(v_a_603_);
return v_res_608_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm(lean_object* v_v_609_, lean_object* v_00_u03b2_610_, lean_object* v_e_611_, lean_object* v_u_612_, lean_object* v_00_u03b1_613_, lean_object* v_a_614_, lean_object* v_b_615_, lean_object* v_ra_616_, lean_object* v_rb_617_, lean_object* v_d_u03b1_618_, lean_object* v_a_619_, lean_object* v_a_620_, lean_object* v_a_621_, lean_object* v_a_622_){
_start:
{
lean_object* v___x_624_; 
v___x_624_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg(v_u_612_, v_00_u03b1_613_, v_a_614_, v_b_615_, v_ra_616_, v_rb_617_, v_d_u03b1_618_, v_a_619_, v_a_620_, v_a_621_, v_a_622_);
return v___x_624_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___boxed(lean_object* v_v_625_, lean_object* v_00_u03b2_626_, lean_object* v_e_627_, lean_object* v_u_628_, lean_object* v_00_u03b1_629_, lean_object* v_a_630_, lean_object* v_b_631_, lean_object* v_ra_632_, lean_object* v_rb_633_, lean_object* v_d_u03b1_634_, lean_object* v_a_635_, lean_object* v_a_636_, lean_object* v_a_637_, lean_object* v_a_638_, lean_object* v_a_639_){
_start:
{
lean_object* v_res_640_; 
v_res_640_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm(v_v_625_, v_00_u03b2_626_, v_e_627_, v_u_628_, v_00_u03b1_629_, v_a_630_, v_b_631_, v_ra_632_, v_rb_633_, v_d_u03b1_634_, v_a_635_, v_a_636_, v_a_637_, v_a_638_);
lean_dec(v_a_638_);
lean_dec_ref(v_a_637_);
lean_dec(v_a_636_);
lean_dec_ref(v_a_635_);
lean_dec_ref(v_e_627_);
lean_dec_ref(v_00_u03b2_626_);
lean_dec(v_v_625_);
return v_res_640_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalEq_spec__0___redArg(lean_object* v_k_641_, uint8_t v_allowLevelAssignments_642_, lean_object* v___y_643_, lean_object* v___y_644_, lean_object* v___y_645_, lean_object* v___y_646_){
_start:
{
lean_object* v___x_648_; 
v___x_648_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), v_allowLevelAssignments_642_, v_k_641_, v___y_643_, v___y_644_, v___y_645_, v___y_646_);
if (lean_obj_tag(v___x_648_) == 0)
{
lean_object* v_a_649_; lean_object* v___x_651_; uint8_t v_isShared_652_; uint8_t v_isSharedCheck_656_; 
v_a_649_ = lean_ctor_get(v___x_648_, 0);
v_isSharedCheck_656_ = !lean_is_exclusive(v___x_648_);
if (v_isSharedCheck_656_ == 0)
{
v___x_651_ = v___x_648_;
v_isShared_652_ = v_isSharedCheck_656_;
goto v_resetjp_650_;
}
else
{
lean_inc(v_a_649_);
lean_dec(v___x_648_);
v___x_651_ = lean_box(0);
v_isShared_652_ = v_isSharedCheck_656_;
goto v_resetjp_650_;
}
v_resetjp_650_:
{
lean_object* v___x_654_; 
if (v_isShared_652_ == 0)
{
v___x_654_ = v___x_651_;
goto v_reusejp_653_;
}
else
{
lean_object* v_reuseFailAlloc_655_; 
v_reuseFailAlloc_655_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_655_, 0, v_a_649_);
v___x_654_ = v_reuseFailAlloc_655_;
goto v_reusejp_653_;
}
v_reusejp_653_:
{
return v___x_654_;
}
}
}
else
{
lean_object* v_a_657_; lean_object* v___x_659_; uint8_t v_isShared_660_; uint8_t v_isSharedCheck_664_; 
v_a_657_ = lean_ctor_get(v___x_648_, 0);
v_isSharedCheck_664_ = !lean_is_exclusive(v___x_648_);
if (v_isSharedCheck_664_ == 0)
{
v___x_659_ = v___x_648_;
v_isShared_660_ = v_isSharedCheck_664_;
goto v_resetjp_658_;
}
else
{
lean_inc(v_a_657_);
lean_dec(v___x_648_);
v___x_659_ = lean_box(0);
v_isShared_660_ = v_isSharedCheck_664_;
goto v_resetjp_658_;
}
v_resetjp_658_:
{
lean_object* v___x_662_; 
if (v_isShared_660_ == 0)
{
v___x_662_ = v___x_659_;
goto v_reusejp_661_;
}
else
{
lean_object* v_reuseFailAlloc_663_; 
v_reuseFailAlloc_663_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_663_, 0, v_a_657_);
v___x_662_ = v_reuseFailAlloc_663_;
goto v_reusejp_661_;
}
v_reusejp_661_:
{
return v___x_662_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalEq_spec__0___redArg___boxed(lean_object* v_k_665_, lean_object* v_allowLevelAssignments_666_, lean_object* v___y_667_, lean_object* v___y_668_, lean_object* v___y_669_, lean_object* v___y_670_, lean_object* v___y_671_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_672_; lean_object* v_res_673_; 
v_allowLevelAssignments_boxed_672_ = lean_unbox(v_allowLevelAssignments_666_);
v_res_673_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalEq_spec__0___redArg(v_k_665_, v_allowLevelAssignments_boxed_672_, v___y_667_, v___y_668_, v___y_669_, v___y_670_);
lean_dec(v___y_670_);
lean_dec_ref(v___y_669_);
lean_dec(v___y_668_);
lean_dec_ref(v___y_667_);
return v_res_673_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalEq_spec__0(lean_object* v_00_u03b1_674_, lean_object* v_k_675_, uint8_t v_allowLevelAssignments_676_, lean_object* v___y_677_, lean_object* v___y_678_, lean_object* v___y_679_, lean_object* v___y_680_){
_start:
{
lean_object* v___x_682_; 
v___x_682_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalEq_spec__0___redArg(v_k_675_, v_allowLevelAssignments_676_, v___y_677_, v___y_678_, v___y_679_, v___y_680_);
return v___x_682_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalEq_spec__0___boxed(lean_object* v_00_u03b1_683_, lean_object* v_k_684_, lean_object* v_allowLevelAssignments_685_, lean_object* v___y_686_, lean_object* v___y_687_, lean_object* v___y_688_, lean_object* v___y_689_, lean_object* v___y_690_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_691_; lean_object* v_res_692_; 
v_allowLevelAssignments_boxed_691_ = lean_unbox(v_allowLevelAssignments_685_);
v_res_692_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalEq_spec__0(v_00_u03b1_683_, v_k_684_, v_allowLevelAssignments_boxed_691_, v___y_686_, v___y_687_, v___y_688_, v___y_689_);
lean_dec(v___y_689_);
lean_dec_ref(v___y_688_);
lean_dec(v___y_687_);
lean_dec_ref(v___y_686_);
return v_res_692_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__0(lean_object* v_fn_693_, lean_object* v___x_694_, lean_object* v___y_695_, lean_object* v___y_696_, lean_object* v___y_697_, lean_object* v___y_698_){
_start:
{
lean_object* v___x_700_; 
v___x_700_ = l_Lean_Meta_isExprDefEq(v_fn_693_, v___x_694_, v___y_695_, v___y_696_, v___y_697_, v___y_698_);
return v___x_700_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__0___boxed(lean_object* v_fn_701_, lean_object* v___x_702_, lean_object* v___y_703_, lean_object* v___y_704_, lean_object* v___y_705_, lean_object* v___y_706_, lean_object* v___y_707_){
_start:
{
lean_object* v_res_708_; 
v_res_708_ = lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__0(v_fn_701_, v___x_702_, v___y_703_, v___y_704_, v___y_705_, v___y_706_);
lean_dec(v___y_706_);
lean_dec_ref(v___y_705_);
lean_dec(v___y_704_);
lean_dec_ref(v___y_703_);
return v_res_708_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__3(void){
_start:
{
lean_object* v___x_717_; lean_object* v___x_718_; lean_object* v___x_719_; 
v___x_717_ = lean_box(0);
v___x_718_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__2));
v___x_719_ = l_Lean_Expr_const___override(v___x_718_, v___x_717_);
return v___x_719_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__6(void){
_start:
{
lean_object* v___x_726_; lean_object* v___x_727_; lean_object* v___x_728_; 
v___x_726_ = lean_box(0);
v___x_727_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__5));
v___x_728_ = l_Lean_Expr_const___override(v___x_727_, v___x_726_);
return v___x_728_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__9(void){
_start:
{
lean_object* v___x_735_; lean_object* v___x_736_; lean_object* v___x_737_; 
v___x_735_ = lean_box(0);
v___x_736_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__8));
v___x_737_ = l_Lean_Expr_const___override(v___x_736_, v___x_735_);
return v___x_737_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__12(void){
_start:
{
lean_object* v___x_744_; lean_object* v___x_745_; lean_object* v___x_746_; 
v___x_744_ = lean_box(0);
v___x_745_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__11));
v___x_746_ = l_Lean_Expr_const___override(v___x_745_, v___x_744_);
return v___x_746_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1(lean_object* v_v_761_, lean_object* v_00_u03b2_762_, lean_object* v_e_763_, lean_object* v___y_764_, lean_object* v___y_765_, lean_object* v___y_766_, lean_object* v___y_767_){
_start:
{
lean_object* v___y_770_; lean_object* v___y_771_; lean_object* v___y_772_; lean_object* v___y_773_; lean_object* v___y_777_; lean_object* v___y_778_; lean_object* v___y_779_; lean_object* v___y_780_; lean_object* v___x_783_; 
v___x_783_ = l_Lean_Meta_whnfR(v_e_763_, v___y_764_, v___y_765_, v___y_766_, v___y_767_);
if (lean_obj_tag(v___x_783_) == 0)
{
lean_object* v_a_784_; lean_object* v___y_786_; lean_object* v___y_787_; lean_object* v___y_788_; lean_object* v___y_789_; 
v_a_784_ = lean_ctor_get(v___x_783_, 0);
lean_inc(v_a_784_);
lean_dec_ref_known(v___x_783_, 1);
if (lean_obj_tag(v_a_784_) == 5)
{
lean_object* v_fn_792_; 
v_fn_792_ = lean_ctor_get(v_a_784_, 0);
lean_inc_ref(v_fn_792_);
if (lean_obj_tag(v_fn_792_) == 5)
{
lean_object* v_arg_793_; lean_object* v_fn_794_; lean_object* v_arg_795_; lean_object* v___x_796_; 
v_arg_793_ = lean_ctor_get(v_a_784_, 1);
lean_inc_ref(v_arg_793_);
lean_dec_ref_known(v_a_784_, 2);
v_fn_794_ = lean_ctor_get(v_fn_792_, 0);
lean_inc_ref(v_fn_794_);
v_arg_795_ = lean_ctor_get(v_fn_792_, 1);
lean_inc_ref(v_arg_795_);
lean_dec_ref_known(v_fn_792_, 2);
v___x_796_ = lp_mathlib_Qq_inferTypeQ_x27(v_arg_795_, v___y_764_, v___y_765_, v___y_766_, v___y_767_);
if (lean_obj_tag(v___x_796_) == 0)
{
lean_object* v_a_797_; lean_object* v_snd_798_; lean_object* v_fst_799_; lean_object* v___x_801_; uint8_t v_isShared_802_; uint8_t v_isSharedCheck_1051_; 
v_a_797_ = lean_ctor_get(v___x_796_, 0);
lean_inc(v_a_797_);
lean_dec_ref_known(v___x_796_, 1);
v_snd_798_ = lean_ctor_get(v_a_797_, 1);
v_fst_799_ = lean_ctor_get(v_a_797_, 0);
v_isSharedCheck_1051_ = !lean_is_exclusive(v_a_797_);
if (v_isSharedCheck_1051_ == 0)
{
v___x_801_ = v_a_797_;
v_isShared_802_ = v_isSharedCheck_1051_;
goto v_resetjp_800_;
}
else
{
lean_inc(v_snd_798_);
lean_inc(v_fst_799_);
lean_dec(v_a_797_);
v___x_801_ = lean_box(0);
v_isShared_802_ = v_isSharedCheck_1051_;
goto v_resetjp_800_;
}
v_resetjp_800_:
{
lean_object* v_fst_803_; lean_object* v_snd_804_; lean_object* v___x_806_; uint8_t v_isShared_807_; uint8_t v_isSharedCheck_1050_; 
v_fst_803_ = lean_ctor_get(v_snd_798_, 0);
v_snd_804_ = lean_ctor_get(v_snd_798_, 1);
v_isSharedCheck_1050_ = !lean_is_exclusive(v_snd_798_);
if (v_isSharedCheck_1050_ == 0)
{
v___x_806_ = v_snd_798_;
v_isShared_807_ = v_isSharedCheck_1050_;
goto v_resetjp_805_;
}
else
{
lean_inc(v_snd_804_);
lean_inc(v_fst_803_);
lean_dec(v_snd_798_);
v___x_806_ = lean_box(0);
v_isShared_807_ = v_isSharedCheck_1050_;
goto v_resetjp_805_;
}
v_resetjp_805_:
{
lean_object* v___x_808_; lean_object* v___x_809_; lean_object* v___x_810_; lean_object* v___x_812_; 
lean_inc(v_fst_799_);
v___x_808_ = l_Lean_Level_succ___override(v_fst_799_);
v___x_809_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__0));
v___x_810_ = lean_box(0);
if (v_isShared_807_ == 0)
{
lean_ctor_set_tag(v___x_806_, 1);
lean_ctor_set(v___x_806_, 1, v___x_810_);
lean_ctor_set(v___x_806_, 0, v___x_808_);
v___x_812_ = v___x_806_;
goto v_reusejp_811_;
}
else
{
lean_object* v_reuseFailAlloc_1049_; 
v_reuseFailAlloc_1049_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1049_, 0, v___x_808_);
lean_ctor_set(v_reuseFailAlloc_1049_, 1, v___x_810_);
v___x_812_ = v_reuseFailAlloc_1049_;
goto v_reusejp_811_;
}
v_reusejp_811_:
{
lean_object* v___x_813_; lean_object* v___x_814_; lean_object* v___f_815_; uint8_t v___x_816_; lean_object* v___x_817_; 
v___x_813_ = l_Lean_Expr_const___override(v___x_809_, v___x_812_);
lean_inc(v_fst_803_);
v___x_814_ = l_Lean_Expr_app___override(v___x_813_, v_fst_803_);
v___f_815_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__0___boxed), 7, 2);
lean_closure_set(v___f_815_, 0, v_fn_794_);
lean_closure_set(v___f_815_, 1, v___x_814_);
v___x_816_ = 0;
v___x_817_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalEq_spec__0___redArg(v___f_815_, v___x_816_, v___y_764_, v___y_765_, v___y_766_, v___y_767_);
if (lean_obj_tag(v___x_817_) == 0)
{
lean_object* v_a_818_; uint8_t v___x_1030_; 
v_a_818_ = lean_ctor_get(v___x_817_, 0);
lean_inc(v_a_818_);
lean_dec_ref_known(v___x_817_, 1);
v___x_1030_ = lean_unbox(v_a_818_);
lean_dec(v_a_818_);
if (v___x_1030_ == 0)
{
lean_object* v___x_1031_; lean_object* v___x_1032_; lean_object* v_a_1033_; lean_object* v___x_1035_; uint8_t v_isShared_1036_; uint8_t v_isSharedCheck_1040_; 
lean_dec(v_snd_804_);
lean_dec(v_fst_803_);
lean_del_object(v___x_801_);
lean_dec(v_fst_799_);
lean_dec_ref(v_arg_793_);
v___x_1031_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20, &lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20);
v___x_1032_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm_spec__0___redArg(v___x_1031_, v___y_764_, v___y_765_, v___y_766_, v___y_767_);
v_a_1033_ = lean_ctor_get(v___x_1032_, 0);
v_isSharedCheck_1040_ = !lean_is_exclusive(v___x_1032_);
if (v_isSharedCheck_1040_ == 0)
{
v___x_1035_ = v___x_1032_;
v_isShared_1036_ = v_isSharedCheck_1040_;
goto v_resetjp_1034_;
}
else
{
lean_inc(v_a_1033_);
lean_dec(v___x_1032_);
v___x_1035_ = lean_box(0);
v_isShared_1036_ = v_isSharedCheck_1040_;
goto v_resetjp_1034_;
}
v_resetjp_1034_:
{
lean_object* v___x_1038_; 
if (v_isShared_1036_ == 0)
{
v___x_1038_ = v___x_1035_;
goto v_reusejp_1037_;
}
else
{
lean_object* v_reuseFailAlloc_1039_; 
v_reuseFailAlloc_1039_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1039_, 0, v_a_1033_);
v___x_1038_ = v_reuseFailAlloc_1039_;
goto v_reusejp_1037_;
}
v_reusejp_1037_:
{
return v___x_1038_;
}
}
}
else
{
goto v___jp_819_;
}
v___jp_819_:
{
lean_object* v___x_820_; 
lean_inc(v_snd_804_);
lean_inc(v_fst_803_);
lean_inc(v_fst_799_);
v___x_820_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v_fst_799_, v_fst_803_, v_snd_804_, v___x_816_, v___y_764_, v___y_765_, v___y_766_, v___y_767_);
if (lean_obj_tag(v___x_820_) == 0)
{
lean_object* v_a_821_; lean_object* v___x_822_; 
v_a_821_ = lean_ctor_get(v___x_820_, 0);
lean_inc(v_a_821_);
lean_dec_ref_known(v___x_820_, 1);
lean_inc_ref(v_arg_793_);
lean_inc(v_fst_803_);
lean_inc(v_fst_799_);
v___x_822_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v_fst_799_, v_fst_803_, v_arg_793_, v___x_816_, v___y_764_, v___y_765_, v___y_766_, v___y_767_);
if (lean_obj_tag(v___x_822_) == 0)
{
switch(lean_obj_tag(v_a_821_))
{
case 0:
{
lean_object* v_a_823_; lean_object* v___x_825_; uint8_t v_isShared_826_; uint8_t v_isSharedCheck_899_; 
lean_dec(v_fst_803_);
lean_del_object(v___x_801_);
lean_dec(v_fst_799_);
v_a_823_ = lean_ctor_get(v___x_822_, 0);
v_isSharedCheck_899_ = !lean_is_exclusive(v___x_822_);
if (v_isSharedCheck_899_ == 0)
{
v___x_825_ = v___x_822_;
v_isShared_826_ = v_isSharedCheck_899_;
goto v_resetjp_824_;
}
else
{
lean_inc(v_a_823_);
lean_dec(v___x_822_);
v___x_825_ = lean_box(0);
v_isShared_826_ = v_isSharedCheck_899_;
goto v_resetjp_824_;
}
v_resetjp_824_:
{
switch(lean_obj_tag(v_a_823_))
{
case 0:
{
uint8_t v_val_827_; 
v_val_827_ = lean_ctor_get_uint8(v_a_821_, sizeof(void*)*1);
if (v_val_827_ == 0)
{
uint8_t v_val_828_; 
v_val_828_ = lean_ctor_get_uint8(v_a_823_, sizeof(void*)*1);
if (v_val_828_ == 0)
{
lean_object* v_proof_829_; lean_object* v_proof_830_; lean_object* v___x_832_; uint8_t v_isShared_833_; uint8_t v_isSharedCheck_846_; 
v_proof_829_ = lean_ctor_get(v_a_821_, 0);
lean_inc_ref(v_proof_829_);
lean_dec_ref_known(v_a_821_, 1);
v_proof_830_ = lean_ctor_get(v_a_823_, 0);
v_isSharedCheck_846_ = !lean_is_exclusive(v_a_823_);
if (v_isSharedCheck_846_ == 0)
{
v___x_832_ = v_a_823_;
v_isShared_833_ = v_isSharedCheck_846_;
goto v_resetjp_831_;
}
else
{
lean_inc(v_proof_830_);
lean_dec(v_a_823_);
v___x_832_ = lean_box(0);
v_isShared_833_ = v_isSharedCheck_846_;
goto v_resetjp_831_;
}
v_resetjp_831_:
{
lean_object* v___x_834_; lean_object* v___x_835_; lean_object* v___x_836_; lean_object* v___x_837_; lean_object* v___x_838_; uint8_t v___x_839_; lean_object* v___x_841_; 
v___x_834_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__3, &lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__3_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__3);
v___x_835_ = l_Lean_Expr_app___override(v___x_834_, v_snd_804_);
v___x_836_ = l_Lean_Expr_app___override(v___x_835_, v_arg_793_);
v___x_837_ = l_Lean_Expr_app___override(v___x_836_, v_proof_829_);
v___x_838_ = l_Lean_Expr_app___override(v___x_837_, v_proof_830_);
v___x_839_ = 1;
if (v_isShared_833_ == 0)
{
lean_ctor_set(v___x_832_, 0, v___x_838_);
v___x_841_ = v___x_832_;
goto v_reusejp_840_;
}
else
{
lean_object* v_reuseFailAlloc_845_; 
v_reuseFailAlloc_845_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_845_, 0, v___x_838_);
v___x_841_ = v_reuseFailAlloc_845_;
goto v_reusejp_840_;
}
v_reusejp_840_:
{
lean_object* v___x_843_; 
lean_ctor_set_uint8(v___x_841_, sizeof(void*)*1, v___x_839_);
if (v_isShared_826_ == 0)
{
lean_ctor_set(v___x_825_, 0, v___x_841_);
v___x_843_ = v___x_825_;
goto v_reusejp_842_;
}
else
{
lean_object* v_reuseFailAlloc_844_; 
v_reuseFailAlloc_844_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_844_, 0, v___x_841_);
v___x_843_ = v_reuseFailAlloc_844_;
goto v_reusejp_842_;
}
v_reusejp_842_:
{
return v___x_843_;
}
}
}
}
else
{
lean_object* v_proof_847_; lean_object* v_proof_848_; lean_object* v___x_850_; uint8_t v_isShared_851_; uint8_t v_isSharedCheck_863_; 
v_proof_847_ = lean_ctor_get(v_a_821_, 0);
lean_inc_ref(v_proof_847_);
lean_dec_ref_known(v_a_821_, 1);
v_proof_848_ = lean_ctor_get(v_a_823_, 0);
v_isSharedCheck_863_ = !lean_is_exclusive(v_a_823_);
if (v_isSharedCheck_863_ == 0)
{
v___x_850_ = v_a_823_;
v_isShared_851_ = v_isSharedCheck_863_;
goto v_resetjp_849_;
}
else
{
lean_inc(v_proof_848_);
lean_dec(v_a_823_);
v___x_850_ = lean_box(0);
v_isShared_851_ = v_isSharedCheck_863_;
goto v_resetjp_849_;
}
v_resetjp_849_:
{
lean_object* v___x_852_; lean_object* v___x_853_; lean_object* v___x_854_; lean_object* v___x_855_; lean_object* v___x_856_; lean_object* v___x_858_; 
v___x_852_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__6, &lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__6_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__6);
v___x_853_ = l_Lean_Expr_app___override(v___x_852_, v_snd_804_);
v___x_854_ = l_Lean_Expr_app___override(v___x_853_, v_arg_793_);
v___x_855_ = l_Lean_Expr_app___override(v___x_854_, v_proof_847_);
v___x_856_ = l_Lean_Expr_app___override(v___x_855_, v_proof_848_);
if (v_isShared_851_ == 0)
{
lean_ctor_set(v___x_850_, 0, v___x_856_);
v___x_858_ = v___x_850_;
goto v_reusejp_857_;
}
else
{
lean_object* v_reuseFailAlloc_862_; 
v_reuseFailAlloc_862_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_862_, 0, v___x_856_);
v___x_858_ = v_reuseFailAlloc_862_;
goto v_reusejp_857_;
}
v_reusejp_857_:
{
lean_object* v___x_860_; 
lean_ctor_set_uint8(v___x_858_, sizeof(void*)*1, v_val_827_);
if (v_isShared_826_ == 0)
{
lean_ctor_set(v___x_825_, 0, v___x_858_);
v___x_860_ = v___x_825_;
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
else
{
uint8_t v_val_864_; 
v_val_864_ = lean_ctor_get_uint8(v_a_823_, sizeof(void*)*1);
if (v_val_864_ == 0)
{
lean_object* v_proof_865_; lean_object* v_proof_866_; lean_object* v___x_868_; uint8_t v_isShared_869_; uint8_t v_isSharedCheck_881_; 
v_proof_865_ = lean_ctor_get(v_a_821_, 0);
lean_inc_ref(v_proof_865_);
lean_dec_ref_known(v_a_821_, 1);
v_proof_866_ = lean_ctor_get(v_a_823_, 0);
v_isSharedCheck_881_ = !lean_is_exclusive(v_a_823_);
if (v_isSharedCheck_881_ == 0)
{
v___x_868_ = v_a_823_;
v_isShared_869_ = v_isSharedCheck_881_;
goto v_resetjp_867_;
}
else
{
lean_inc(v_proof_866_);
lean_dec(v_a_823_);
v___x_868_ = lean_box(0);
v_isShared_869_ = v_isSharedCheck_881_;
goto v_resetjp_867_;
}
v_resetjp_867_:
{
lean_object* v___x_870_; lean_object* v___x_871_; lean_object* v___x_872_; lean_object* v___x_873_; lean_object* v___x_874_; lean_object* v___x_876_; 
v___x_870_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__9, &lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__9_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__9);
v___x_871_ = l_Lean_Expr_app___override(v___x_870_, v_snd_804_);
v___x_872_ = l_Lean_Expr_app___override(v___x_871_, v_arg_793_);
v___x_873_ = l_Lean_Expr_app___override(v___x_872_, v_proof_865_);
v___x_874_ = l_Lean_Expr_app___override(v___x_873_, v_proof_866_);
if (v_isShared_869_ == 0)
{
lean_ctor_set(v___x_868_, 0, v___x_874_);
v___x_876_ = v___x_868_;
goto v_reusejp_875_;
}
else
{
lean_object* v_reuseFailAlloc_880_; 
v_reuseFailAlloc_880_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_880_, 0, v___x_874_);
lean_ctor_set_uint8(v_reuseFailAlloc_880_, sizeof(void*)*1, v_val_864_);
v___x_876_ = v_reuseFailAlloc_880_;
goto v_reusejp_875_;
}
v_reusejp_875_:
{
lean_object* v___x_878_; 
if (v_isShared_826_ == 0)
{
lean_ctor_set(v___x_825_, 0, v___x_876_);
v___x_878_ = v___x_825_;
goto v_reusejp_877_;
}
else
{
lean_object* v_reuseFailAlloc_879_; 
v_reuseFailAlloc_879_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_879_, 0, v___x_876_);
v___x_878_ = v_reuseFailAlloc_879_;
goto v_reusejp_877_;
}
v_reusejp_877_:
{
return v___x_878_;
}
}
}
}
else
{
lean_object* v_proof_882_; lean_object* v_proof_883_; lean_object* v___x_885_; uint8_t v_isShared_886_; uint8_t v_isSharedCheck_898_; 
v_proof_882_ = lean_ctor_get(v_a_821_, 0);
lean_inc_ref(v_proof_882_);
lean_dec_ref_known(v_a_821_, 1);
v_proof_883_ = lean_ctor_get(v_a_823_, 0);
v_isSharedCheck_898_ = !lean_is_exclusive(v_a_823_);
if (v_isSharedCheck_898_ == 0)
{
v___x_885_ = v_a_823_;
v_isShared_886_ = v_isSharedCheck_898_;
goto v_resetjp_884_;
}
else
{
lean_inc(v_proof_883_);
lean_dec(v_a_823_);
v___x_885_ = lean_box(0);
v_isShared_886_ = v_isSharedCheck_898_;
goto v_resetjp_884_;
}
v_resetjp_884_:
{
lean_object* v___x_887_; lean_object* v___x_888_; lean_object* v___x_889_; lean_object* v___x_890_; lean_object* v___x_891_; lean_object* v___x_893_; 
v___x_887_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__12, &lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__12_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__12);
v___x_888_ = l_Lean_Expr_app___override(v___x_887_, v_snd_804_);
v___x_889_ = l_Lean_Expr_app___override(v___x_888_, v_arg_793_);
v___x_890_ = l_Lean_Expr_app___override(v___x_889_, v_proof_882_);
v___x_891_ = l_Lean_Expr_app___override(v___x_890_, v_proof_883_);
if (v_isShared_886_ == 0)
{
lean_ctor_set(v___x_885_, 0, v___x_891_);
v___x_893_ = v___x_885_;
goto v_reusejp_892_;
}
else
{
lean_object* v_reuseFailAlloc_897_; 
v_reuseFailAlloc_897_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_897_, 0, v___x_891_);
lean_ctor_set_uint8(v_reuseFailAlloc_897_, sizeof(void*)*1, v_val_864_);
v___x_893_ = v_reuseFailAlloc_897_;
goto v_reusejp_892_;
}
v_reusejp_892_:
{
lean_object* v___x_895_; 
if (v_isShared_826_ == 0)
{
lean_ctor_set(v___x_825_, 0, v___x_893_);
v___x_895_ = v___x_825_;
goto v_reusejp_894_;
}
else
{
lean_object* v_reuseFailAlloc_896_; 
v_reuseFailAlloc_896_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_896_, 0, v___x_893_);
v___x_895_ = v_reuseFailAlloc_896_;
goto v_reusejp_894_;
}
v_reusejp_894_:
{
return v___x_895_;
}
}
}
}
}
}
case 4:
{
lean_dec_ref_known(v_a_823_, 5);
lean_del_object(v___x_825_);
lean_dec_ref_known(v_a_821_, 1);
lean_dec(v_snd_804_);
lean_dec_ref(v_arg_793_);
v___y_777_ = v___y_764_;
v___y_778_ = v___y_765_;
v___y_779_ = v___y_766_;
v___y_780_ = v___y_767_;
goto v___jp_776_;
}
case 3:
{
lean_dec_ref_known(v_a_823_, 5);
lean_del_object(v___x_825_);
lean_dec_ref_known(v_a_821_, 1);
lean_dec(v_snd_804_);
lean_dec_ref(v_arg_793_);
v___y_777_ = v___y_764_;
v___y_778_ = v___y_765_;
v___y_779_ = v___y_766_;
v___y_780_ = v___y_767_;
goto v___jp_776_;
}
case 2:
{
lean_dec_ref_known(v_a_823_, 3);
lean_del_object(v___x_825_);
lean_dec_ref_known(v_a_821_, 1);
lean_dec(v_snd_804_);
lean_dec_ref(v_arg_793_);
v___y_777_ = v___y_764_;
v___y_778_ = v___y_765_;
v___y_779_ = v___y_766_;
v___y_780_ = v___y_767_;
goto v___jp_776_;
}
default: 
{
lean_del_object(v___x_825_);
lean_dec_ref_known(v_a_821_, 1);
lean_dec(v_a_823_);
lean_dec(v_snd_804_);
lean_dec_ref(v_arg_793_);
v___y_777_ = v___y_764_;
v___y_778_ = v___y_765_;
v___y_779_ = v___y_766_;
v___y_780_ = v___y_767_;
goto v___jp_776_;
}
}
}
}
case 1:
{
lean_object* v_a_900_; lean_object* v___x_902_; uint8_t v_isShared_903_; uint8_t v_isSharedCheck_972_; 
v_a_900_ = lean_ctor_get(v___x_822_, 0);
v_isSharedCheck_972_ = !lean_is_exclusive(v___x_822_);
if (v_isSharedCheck_972_ == 0)
{
v___x_902_ = v___x_822_;
v_isShared_903_ = v_isSharedCheck_972_;
goto v_resetjp_901_;
}
else
{
lean_inc(v_a_900_);
lean_dec(v___x_822_);
v___x_902_ = lean_box(0);
v_isShared_903_ = v_isSharedCheck_972_;
goto v_resetjp_901_;
}
v_resetjp_901_:
{
switch(lean_obj_tag(v_a_900_))
{
case 0:
{
lean_dec_ref_known(v_a_900_, 1);
lean_del_object(v___x_902_);
lean_dec_ref_known(v_a_821_, 3);
lean_dec(v_snd_804_);
lean_dec(v_fst_803_);
lean_del_object(v___x_801_);
lean_dec(v_fst_799_);
lean_dec_ref(v_arg_793_);
v___y_770_ = v___y_764_;
v___y_771_ = v___y_765_;
v___y_772_ = v___y_766_;
v___y_773_ = v___y_767_;
goto v___jp_769_;
}
case 1:
{
lean_object* v_inst_904_; lean_object* v_lit_905_; lean_object* v_proof_906_; lean_object* v_inst_907_; lean_object* v_lit_908_; lean_object* v_proof_909_; lean_object* v___x_910_; lean_object* v___x_911_; uint8_t v___x_912_; 
v_inst_904_ = lean_ctor_get(v_a_821_, 0);
lean_inc_ref(v_inst_904_);
v_lit_905_ = lean_ctor_get(v_a_821_, 1);
lean_inc_ref(v_lit_905_);
v_proof_906_ = lean_ctor_get(v_a_821_, 2);
lean_inc_ref(v_proof_906_);
lean_dec_ref_known(v_a_821_, 3);
v_inst_907_ = lean_ctor_get(v_a_900_, 0);
lean_inc_ref(v_inst_907_);
v_lit_908_ = lean_ctor_get(v_a_900_, 1);
lean_inc_ref(v_lit_908_);
v_proof_909_ = lean_ctor_get(v_a_900_, 2);
lean_inc_ref(v_proof_909_);
lean_dec_ref_known(v_a_900_, 3);
v___x_910_ = lp_batteries_Lean_Expr_natLit_x21(v_lit_905_);
v___x_911_ = lp_batteries_Lean_Expr_natLit_x21(v_lit_908_);
v___x_912_ = lean_nat_dec_eq(v___x_910_, v___x_911_);
lean_dec(v___x_911_);
lean_dec(v___x_910_);
if (v___x_912_ == 0)
{
lean_object* v___x_913_; 
lean_del_object(v___x_902_);
lean_inc(v_fst_803_);
lean_inc(v_fst_799_);
v___x_913_ = lp_mathlib_Mathlib_Meta_NormNum_inferCharZeroOfAddMonoidWithOne_x3f(v_fst_799_, v_fst_803_, v_inst_907_, v___y_764_, v___y_765_, v___y_766_, v___y_767_);
if (lean_obj_tag(v___x_913_) == 0)
{
lean_object* v_a_914_; lean_object* v___x_916_; uint8_t v_isShared_917_; uint8_t v_isSharedCheck_941_; 
v_a_914_ = lean_ctor_get(v___x_913_, 0);
v_isSharedCheck_941_ = !lean_is_exclusive(v___x_913_);
if (v_isSharedCheck_941_ == 0)
{
v___x_916_ = v___x_913_;
v_isShared_917_ = v_isSharedCheck_941_;
goto v_resetjp_915_;
}
else
{
lean_inc(v_a_914_);
lean_dec(v___x_913_);
v___x_916_ = lean_box(0);
v_isShared_917_ = v_isSharedCheck_941_;
goto v_resetjp_915_;
}
v_resetjp_915_:
{
if (lean_obj_tag(v_a_914_) == 1)
{
lean_object* v_val_918_; lean_object* v___x_919_; lean_object* v___x_920_; lean_object* v___x_922_; 
v_val_918_ = lean_ctor_get(v_a_914_, 0);
lean_inc(v_val_918_);
lean_dec_ref_known(v_a_914_, 1);
v___x_919_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__13, &lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__13_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__13);
v___x_920_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__14));
if (v_isShared_802_ == 0)
{
lean_ctor_set_tag(v___x_801_, 1);
lean_ctor_set(v___x_801_, 1, v___x_810_);
v___x_922_ = v___x_801_;
goto v_reusejp_921_;
}
else
{
lean_object* v_reuseFailAlloc_938_; 
v_reuseFailAlloc_938_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_938_, 0, v_fst_799_);
lean_ctor_set(v_reuseFailAlloc_938_, 1, v___x_810_);
v___x_922_ = v_reuseFailAlloc_938_;
goto v_reusejp_921_;
}
v_reusejp_921_:
{
lean_object* v___x_923_; lean_object* v___x_924_; lean_object* v___x_925_; lean_object* v___x_926_; lean_object* v___x_927_; lean_object* v___x_928_; lean_object* v___x_929_; lean_object* v___x_930_; lean_object* v___x_931_; lean_object* v___x_932_; lean_object* v___x_933_; lean_object* v___x_934_; lean_object* v___x_936_; 
v___x_923_ = l_Lean_Expr_const___override(v___x_920_, v___x_922_);
v___x_924_ = l_Lean_Expr_app___override(v___x_923_, v_fst_803_);
v___x_925_ = l_Lean_Expr_app___override(v___x_924_, v_inst_904_);
v___x_926_ = l_Lean_Expr_app___override(v___x_925_, v_val_918_);
v___x_927_ = l_Lean_Expr_app___override(v___x_926_, v_snd_804_);
v___x_928_ = l_Lean_Expr_app___override(v___x_927_, v_arg_793_);
v___x_929_ = l_Lean_Expr_app___override(v___x_928_, v_lit_905_);
v___x_930_ = l_Lean_Expr_app___override(v___x_929_, v_lit_908_);
v___x_931_ = l_Lean_Expr_app___override(v___x_930_, v_proof_906_);
v___x_932_ = l_Lean_Expr_app___override(v___x_931_, v_proof_909_);
v___x_933_ = l_Lean_Expr_app___override(v___x_932_, v___x_919_);
v___x_934_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_934_, 0, v___x_933_);
lean_ctor_set_uint8(v___x_934_, sizeof(void*)*1, v___x_816_);
if (v_isShared_917_ == 0)
{
lean_ctor_set(v___x_916_, 0, v___x_934_);
v___x_936_ = v___x_916_;
goto v_reusejp_935_;
}
else
{
lean_object* v_reuseFailAlloc_937_; 
v_reuseFailAlloc_937_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_937_, 0, v___x_934_);
v___x_936_ = v_reuseFailAlloc_937_;
goto v_reusejp_935_;
}
v_reusejp_935_:
{
return v___x_936_;
}
}
}
else
{
lean_object* v___x_939_; lean_object* v___x_940_; 
lean_del_object(v___x_916_);
lean_dec(v_a_914_);
lean_dec_ref(v_proof_909_);
lean_dec_ref(v_lit_908_);
lean_dec_ref(v_proof_906_);
lean_dec_ref(v_lit_905_);
lean_dec_ref(v_inst_904_);
lean_dec(v_snd_804_);
lean_dec(v_fst_803_);
lean_del_object(v___x_801_);
lean_dec(v_fst_799_);
lean_dec_ref(v_arg_793_);
v___x_939_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20, &lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20);
v___x_940_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm_spec__0___redArg(v___x_939_, v___y_764_, v___y_765_, v___y_766_, v___y_767_);
return v___x_940_;
}
}
}
else
{
lean_object* v_a_942_; lean_object* v___x_944_; uint8_t v_isShared_945_; uint8_t v_isSharedCheck_949_; 
lean_dec_ref(v_proof_909_);
lean_dec_ref(v_lit_908_);
lean_dec_ref(v_proof_906_);
lean_dec_ref(v_lit_905_);
lean_dec_ref(v_inst_904_);
lean_dec(v_snd_804_);
lean_dec(v_fst_803_);
lean_del_object(v___x_801_);
lean_dec(v_fst_799_);
lean_dec_ref(v_arg_793_);
v_a_942_ = lean_ctor_get(v___x_913_, 0);
v_isSharedCheck_949_ = !lean_is_exclusive(v___x_913_);
if (v_isSharedCheck_949_ == 0)
{
v___x_944_ = v___x_913_;
v_isShared_945_ = v_isSharedCheck_949_;
goto v_resetjp_943_;
}
else
{
lean_inc(v_a_942_);
lean_dec(v___x_913_);
v___x_944_ = lean_box(0);
v_isShared_945_ = v_isSharedCheck_949_;
goto v_resetjp_943_;
}
v_resetjp_943_:
{
lean_object* v___x_947_; 
if (v_isShared_945_ == 0)
{
v___x_947_ = v___x_944_;
goto v_reusejp_946_;
}
else
{
lean_object* v_reuseFailAlloc_948_; 
v_reuseFailAlloc_948_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_948_, 0, v_a_942_);
v___x_947_ = v_reuseFailAlloc_948_;
goto v_reusejp_946_;
}
v_reusejp_946_:
{
return v___x_947_;
}
}
}
}
else
{
lean_object* v___x_950_; lean_object* v___x_952_; 
lean_dec_ref(v_lit_908_);
lean_dec_ref(v_inst_907_);
v___x_950_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__16));
if (v_isShared_802_ == 0)
{
lean_ctor_set_tag(v___x_801_, 1);
lean_ctor_set(v___x_801_, 1, v___x_810_);
v___x_952_ = v___x_801_;
goto v_reusejp_951_;
}
else
{
lean_object* v_reuseFailAlloc_965_; 
v_reuseFailAlloc_965_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_965_, 0, v_fst_799_);
lean_ctor_set(v_reuseFailAlloc_965_, 1, v___x_810_);
v___x_952_ = v_reuseFailAlloc_965_;
goto v_reusejp_951_;
}
v_reusejp_951_:
{
lean_object* v___x_953_; lean_object* v___x_954_; lean_object* v___x_955_; lean_object* v___x_956_; lean_object* v___x_957_; lean_object* v___x_958_; lean_object* v___x_959_; lean_object* v___x_960_; lean_object* v___x_961_; lean_object* v___x_963_; 
v___x_953_ = l_Lean_Expr_const___override(v___x_950_, v___x_952_);
v___x_954_ = l_Lean_Expr_app___override(v___x_953_, v_fst_803_);
v___x_955_ = l_Lean_Expr_app___override(v___x_954_, v_inst_904_);
v___x_956_ = l_Lean_Expr_app___override(v___x_955_, v_snd_804_);
v___x_957_ = l_Lean_Expr_app___override(v___x_956_, v_arg_793_);
v___x_958_ = l_Lean_Expr_app___override(v___x_957_, v_lit_905_);
v___x_959_ = l_Lean_Expr_app___override(v___x_958_, v_proof_906_);
v___x_960_ = l_Lean_Expr_app___override(v___x_959_, v_proof_909_);
v___x_961_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_961_, 0, v___x_960_);
lean_ctor_set_uint8(v___x_961_, sizeof(void*)*1, v___x_912_);
if (v_isShared_903_ == 0)
{
lean_ctor_set(v___x_902_, 0, v___x_961_);
v___x_963_ = v___x_902_;
goto v_reusejp_962_;
}
else
{
lean_object* v_reuseFailAlloc_964_; 
v_reuseFailAlloc_964_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_964_, 0, v___x_961_);
v___x_963_ = v_reuseFailAlloc_964_;
goto v_reusejp_962_;
}
v_reusejp_962_:
{
return v___x_963_;
}
}
}
}
case 2:
{
lean_object* v_inst_966_; lean_object* v___x_967_; 
lean_del_object(v___x_902_);
lean_del_object(v___x_801_);
v_inst_966_ = lean_ctor_get(v_a_900_, 0);
lean_inc_ref(v_inst_966_);
v___x_967_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg(v_fst_799_, v_fst_803_, v_snd_804_, v_arg_793_, v_a_821_, v_a_900_, v_inst_966_, v___y_764_, v___y_765_, v___y_766_, v___y_767_);
return v___x_967_;
}
case 3:
{
lean_object* v_inst_968_; lean_object* v___x_969_; 
lean_del_object(v___x_902_);
lean_del_object(v___x_801_);
v_inst_968_ = lean_ctor_get(v_a_900_, 0);
lean_inc_ref(v_inst_968_);
v___x_969_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg(v_fst_799_, v_fst_803_, v_snd_804_, v_arg_793_, v_a_821_, v_a_900_, v_inst_968_, v___y_764_, v___y_765_, v___y_766_, v___y_767_);
return v___x_969_;
}
default: 
{
lean_object* v_inst_970_; lean_object* v___x_971_; 
lean_del_object(v___x_902_);
lean_del_object(v___x_801_);
v_inst_970_ = lean_ctor_get(v_a_900_, 0);
lean_inc_ref(v_inst_970_);
v___x_971_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg(v_fst_799_, v_fst_803_, v_snd_804_, v_arg_793_, v_a_821_, v_a_900_, v_inst_970_, v___y_764_, v___y_765_, v___y_766_, v___y_767_);
return v___x_971_;
}
}
}
}
case 2:
{
lean_object* v_a_973_; 
v_a_973_ = lean_ctor_get(v___x_822_, 0);
lean_inc(v_a_973_);
lean_dec_ref_known(v___x_822_, 1);
switch(lean_obj_tag(v_a_973_))
{
case 0:
{
lean_dec_ref_known(v_a_973_, 1);
lean_dec_ref_known(v_a_821_, 3);
lean_dec(v_snd_804_);
lean_dec(v_fst_803_);
lean_del_object(v___x_801_);
lean_dec(v_fst_799_);
lean_dec_ref(v_arg_793_);
v___y_770_ = v___y_764_;
v___y_771_ = v___y_765_;
v___y_772_ = v___y_766_;
v___y_773_ = v___y_767_;
goto v___jp_769_;
}
case 4:
{
lean_object* v_inst_974_; lean_object* v___x_975_; 
lean_del_object(v___x_801_);
v_inst_974_ = lean_ctor_get(v_a_973_, 0);
lean_inc_ref(v_inst_974_);
v___x_975_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg(v_fst_799_, v_fst_803_, v_snd_804_, v_arg_793_, v_a_821_, v_a_973_, v_inst_974_, v___y_764_, v___y_765_, v___y_766_, v___y_767_);
return v___x_975_;
}
case 3:
{
lean_object* v___x_976_; lean_object* v___x_978_; 
v___x_976_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__17));
lean_inc(v_fst_799_);
if (v_isShared_802_ == 0)
{
lean_ctor_set_tag(v___x_801_, 1);
lean_ctor_set(v___x_801_, 1, v___x_810_);
v___x_978_ = v___x_801_;
goto v_reusejp_977_;
}
else
{
lean_object* v_reuseFailAlloc_992_; 
v_reuseFailAlloc_992_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_992_, 0, v_fst_799_);
lean_ctor_set(v_reuseFailAlloc_992_, 1, v___x_810_);
v___x_978_ = v_reuseFailAlloc_992_;
goto v_reusejp_977_;
}
v_reusejp_977_:
{
lean_object* v___x_979_; lean_object* v___x_980_; lean_object* v___x_981_; 
v___x_979_ = l_Lean_Expr_const___override(v___x_976_, v___x_978_);
lean_inc(v_fst_803_);
v___x_980_ = l_Lean_Expr_app___override(v___x_979_, v_fst_803_);
v___x_981_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_980_, v___y_764_, v___y_765_, v___y_766_, v___y_767_);
if (lean_obj_tag(v___x_981_) == 0)
{
lean_object* v_a_982_; lean_object* v___x_983_; 
v_a_982_ = lean_ctor_get(v___x_981_, 0);
lean_inc(v_a_982_);
lean_dec_ref_known(v___x_981_, 1);
v___x_983_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg(v_fst_799_, v_fst_803_, v_snd_804_, v_arg_793_, v_a_821_, v_a_973_, v_a_982_, v___y_764_, v___y_765_, v___y_766_, v___y_767_);
return v___x_983_;
}
else
{
lean_object* v_a_984_; lean_object* v___x_986_; uint8_t v_isShared_987_; uint8_t v_isSharedCheck_991_; 
lean_dec_ref_known(v_a_973_, 5);
lean_dec_ref_known(v_a_821_, 3);
lean_dec(v_snd_804_);
lean_dec(v_fst_803_);
lean_dec(v_fst_799_);
lean_dec_ref(v_arg_793_);
v_a_984_ = lean_ctor_get(v___x_981_, 0);
v_isSharedCheck_991_ = !lean_is_exclusive(v___x_981_);
if (v_isSharedCheck_991_ == 0)
{
v___x_986_ = v___x_981_;
v_isShared_987_ = v_isSharedCheck_991_;
goto v_resetjp_985_;
}
else
{
lean_inc(v_a_984_);
lean_dec(v___x_981_);
v___x_986_ = lean_box(0);
v_isShared_987_ = v_isSharedCheck_991_;
goto v_resetjp_985_;
}
v_resetjp_985_:
{
lean_object* v___x_989_; 
if (v_isShared_987_ == 0)
{
v___x_989_ = v___x_986_;
goto v_reusejp_988_;
}
else
{
lean_object* v_reuseFailAlloc_990_; 
v_reuseFailAlloc_990_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_990_, 0, v_a_984_);
v___x_989_ = v_reuseFailAlloc_990_;
goto v_reusejp_988_;
}
v_reusejp_988_:
{
return v___x_989_;
}
}
}
}
}
case 2:
{
lean_object* v_inst_993_; lean_object* v___x_994_; 
lean_del_object(v___x_801_);
v_inst_993_ = lean_ctor_get(v_a_821_, 0);
lean_inc_ref(v_inst_993_);
v___x_994_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg(v_fst_799_, v_fst_803_, v_snd_804_, v_arg_793_, v_a_821_, v_a_973_, v_inst_993_, v___y_764_, v___y_765_, v___y_766_, v___y_767_);
return v___x_994_;
}
default: 
{
lean_object* v_inst_995_; lean_object* v___x_996_; 
lean_del_object(v___x_801_);
v_inst_995_ = lean_ctor_get(v_a_821_, 0);
lean_inc_ref(v_inst_995_);
v___x_996_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg(v_fst_799_, v_fst_803_, v_snd_804_, v_arg_793_, v_a_821_, v_a_973_, v_inst_995_, v___y_764_, v___y_765_, v___y_766_, v___y_767_);
return v___x_996_;
}
}
}
case 3:
{
lean_object* v_a_997_; 
v_a_997_ = lean_ctor_get(v___x_822_, 0);
lean_inc(v_a_997_);
lean_dec_ref_known(v___x_822_, 1);
switch(lean_obj_tag(v_a_997_))
{
case 0:
{
lean_dec_ref_known(v_a_997_, 1);
lean_dec_ref_known(v_a_821_, 5);
lean_dec(v_snd_804_);
lean_dec(v_fst_803_);
lean_del_object(v___x_801_);
lean_dec(v_fst_799_);
lean_dec_ref(v_arg_793_);
v___y_770_ = v___y_764_;
v___y_771_ = v___y_765_;
v___y_772_ = v___y_766_;
v___y_773_ = v___y_767_;
goto v___jp_769_;
}
case 4:
{
lean_object* v_inst_998_; lean_object* v___x_999_; 
lean_del_object(v___x_801_);
v_inst_998_ = lean_ctor_get(v_a_997_, 0);
lean_inc_ref(v_inst_998_);
v___x_999_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg(v_fst_799_, v_fst_803_, v_snd_804_, v_arg_793_, v_a_821_, v_a_997_, v_inst_998_, v___y_764_, v___y_765_, v___y_766_, v___y_767_);
return v___x_999_;
}
case 2:
{
lean_object* v___x_1000_; lean_object* v___x_1002_; 
v___x_1000_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___closed__17));
lean_inc(v_fst_799_);
if (v_isShared_802_ == 0)
{
lean_ctor_set_tag(v___x_801_, 1);
lean_ctor_set(v___x_801_, 1, v___x_810_);
v___x_1002_ = v___x_801_;
goto v_reusejp_1001_;
}
else
{
lean_object* v_reuseFailAlloc_1016_; 
v_reuseFailAlloc_1016_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1016_, 0, v_fst_799_);
lean_ctor_set(v_reuseFailAlloc_1016_, 1, v___x_810_);
v___x_1002_ = v_reuseFailAlloc_1016_;
goto v_reusejp_1001_;
}
v_reusejp_1001_:
{
lean_object* v___x_1003_; lean_object* v___x_1004_; lean_object* v___x_1005_; 
v___x_1003_ = l_Lean_Expr_const___override(v___x_1000_, v___x_1002_);
lean_inc(v_fst_803_);
v___x_1004_ = l_Lean_Expr_app___override(v___x_1003_, v_fst_803_);
v___x_1005_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_1004_, v___y_764_, v___y_765_, v___y_766_, v___y_767_);
if (lean_obj_tag(v___x_1005_) == 0)
{
lean_object* v_a_1006_; lean_object* v___x_1007_; 
v_a_1006_ = lean_ctor_get(v___x_1005_, 0);
lean_inc(v_a_1006_);
lean_dec_ref_known(v___x_1005_, 1);
v___x_1007_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg(v_fst_799_, v_fst_803_, v_snd_804_, v_arg_793_, v_a_821_, v_a_997_, v_a_1006_, v___y_764_, v___y_765_, v___y_766_, v___y_767_);
return v___x_1007_;
}
else
{
lean_object* v_a_1008_; lean_object* v___x_1010_; uint8_t v_isShared_1011_; uint8_t v_isSharedCheck_1015_; 
lean_dec_ref_known(v_a_997_, 3);
lean_dec_ref_known(v_a_821_, 5);
lean_dec(v_snd_804_);
lean_dec(v_fst_803_);
lean_dec(v_fst_799_);
lean_dec_ref(v_arg_793_);
v_a_1008_ = lean_ctor_get(v___x_1005_, 0);
v_isSharedCheck_1015_ = !lean_is_exclusive(v___x_1005_);
if (v_isSharedCheck_1015_ == 0)
{
v___x_1010_ = v___x_1005_;
v_isShared_1011_ = v_isSharedCheck_1015_;
goto v_resetjp_1009_;
}
else
{
lean_inc(v_a_1008_);
lean_dec(v___x_1005_);
v___x_1010_ = lean_box(0);
v_isShared_1011_ = v_isSharedCheck_1015_;
goto v_resetjp_1009_;
}
v_resetjp_1009_:
{
lean_object* v___x_1013_; 
if (v_isShared_1011_ == 0)
{
v___x_1013_ = v___x_1010_;
goto v_reusejp_1012_;
}
else
{
lean_object* v_reuseFailAlloc_1014_; 
v_reuseFailAlloc_1014_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1014_, 0, v_a_1008_);
v___x_1013_ = v_reuseFailAlloc_1014_;
goto v_reusejp_1012_;
}
v_reusejp_1012_:
{
return v___x_1013_;
}
}
}
}
}
case 3:
{
lean_object* v_inst_1017_; lean_object* v___x_1018_; 
lean_del_object(v___x_801_);
v_inst_1017_ = lean_ctor_get(v_a_821_, 0);
lean_inc_ref(v_inst_1017_);
v___x_1018_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg(v_fst_799_, v_fst_803_, v_snd_804_, v_arg_793_, v_a_821_, v_a_997_, v_inst_1017_, v___y_764_, v___y_765_, v___y_766_, v___y_767_);
return v___x_1018_;
}
default: 
{
lean_object* v_inst_1019_; lean_object* v___x_1020_; 
lean_del_object(v___x_801_);
v_inst_1019_ = lean_ctor_get(v_a_821_, 0);
lean_inc_ref(v_inst_1019_);
v___x_1020_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_nnratArm___redArg(v_fst_799_, v_fst_803_, v_snd_804_, v_arg_793_, v_a_821_, v_a_997_, v_inst_1019_, v___y_764_, v___y_765_, v___y_766_, v___y_767_);
return v___x_1020_;
}
}
}
default: 
{
lean_object* v_a_1021_; 
lean_del_object(v___x_801_);
v_a_1021_ = lean_ctor_get(v___x_822_, 0);
lean_inc(v_a_1021_);
lean_dec_ref_known(v___x_822_, 1);
switch(lean_obj_tag(v_a_1021_))
{
case 0:
{
lean_dec_ref_known(v_a_1021_, 1);
lean_dec_ref_known(v_a_821_, 5);
lean_dec(v_snd_804_);
lean_dec(v_fst_803_);
lean_dec(v_fst_799_);
lean_dec_ref(v_arg_793_);
v___y_770_ = v___y_764_;
v___y_771_ = v___y_765_;
v___y_772_ = v___y_766_;
v___y_773_ = v___y_767_;
goto v___jp_769_;
}
case 4:
{
lean_object* v_inst_1022_; lean_object* v___x_1023_; 
v_inst_1022_ = lean_ctor_get(v_a_821_, 0);
lean_inc_ref(v_inst_1022_);
v___x_1023_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg(v_fst_799_, v_fst_803_, v_snd_804_, v_arg_793_, v_a_821_, v_a_1021_, v_inst_1022_, v___y_764_, v___y_765_, v___y_766_, v___y_767_);
return v___x_1023_;
}
case 3:
{
lean_object* v_inst_1024_; lean_object* v___x_1025_; 
v_inst_1024_ = lean_ctor_get(v_a_821_, 0);
lean_inc_ref(v_inst_1024_);
v___x_1025_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg(v_fst_799_, v_fst_803_, v_snd_804_, v_arg_793_, v_a_821_, v_a_1021_, v_inst_1024_, v___y_764_, v___y_765_, v___y_766_, v___y_767_);
return v___x_1025_;
}
case 2:
{
lean_object* v_inst_1026_; lean_object* v___x_1027_; 
v_inst_1026_ = lean_ctor_get(v_a_821_, 0);
lean_inc_ref(v_inst_1026_);
v___x_1027_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg(v_fst_799_, v_fst_803_, v_snd_804_, v_arg_793_, v_a_821_, v_a_1021_, v_inst_1026_, v___y_764_, v___y_765_, v___y_766_, v___y_767_);
return v___x_1027_;
}
default: 
{
lean_object* v_inst_1028_; lean_object* v___x_1029_; 
v_inst_1028_ = lean_ctor_get(v_a_821_, 0);
lean_inc_ref(v_inst_1028_);
v___x_1029_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_ratArm___redArg(v_fst_799_, v_fst_803_, v_snd_804_, v_arg_793_, v_a_821_, v_a_1021_, v_inst_1028_, v___y_764_, v___y_765_, v___y_766_, v___y_767_);
return v___x_1029_;
}
}
}
}
}
else
{
lean_dec(v_a_821_);
lean_dec(v_snd_804_);
lean_dec(v_fst_803_);
lean_del_object(v___x_801_);
lean_dec(v_fst_799_);
lean_dec_ref(v_arg_793_);
return v___x_822_;
}
}
else
{
lean_dec(v_snd_804_);
lean_dec(v_fst_803_);
lean_del_object(v___x_801_);
lean_dec(v_fst_799_);
lean_dec_ref(v_arg_793_);
return v___x_820_;
}
}
}
else
{
lean_object* v_a_1041_; lean_object* v___x_1043_; uint8_t v_isShared_1044_; uint8_t v_isSharedCheck_1048_; 
lean_dec(v_snd_804_);
lean_dec(v_fst_803_);
lean_del_object(v___x_801_);
lean_dec(v_fst_799_);
lean_dec_ref(v_arg_793_);
v_a_1041_ = lean_ctor_get(v___x_817_, 0);
v_isSharedCheck_1048_ = !lean_is_exclusive(v___x_817_);
if (v_isSharedCheck_1048_ == 0)
{
v___x_1043_ = v___x_817_;
v_isShared_1044_ = v_isSharedCheck_1048_;
goto v_resetjp_1042_;
}
else
{
lean_inc(v_a_1041_);
lean_dec(v___x_817_);
v___x_1043_ = lean_box(0);
v_isShared_1044_ = v_isSharedCheck_1048_;
goto v_resetjp_1042_;
}
v_resetjp_1042_:
{
lean_object* v___x_1046_; 
if (v_isShared_1044_ == 0)
{
v___x_1046_ = v___x_1043_;
goto v_reusejp_1045_;
}
else
{
lean_object* v_reuseFailAlloc_1047_; 
v_reuseFailAlloc_1047_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1047_, 0, v_a_1041_);
v___x_1046_ = v_reuseFailAlloc_1047_;
goto v_reusejp_1045_;
}
v_reusejp_1045_:
{
return v___x_1046_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_1052_; lean_object* v___x_1054_; uint8_t v_isShared_1055_; uint8_t v_isSharedCheck_1059_; 
lean_dec_ref(v_fn_794_);
lean_dec_ref(v_arg_793_);
v_a_1052_ = lean_ctor_get(v___x_796_, 0);
v_isSharedCheck_1059_ = !lean_is_exclusive(v___x_796_);
if (v_isSharedCheck_1059_ == 0)
{
v___x_1054_ = v___x_796_;
v_isShared_1055_ = v_isSharedCheck_1059_;
goto v_resetjp_1053_;
}
else
{
lean_inc(v_a_1052_);
lean_dec(v___x_796_);
v___x_1054_ = lean_box(0);
v_isShared_1055_ = v_isSharedCheck_1059_;
goto v_resetjp_1053_;
}
v_resetjp_1053_:
{
lean_object* v___x_1057_; 
if (v_isShared_1055_ == 0)
{
v___x_1057_ = v___x_1054_;
goto v_reusejp_1056_;
}
else
{
lean_object* v_reuseFailAlloc_1058_; 
v_reuseFailAlloc_1058_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1058_, 0, v_a_1052_);
v___x_1057_ = v_reuseFailAlloc_1058_;
goto v_reusejp_1056_;
}
v_reusejp_1056_:
{
return v___x_1057_;
}
}
}
}
else
{
lean_dec_ref(v_fn_792_);
lean_dec_ref_known(v_a_784_, 2);
v___y_786_ = v___y_764_;
v___y_787_ = v___y_765_;
v___y_788_ = v___y_766_;
v___y_789_ = v___y_767_;
goto v___jp_785_;
}
}
else
{
lean_dec(v_a_784_);
v___y_786_ = v___y_764_;
v___y_787_ = v___y_765_;
v___y_788_ = v___y_766_;
v___y_789_ = v___y_767_;
goto v___jp_785_;
}
v___jp_785_:
{
lean_object* v___x_790_; lean_object* v___x_791_; 
v___x_790_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20, &lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20);
v___x_791_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm_spec__0___redArg(v___x_790_, v___y_786_, v___y_787_, v___y_788_, v___y_789_);
return v___x_791_;
}
}
else
{
lean_object* v_a_1060_; lean_object* v___x_1062_; uint8_t v_isShared_1063_; uint8_t v_isSharedCheck_1067_; 
v_a_1060_ = lean_ctor_get(v___x_783_, 0);
v_isSharedCheck_1067_ = !lean_is_exclusive(v___x_783_);
if (v_isSharedCheck_1067_ == 0)
{
v___x_1062_ = v___x_783_;
v_isShared_1063_ = v_isSharedCheck_1067_;
goto v_resetjp_1061_;
}
else
{
lean_inc(v_a_1060_);
lean_dec(v___x_783_);
v___x_1062_ = lean_box(0);
v_isShared_1063_ = v_isSharedCheck_1067_;
goto v_resetjp_1061_;
}
v_resetjp_1061_:
{
lean_object* v___x_1065_; 
if (v_isShared_1063_ == 0)
{
v___x_1065_ = v___x_1062_;
goto v_reusejp_1064_;
}
else
{
lean_object* v_reuseFailAlloc_1066_; 
v_reuseFailAlloc_1066_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1066_, 0, v_a_1060_);
v___x_1065_ = v_reuseFailAlloc_1066_;
goto v_reusejp_1064_;
}
v_reusejp_1064_:
{
return v___x_1065_;
}
}
}
v___jp_769_:
{
lean_object* v___x_774_; lean_object* v___x_775_; 
v___x_774_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20, &lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20);
v___x_775_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm_spec__0___redArg(v___x_774_, v___y_770_, v___y_771_, v___y_772_, v___y_773_);
return v___x_775_;
}
v___jp_776_:
{
lean_object* v___x_781_; lean_object* v___x_782_; 
v___x_781_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20, &lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm___redArg___closed__20);
v___x_782_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_NormNum_Eq_0__Mathlib_Meta_NormNum_evalEq_intArm_spec__0___redArg(v___x_781_, v___y_777_, v___y_778_, v___y_779_, v___y_780_);
return v___x_782_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1___boxed(lean_object* v_v_1068_, lean_object* v_00_u03b2_1069_, lean_object* v_e_1070_, lean_object* v___y_1071_, lean_object* v___y_1072_, lean_object* v___y_1073_, lean_object* v___y_1074_, lean_object* v___y_1075_){
_start:
{
lean_object* v_res_1076_; 
v_res_1076_ = lp_mathlib_Mathlib_Meta_NormNum_evalEq___lam__1(v_v_1068_, v_00_u03b2_1069_, v_e_1070_, v___y_1071_, v___y_1072_, v___y_1073_, v___y_1074_);
lean_dec(v___y_1074_);
lean_dec_ref(v___y_1073_);
lean_dec(v___y_1072_);
lean_dec_ref(v___y_1071_);
lean_dec_ref(v_00_u03b2_1069_);
lean_dec(v_v_1068_);
return v_res_1076_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Inv(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Eq(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Inv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_NormNum_Eq(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Tactic_NormNum_Inv(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_NormNum_Eq(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_NormNum_Inv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Eq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_NormNum_Eq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_NormNum_Eq(builtin);
}
#ifdef __cplusplus
}
#endif
