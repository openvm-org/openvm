// Lean compiler output
// Module: Mathlib.Tactic.NormNum.DivMod
// Imports: public import Init public meta import Init public import Mathlib.Tactic.NormNum.Ineq public meta import Mathlib.Data.Int.Init
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
lean_object* l_Lean_Meta_whnfR(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_derive(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toInt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_Lean_Expr_natLit_x21(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* lean_int_ediv(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit(lean_object*);
lean_object* lp_mathlib_Int_natMod(lean_object*, lean_object*);
lean_object* l_Lean_mkRawNatLit(lean_object*);
lean_object* lean_int_mul(lean_object*, lean_object*);
lean_object* l_Lean_Level_succ___override(lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isInt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_lit___override(lean_object*);
lean_object* lean_int_neg(lean_object*);
lean_object* lean_int_emod(lean_object*, lean_object*);
uint8_t lean_int_dec_eq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00__private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core_spec__0(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__1;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__2;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Int"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__3_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "refl"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__0_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__7_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__6_value),LEAN_SCALAR_PTR_LITERAL(72, 6, 107, 181, 0, 125, 21, 187)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__7_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__8;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__9;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Bool"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__10_value),LEAN_SCALAR_PTR_LITERAL(250, 44, 198, 216, 184, 195, 199, 178)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__11_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__12;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "true"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__10_value),LEAN_SCALAR_PTR_LITERAL(250, 44, 198, 216, 184, 195, 199, 178)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__14_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__13_value),LEAN_SCALAR_PTR_LITERAL(22, 245, 194, 28, 184, 9, 113, 128)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__14_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__15;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__16;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__17;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__18_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__19_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "NormNum"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__20_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "isInt_ediv"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__21_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__22_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__18_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__22_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__22_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__19_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__22_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__22_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__20_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__22_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__21_value),LEAN_SCALAR_PTR_LITERAL(87, 104, 58, 240, 227, 255, 72, 189)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__22_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__23;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__1___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__1(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HDiv"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hDiv"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(74, 223, 78, 88, 255, 236, 144, 164)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(26, 183, 188, 240, 156, 118, 170, 84)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__8;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__9;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__10;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__11;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instHDiv"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(34, 70, 113, 198, 157, 211, 131, 18)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__13_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__14;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__15;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "instDiv"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__3_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__17_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__16_value),LEAN_SCALAR_PTR_LITERAL(154, 154, 103, 19, 118, 118, 20, 12)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__17_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__18;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__19;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__20;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "instAddMonoidWithOne"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__22_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__18_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__22_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__22_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__19_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__22_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__22_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__20_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__22_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__21_value),LEAN_SCALAR_PTR_LITERAL(129, 65, 157, 144, 0, 78, 170, 16)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__22_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__23;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__24_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__25;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "isInt_ediv_zero"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__27_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__18_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__27_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__27_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__19_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__27_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__27_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__20_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__27_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__26_value),LEAN_SCALAR_PTR_LITERAL(78, 128, 188, 84, 60, 177, 127, 214)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__27_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__28;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Neg"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__29_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "neg"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__30_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__31_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__29_value),LEAN_SCALAR_PTR_LITERAL(94, 4, 109, 108, 64, 81, 153, 133)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__31_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__30_value),LEAN_SCALAR_PTR_LITERAL(105, 26, 70, 221, 245, 238, 127, 238)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__31_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__32_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__32;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__33_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__33;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "instNegInt"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__34_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__35_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__3_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__35_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__34_value),LEAN_SCALAR_PTR_LITERAL(217, 109, 233, 1, 211, 122, 77, 88)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__35_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__36_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__36;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__37_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__37;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "isNat_neg_of_isNegNat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__38_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__39_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__18_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__39_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__39_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__19_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__39_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__39_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__20_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__39_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__38_value),LEAN_SCALAR_PTR_LITERAL(25, 18, 203, 115, 148, 97, 37, 206)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__39 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__39_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__40_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__40;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "isInt_ediv_neg"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__41 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__41_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__42_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__18_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__42_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__42_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__19_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__42_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__42_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__20_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__42_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__41_value),LEAN_SCALAR_PTR_LITERAL(134, 157, 48, 211, 91, 236, 81, 163)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__42 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__42_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__43_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__43;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instRing"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__44 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__44_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__45_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__3_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__45_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__44_value),LEAN_SCALAR_PTR_LITERAL(42, 135, 58, 37, 72, 75, 21, 158)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__45 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__45_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__46_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__46;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "evalIntDiv"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__18_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__19_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___closed__2_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__20_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___closed__1_value),LEAN_SCALAR_PTR_LITERAL(15, 41, 102, 185, 174, 86, 42, 43)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___closed__2_value),LEAN_SCALAR_PTR_LITERAL(1, 1, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___closed__3_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntMod_go_core___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "isInt_emod"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntMod_go_core___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntMod_go_core___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntMod_go_core___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__18_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntMod_go_core___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntMod_go_core___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__19_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntMod_go_core___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntMod_go_core___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__20_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntMod_go_core___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntMod_go_core___closed__1_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntMod_go_core___closed__0_value),LEAN_SCALAR_PTR_LITERAL(23, 225, 187, 125, 76, 196, 155, 251)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntMod_go_core___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntMod_go_core___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntMod_go_core___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntMod_go_core___closed__2;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntMod_go_core(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntMod_go_core___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HMod"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hMod"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__2_value),LEAN_SCALAR_PTR_LITERAL(93, 4, 3, 35, 188, 254, 191, 190)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__3_value),LEAN_SCALAR_PTR_LITERAL(120, 199, 142, 238, 9, 44, 94, 134)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__6;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__7;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__8;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instHMod"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__9_value),LEAN_SCALAR_PTR_LITERAL(242, 7, 29, 140, 31, 32, 204, 87)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__10_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__11;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__12;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "instMod"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__3_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__13_value),LEAN_SCALAR_PTR_LITERAL(155, 18, 147, 153, 76, 63, 153, 183)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__14_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__15;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__16;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__17;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "isInt_emod_zero"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__18_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__19_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__19_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__19_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__19_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__19_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__20_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__19_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__18_value),LEAN_SCALAR_PTR_LITERAL(142, 192, 218, 215, 55, 206, 249, 84)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__19_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__20;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "isInt_emod_neg"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__22_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__18_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__22_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__22_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__19_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__22_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__22_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__20_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__22_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__21_value),LEAN_SCALAR_PTR_LITERAL(44, 169, 134, 208, 83, 150, 33, 41)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__22_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__23;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_NormNum_evalIntMod___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_NormNum_evalIntMod___lam__1___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalIntMod___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "evalIntMod"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntMod___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__18_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntMod___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__19_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntMod___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod___closed__2_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__20_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntMod___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod___closed__1_value),LEAN_SCALAR_PTR_LITERAL(0, 105, 92, 77, 237, 143, 13, 25)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntMod___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod___closed__2_value),LEAN_SCALAR_PTR_LITERAL(1, 1, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod___closed__3_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__0;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "isInt_dvd_false"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__18_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__19_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__2_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__20_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(158, 6, 188, 90, 39, 168, 184, 55)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "isInt_dvd_true"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__18_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__5_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__19_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__5_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__20_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__5_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(176, 14, 241, 164, 85, 103, 45, 159)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__6;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Dvd"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "dvd"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(255, 71, 229, 107, 63, 192, 93, 62)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__9_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(233, 16, 181, 127, 123, 63, 3, 18)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__9_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__10;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__11;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "instDvd"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__3_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__13_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(164, 20, 243, 72, 185, 226, 91, 120)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__13_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__14;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__15;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "evalIntDvd"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__18_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__19_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___closed__2_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__20_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___closed__1_value),LEAN_SCALAR_PTR_LITERAL(172, 88, 200, 246, 200, 75, 227, 144)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___closed__2_value),LEAN_SCALAR_PTR_LITERAL(1, 1, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___closed__3_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00__private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core_spec__0(lean_object* v_a_1_){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_nat_to_int(v_a_1_);
return v___x_2_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__1(void){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; 
v___x_4_ = lean_box(0);
v___x_5_ = l_Lean_Level_succ___override(v___x_4_);
return v___x_5_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__2(void){
_start:
{
lean_object* v___x_6_; lean_object* v___x_7_; lean_object* v___x_8_; 
v___x_6_ = lean_box(0);
v___x_7_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__1, &lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__1);
v___x_8_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_8_, 0, v___x_7_);
lean_ctor_set(v___x_8_, 1, v___x_6_);
return v___x_8_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5(void){
_start:
{
lean_object* v___x_12_; lean_object* v___x_13_; lean_object* v___x_14_; 
v___x_12_ = lean_box(0);
v___x_13_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__4));
v___x_14_ = l_Lean_Expr_const___override(v___x_13_, v___x_12_);
return v___x_14_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__8(void){
_start:
{
lean_object* v___x_19_; lean_object* v___x_20_; lean_object* v___x_21_; 
v___x_19_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__2, &lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__2);
v___x_20_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__7));
v___x_21_ = l_Lean_Expr_const___override(v___x_20_, v___x_19_);
return v___x_21_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__9(void){
_start:
{
lean_object* v___x_22_; lean_object* v___x_23_; lean_object* v___x_24_; 
v___x_22_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5, &lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5);
v___x_23_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__8, &lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__8_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__8);
v___x_24_ = l_Lean_Expr_app___override(v___x_23_, v___x_22_);
return v___x_24_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__12(void){
_start:
{
lean_object* v___x_28_; lean_object* v___x_29_; lean_object* v___x_30_; 
v___x_28_ = lean_box(0);
v___x_29_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__11));
v___x_30_ = l_Lean_Expr_const___override(v___x_29_, v___x_28_);
return v___x_30_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__15(void){
_start:
{
lean_object* v___x_35_; lean_object* v___x_36_; lean_object* v___x_37_; 
v___x_35_ = lean_box(0);
v___x_36_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__14));
v___x_37_ = l_Lean_Expr_const___override(v___x_36_, v___x_35_);
return v___x_37_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__16(void){
_start:
{
lean_object* v___x_38_; lean_object* v___x_39_; lean_object* v___x_40_; 
v___x_38_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__12, &lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__12_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__12);
v___x_39_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__8, &lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__8_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__8);
v___x_40_ = l_Lean_Expr_app___override(v___x_39_, v___x_38_);
return v___x_40_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__17(void){
_start:
{
lean_object* v___x_41_; lean_object* v___x_42_; lean_object* v_pf_u2083_43_; 
v___x_41_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__15, &lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__15_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__15);
v___x_42_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__16, &lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__16_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__16);
v_pf_u2083_43_ = l_Lean_Expr_app___override(v___x_42_, v___x_41_);
return v_pf_u2083_43_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__23(void){
_start:
{
lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; 
v___x_53_ = lean_box(0);
v___x_54_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__22));
v___x_55_ = l_Lean_Expr_const___override(v___x_54_, v___x_53_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core(lean_object* v_a_56_, lean_object* v_na_57_, lean_object* v_za_58_, lean_object* v_pa_59_, lean_object* v_b_60_, lean_object* v_nb_61_, lean_object* v_pb_62_){
_start:
{
lean_object* v_b_63_; lean_object* v___x_64_; lean_object* v_q_65_; lean_object* v_nq_66_; lean_object* v_r_67_; lean_object* v_nr_68_; lean_object* v_m_69_; lean_object* v_nm_70_; lean_object* v___x_71_; lean_object* v_pf_u2081_72_; lean_object* v_pf_u2082_73_; lean_object* v_pf_u2083_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; 
v_b_63_ = lp_batteries_Lean_Expr_natLit_x21(v_nb_61_);
v___x_64_ = lean_nat_to_int(v_b_63_);
v_q_65_ = lean_int_ediv(v_za_58_, v___x_64_);
v_nq_66_ = lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit(v_q_65_);
v_r_67_ = lp_mathlib_Int_natMod(v_za_58_, v___x_64_);
v_nr_68_ = l_Lean_mkRawNatLit(v_r_67_);
v_m_69_ = lean_int_mul(v_q_65_, v___x_64_);
lean_dec(v___x_64_);
v_nm_70_ = lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit(v_m_69_);
lean_dec(v_m_69_);
v___x_71_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__9, &lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__9_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__9);
lean_inc_ref(v_nm_70_);
v_pf_u2081_72_ = l_Lean_Expr_app___override(v___x_71_, v_nm_70_);
lean_inc_ref(v_na_57_);
v_pf_u2082_73_ = l_Lean_Expr_app___override(v___x_71_, v_na_57_);
v_pf_u2083_74_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__17, &lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__17_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__17);
v___x_75_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__23, &lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__23_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__23);
v___x_76_ = l_Lean_Expr_app___override(v___x_75_, v_a_56_);
v___x_77_ = l_Lean_Expr_app___override(v___x_76_, v_b_60_);
lean_inc_ref(v_nq_66_);
v___x_78_ = l_Lean_Expr_app___override(v___x_77_, v_nq_66_);
v___x_79_ = l_Lean_Expr_app___override(v___x_78_, v_nm_70_);
v___x_80_ = l_Lean_Expr_app___override(v___x_79_, v_na_57_);
v___x_81_ = l_Lean_Expr_app___override(v___x_80_, v_nb_61_);
v___x_82_ = l_Lean_Expr_app___override(v___x_81_, v_nr_68_);
v___x_83_ = l_Lean_Expr_app___override(v___x_82_, v_pa_59_);
v___x_84_ = l_Lean_Expr_app___override(v___x_83_, v_pb_62_);
v___x_85_ = l_Lean_Expr_app___override(v___x_84_, v_pf_u2081_72_);
v___x_86_ = l_Lean_Expr_app___override(v___x_85_, v_pf_u2082_73_);
v___x_87_ = l_Lean_Expr_app___override(v___x_86_, v_pf_u2083_74_);
v___x_88_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_88_, 0, v_nq_66_);
lean_ctor_set(v___x_88_, 1, v___x_87_);
v___x_89_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_89_, 0, v_q_65_);
lean_ctor_set(v___x_89_, 1, v___x_88_);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___boxed(lean_object* v_a_90_, lean_object* v_na_91_, lean_object* v_za_92_, lean_object* v_pa_93_, lean_object* v_b_94_, lean_object* v_nb_95_, lean_object* v_pb_96_){
_start:
{
lean_object* v_res_97_; 
v_res_97_ = lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core(v_a_90_, v_na_91_, v_za_92_, v_pa_93_, v_b_94_, v_nb_95_, v_pb_96_);
lean_dec(v_za_92_);
return v_res_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__1___redArg(lean_object* v_k_98_, uint8_t v_allowLevelAssignments_99_, lean_object* v___y_100_, lean_object* v___y_101_, lean_object* v___y_102_, lean_object* v___y_103_){
_start:
{
lean_object* v___x_105_; 
v___x_105_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), v_allowLevelAssignments_99_, v_k_98_, v___y_100_, v___y_101_, v___y_102_, v___y_103_);
if (lean_obj_tag(v___x_105_) == 0)
{
lean_object* v_a_106_; lean_object* v___x_108_; uint8_t v_isShared_109_; uint8_t v_isSharedCheck_113_; 
v_a_106_ = lean_ctor_get(v___x_105_, 0);
v_isSharedCheck_113_ = !lean_is_exclusive(v___x_105_);
if (v_isSharedCheck_113_ == 0)
{
v___x_108_ = v___x_105_;
v_isShared_109_ = v_isSharedCheck_113_;
goto v_resetjp_107_;
}
else
{
lean_inc(v_a_106_);
lean_dec(v___x_105_);
v___x_108_ = lean_box(0);
v_isShared_109_ = v_isSharedCheck_113_;
goto v_resetjp_107_;
}
v_resetjp_107_:
{
lean_object* v___x_111_; 
if (v_isShared_109_ == 0)
{
v___x_111_ = v___x_108_;
goto v_reusejp_110_;
}
else
{
lean_object* v_reuseFailAlloc_112_; 
v_reuseFailAlloc_112_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_112_, 0, v_a_106_);
v___x_111_ = v_reuseFailAlloc_112_;
goto v_reusejp_110_;
}
v_reusejp_110_:
{
return v___x_111_;
}
}
}
else
{
lean_object* v_a_114_; lean_object* v___x_116_; uint8_t v_isShared_117_; uint8_t v_isSharedCheck_121_; 
v_a_114_ = lean_ctor_get(v___x_105_, 0);
v_isSharedCheck_121_ = !lean_is_exclusive(v___x_105_);
if (v_isSharedCheck_121_ == 0)
{
v___x_116_ = v___x_105_;
v_isShared_117_ = v_isSharedCheck_121_;
goto v_resetjp_115_;
}
else
{
lean_inc(v_a_114_);
lean_dec(v___x_105_);
v___x_116_ = lean_box(0);
v_isShared_117_ = v_isSharedCheck_121_;
goto v_resetjp_115_;
}
v_resetjp_115_:
{
lean_object* v___x_119_; 
if (v_isShared_117_ == 0)
{
v___x_119_ = v___x_116_;
goto v_reusejp_118_;
}
else
{
lean_object* v_reuseFailAlloc_120_; 
v_reuseFailAlloc_120_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_120_, 0, v_a_114_);
v___x_119_ = v_reuseFailAlloc_120_;
goto v_reusejp_118_;
}
v_reusejp_118_:
{
return v___x_119_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__1___redArg___boxed(lean_object* v_k_122_, lean_object* v_allowLevelAssignments_123_, lean_object* v___y_124_, lean_object* v___y_125_, lean_object* v___y_126_, lean_object* v___y_127_, lean_object* v___y_128_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_129_; lean_object* v_res_130_; 
v_allowLevelAssignments_boxed_129_ = lean_unbox(v_allowLevelAssignments_123_);
v_res_130_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__1___redArg(v_k_122_, v_allowLevelAssignments_boxed_129_, v___y_124_, v___y_125_, v___y_126_, v___y_127_);
lean_dec(v___y_127_);
lean_dec_ref(v___y_126_);
lean_dec(v___y_125_);
lean_dec_ref(v___y_124_);
return v_res_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__1(lean_object* v_00_u03b1_131_, lean_object* v_k_132_, uint8_t v_allowLevelAssignments_133_, lean_object* v___y_134_, lean_object* v___y_135_, lean_object* v___y_136_, lean_object* v___y_137_){
_start:
{
lean_object* v___x_139_; 
v___x_139_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__1___redArg(v_k_132_, v_allowLevelAssignments_133_, v___y_134_, v___y_135_, v___y_136_, v___y_137_);
return v___x_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__1___boxed(lean_object* v_00_u03b1_140_, lean_object* v_k_141_, lean_object* v_allowLevelAssignments_142_, lean_object* v___y_143_, lean_object* v___y_144_, lean_object* v___y_145_, lean_object* v___y_146_, lean_object* v___y_147_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_148_; lean_object* v_res_149_; 
v_allowLevelAssignments_boxed_148_ = lean_unbox(v_allowLevelAssignments_142_);
v_res_149_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__1(v_00_u03b1_140_, v_k_141_, v_allowLevelAssignments_boxed_148_, v___y_143_, v___y_144_, v___y_145_, v___y_146_);
lean_dec(v___y_146_);
lean_dec_ref(v___y_145_);
lean_dec(v___y_144_);
lean_dec_ref(v___y_143_);
return v_res_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__0(lean_object* v_fn_150_, lean_object* v___x_151_, lean_object* v___y_152_, lean_object* v___y_153_, lean_object* v___y_154_, lean_object* v___y_155_){
_start:
{
lean_object* v___x_157_; 
v___x_157_ = l_Lean_Meta_isExprDefEq(v_fn_150_, v___x_151_, v___y_152_, v___y_153_, v___y_154_, v___y_155_);
return v___x_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__0___boxed(lean_object* v_fn_158_, lean_object* v___x_159_, lean_object* v___y_160_, lean_object* v___y_161_, lean_object* v___y_162_, lean_object* v___y_163_, lean_object* v___y_164_){
_start:
{
lean_object* v_res_165_; 
v_res_165_ = lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__0(v_fn_158_, v___x_159_, v___y_160_, v___y_161_, v___y_162_, v___y_163_);
lean_dec(v___y_163_);
lean_dec_ref(v___y_162_);
lean_dec(v___y_161_);
lean_dec_ref(v___y_160_);
return v_res_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__0_spec__0(lean_object* v_msgData_166_, lean_object* v___y_167_, lean_object* v___y_168_, lean_object* v___y_169_, lean_object* v___y_170_){
_start:
{
lean_object* v___x_172_; lean_object* v_env_173_; lean_object* v___x_174_; lean_object* v_mctx_175_; lean_object* v_lctx_176_; lean_object* v_options_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; 
v___x_172_ = lean_st_ref_get(v___y_170_);
v_env_173_ = lean_ctor_get(v___x_172_, 0);
lean_inc_ref(v_env_173_);
lean_dec(v___x_172_);
v___x_174_ = lean_st_ref_get(v___y_168_);
v_mctx_175_ = lean_ctor_get(v___x_174_, 0);
lean_inc_ref(v_mctx_175_);
lean_dec(v___x_174_);
v_lctx_176_ = lean_ctor_get(v___y_167_, 2);
v_options_177_ = lean_ctor_get(v___y_169_, 2);
lean_inc_ref(v_options_177_);
lean_inc_ref(v_lctx_176_);
v___x_178_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_178_, 0, v_env_173_);
lean_ctor_set(v___x_178_, 1, v_mctx_175_);
lean_ctor_set(v___x_178_, 2, v_lctx_176_);
lean_ctor_set(v___x_178_, 3, v_options_177_);
v___x_179_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_179_, 0, v___x_178_);
lean_ctor_set(v___x_179_, 1, v_msgData_166_);
v___x_180_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_180_, 0, v___x_179_);
return v___x_180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__0_spec__0___boxed(lean_object* v_msgData_181_, lean_object* v___y_182_, lean_object* v___y_183_, lean_object* v___y_184_, lean_object* v___y_185_, lean_object* v___y_186_){
_start:
{
lean_object* v_res_187_; 
v_res_187_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__0_spec__0(v_msgData_181_, v___y_182_, v___y_183_, v___y_184_, v___y_185_);
lean_dec(v___y_185_);
lean_dec_ref(v___y_184_);
lean_dec(v___y_183_);
lean_dec_ref(v___y_182_);
return v_res_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__0___redArg(lean_object* v_msg_188_, lean_object* v___y_189_, lean_object* v___y_190_, lean_object* v___y_191_, lean_object* v___y_192_){
_start:
{
lean_object* v_ref_194_; lean_object* v___x_195_; lean_object* v_a_196_; lean_object* v___x_198_; uint8_t v_isShared_199_; uint8_t v_isSharedCheck_204_; 
v_ref_194_ = lean_ctor_get(v___y_191_, 5);
v___x_195_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__0_spec__0(v_msg_188_, v___y_189_, v___y_190_, v___y_191_, v___y_192_);
v_a_196_ = lean_ctor_get(v___x_195_, 0);
v_isSharedCheck_204_ = !lean_is_exclusive(v___x_195_);
if (v_isSharedCheck_204_ == 0)
{
v___x_198_ = v___x_195_;
v_isShared_199_ = v_isSharedCheck_204_;
goto v_resetjp_197_;
}
else
{
lean_inc(v_a_196_);
lean_dec(v___x_195_);
v___x_198_ = lean_box(0);
v_isShared_199_ = v_isSharedCheck_204_;
goto v_resetjp_197_;
}
v_resetjp_197_:
{
lean_object* v___x_200_; lean_object* v___x_202_; 
lean_inc(v_ref_194_);
v___x_200_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_200_, 0, v_ref_194_);
lean_ctor_set(v___x_200_, 1, v_a_196_);
if (v_isShared_199_ == 0)
{
lean_ctor_set_tag(v___x_198_, 1);
lean_ctor_set(v___x_198_, 0, v___x_200_);
v___x_202_ = v___x_198_;
goto v_reusejp_201_;
}
else
{
lean_object* v_reuseFailAlloc_203_; 
v_reuseFailAlloc_203_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_203_, 0, v___x_200_);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__0___redArg___boxed(lean_object* v_msg_205_, lean_object* v___y_206_, lean_object* v___y_207_, lean_object* v___y_208_, lean_object* v___y_209_, lean_object* v___y_210_){
_start:
{
lean_object* v_res_211_; 
v_res_211_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__0___redArg(v_msg_205_, v___y_206_, v___y_207_, v___y_208_, v___y_209_);
lean_dec(v___y_209_);
lean_dec_ref(v___y_208_);
lean_dec(v___y_207_);
lean_dec_ref(v___y_206_);
return v_res_211_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1(void){
_start:
{
lean_object* v___x_213_; lean_object* v___x_214_; 
v___x_213_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__0));
v___x_214_ = l_Lean_stringToMessageData(v___x_213_);
return v___x_214_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__8(void){
_start:
{
lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; 
v___x_229_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__7));
v___x_230_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__4));
v___x_231_ = l_Lean_Expr_const___override(v___x_230_, v___x_229_);
return v___x_231_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__9(void){
_start:
{
lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; 
v___x_232_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5, &lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5);
v___x_233_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__8, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__8_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__8);
v___x_234_ = l_Lean_Expr_app___override(v___x_233_, v___x_232_);
return v___x_234_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__10(void){
_start:
{
lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; 
v___x_235_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5, &lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5);
v___x_236_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__9, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__9_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__9);
v___x_237_ = l_Lean_Expr_app___override(v___x_236_, v___x_235_);
return v___x_237_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__11(void){
_start:
{
lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; 
v___x_238_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5, &lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5);
v___x_239_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__10, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__10_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__10);
v___x_240_ = l_Lean_Expr_app___override(v___x_239_, v___x_238_);
return v___x_240_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__14(void){
_start:
{
lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; 
v___x_244_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__5));
v___x_245_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__13));
v___x_246_ = l_Lean_Expr_const___override(v___x_245_, v___x_244_);
return v___x_246_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__15(void){
_start:
{
lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; 
v___x_247_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5, &lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5);
v___x_248_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__14, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__14_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__14);
v___x_249_ = l_Lean_Expr_app___override(v___x_248_, v___x_247_);
return v___x_249_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__18(void){
_start:
{
lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; 
v___x_254_ = lean_box(0);
v___x_255_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__17));
v___x_256_ = l_Lean_Expr_const___override(v___x_255_, v___x_254_);
return v___x_256_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__19(void){
_start:
{
lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; 
v___x_257_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__18, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__18_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__18);
v___x_258_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__15, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__15_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__15);
v___x_259_ = l_Lean_Expr_app___override(v___x_258_, v___x_257_);
return v___x_259_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__20(void){
_start:
{
lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; 
v___x_260_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__19, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__19_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__19);
v___x_261_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__11, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__11_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__11);
v___x_262_ = l_Lean_Expr_app___override(v___x_261_, v___x_260_);
return v___x_262_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__23(void){
_start:
{
lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; 
v___x_269_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__5));
v___x_270_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__22));
v___x_271_ = l_Lean_Expr_const___override(v___x_270_, v___x_269_);
return v___x_271_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__25(void){
_start:
{
lean_object* v___x_274_; lean_object* v___x_275_; 
v___x_274_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__24));
v___x_275_ = l_Lean_Expr_lit___override(v___x_274_);
return v___x_275_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__28(void){
_start:
{
lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; 
v___x_282_ = lean_box(0);
v___x_283_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__27));
v___x_284_ = l_Lean_Expr_const___override(v___x_283_, v___x_282_);
return v___x_284_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__32(void){
_start:
{
lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; 
v___x_290_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__5));
v___x_291_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__31));
v___x_292_ = l_Lean_Expr_const___override(v___x_291_, v___x_290_);
return v___x_292_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__33(void){
_start:
{
lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; 
v___x_293_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5, &lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5);
v___x_294_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__32, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__32_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__32);
v___x_295_ = l_Lean_Expr_app___override(v___x_294_, v___x_293_);
return v___x_295_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__36(void){
_start:
{
lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; 
v___x_300_ = lean_box(0);
v___x_301_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__35));
v___x_302_ = l_Lean_Expr_const___override(v___x_301_, v___x_300_);
return v___x_302_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__37(void){
_start:
{
lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; 
v___x_303_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__36, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__36_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__36);
v___x_304_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__33, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__33_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__33);
v___x_305_ = l_Lean_Expr_app___override(v___x_304_, v___x_303_);
return v___x_305_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__40(void){
_start:
{
lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_314_; 
v___x_312_ = lean_box(0);
v___x_313_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__39));
v___x_314_ = l_Lean_Expr_const___override(v___x_313_, v___x_312_);
return v___x_314_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__43(void){
_start:
{
lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_323_; 
v___x_321_ = lean_box(0);
v___x_322_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__42));
v___x_323_ = l_Lean_Expr_const___override(v___x_322_, v___x_321_);
return v___x_323_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__46(void){
_start:
{
lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; 
v___x_328_ = lean_box(0);
v___x_329_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__45));
v___x_330_ = l_Lean_Expr_const___override(v___x_329_, v___x_328_);
return v___x_330_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1(lean_object* v_u_331_, lean_object* v_00_u03b1_332_, lean_object* v_e_333_, lean_object* v___y_334_, lean_object* v___y_335_, lean_object* v___y_336_, lean_object* v___y_337_){
_start:
{
lean_object* v___x_339_; 
lean_inc_ref(v_e_333_);
v___x_339_ = l_Lean_Meta_whnfR(v_e_333_, v___y_334_, v___y_335_, v___y_336_, v___y_337_);
if (lean_obj_tag(v___x_339_) == 0)
{
lean_object* v_a_340_; lean_object* v___y_342_; lean_object* v___y_343_; lean_object* v___y_344_; lean_object* v___y_345_; 
v_a_340_ = lean_ctor_get(v___x_339_, 0);
lean_inc(v_a_340_);
lean_dec_ref_known(v___x_339_, 1);
if (lean_obj_tag(v_a_340_) == 5)
{
lean_object* v_fn_348_; 
v_fn_348_ = lean_ctor_get(v_a_340_, 0);
lean_inc_ref(v_fn_348_);
if (lean_obj_tag(v_fn_348_) == 5)
{
lean_object* v_arg_349_; lean_object* v_fn_350_; lean_object* v_arg_351_; lean_object* v___x_352_; lean_object* v___x_353_; lean_object* v___x_354_; lean_object* v___f_355_; uint8_t v___x_356_; lean_object* v___y_358_; lean_object* v_a_359_; lean_object* v___x_451_; 
v_arg_349_ = lean_ctor_get(v_a_340_, 1);
lean_inc_ref(v_arg_349_);
lean_dec_ref_known(v_a_340_, 2);
v_fn_350_ = lean_ctor_get(v_fn_348_, 0);
lean_inc_ref(v_fn_350_);
v_arg_351_ = lean_ctor_get(v_fn_348_, 1);
lean_inc_ref(v_arg_351_);
lean_dec_ref_known(v_fn_348_, 2);
v___x_352_ = lean_box(0);
v___x_353_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5, &lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5);
v___x_354_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__20, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__20_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__20);
v___f_355_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__0___boxed), 7, 2);
lean_closure_set(v___f_355_, 0, v_fn_350_);
lean_closure_set(v___f_355_, 1, v___x_354_);
v___x_356_ = 0;
v___x_451_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__1___redArg(v___f_355_, v___x_356_, v___y_334_, v___y_335_, v___y_336_, v___y_337_);
if (lean_obj_tag(v___x_451_) == 0)
{
lean_object* v_a_452_; uint8_t v___x_453_; 
v_a_452_ = lean_ctor_get(v___x_451_, 0);
lean_inc(v_a_452_);
lean_dec_ref_known(v___x_451_, 1);
v___x_453_ = lean_unbox(v_a_452_);
lean_dec(v_a_452_);
if (v___x_453_ == 0)
{
lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v_a_456_; lean_object* v___x_458_; uint8_t v_isShared_459_; uint8_t v_isSharedCheck_463_; 
lean_dec_ref(v_arg_351_);
lean_dec_ref(v_arg_349_);
lean_dec_ref(v_e_333_);
lean_dec_ref(v_00_u03b1_332_);
lean_dec(v_u_331_);
v___x_454_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1);
v___x_455_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__0___redArg(v___x_454_, v___y_334_, v___y_335_, v___y_336_, v___y_337_);
v_a_456_ = lean_ctor_get(v___x_455_, 0);
v_isSharedCheck_463_ = !lean_is_exclusive(v___x_455_);
if (v_isSharedCheck_463_ == 0)
{
v___x_458_ = v___x_455_;
v_isShared_459_ = v_isSharedCheck_463_;
goto v_resetjp_457_;
}
else
{
lean_inc(v_a_456_);
lean_dec(v___x_455_);
v___x_458_ = lean_box(0);
v_isShared_459_ = v_isSharedCheck_463_;
goto v_resetjp_457_;
}
v_resetjp_457_:
{
lean_object* v___x_461_; 
if (v_isShared_459_ == 0)
{
v___x_461_ = v___x_458_;
goto v_reusejp_460_;
}
else
{
lean_object* v_reuseFailAlloc_462_; 
v_reuseFailAlloc_462_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_462_, 0, v_a_456_);
v___x_461_ = v_reuseFailAlloc_462_;
goto v_reusejp_460_;
}
v_reusejp_460_:
{
return v___x_461_;
}
}
}
else
{
goto v___jp_435_;
}
}
else
{
lean_object* v_a_464_; lean_object* v___x_466_; uint8_t v_isShared_467_; uint8_t v_isSharedCheck_471_; 
lean_dec_ref(v_arg_351_);
lean_dec_ref(v_arg_349_);
lean_dec_ref(v_e_333_);
lean_dec_ref(v_00_u03b1_332_);
lean_dec(v_u_331_);
v_a_464_ = lean_ctor_get(v___x_451_, 0);
v_isSharedCheck_471_ = !lean_is_exclusive(v___x_451_);
if (v_isSharedCheck_471_ == 0)
{
v___x_466_ = v___x_451_;
v_isShared_467_ = v_isSharedCheck_471_;
goto v_resetjp_465_;
}
else
{
lean_inc(v_a_464_);
lean_dec(v___x_451_);
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
v___jp_357_:
{
lean_object* v_snd_360_; lean_object* v_fst_361_; lean_object* v_fst_362_; lean_object* v_snd_363_; lean_object* v___x_364_; 
v_snd_360_ = lean_ctor_get(v_a_359_, 1);
lean_inc(v_snd_360_);
v_fst_361_ = lean_ctor_get(v_a_359_, 0);
lean_inc(v_fst_361_);
lean_dec_ref(v_a_359_);
v_fst_362_ = lean_ctor_get(v_snd_360_, 0);
lean_inc(v_fst_362_);
v_snd_363_ = lean_ctor_get(v_snd_360_, 1);
lean_inc(v_snd_363_);
lean_dec(v_snd_360_);
lean_inc_ref(v_arg_349_);
v___x_364_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v___x_352_, v___x_353_, v_arg_349_, v___x_356_, v___y_334_, v___y_335_, v___y_336_, v___y_337_);
if (lean_obj_tag(v___x_364_) == 0)
{
lean_object* v_a_365_; lean_object* v___x_367_; uint8_t v_isShared_368_; uint8_t v_isSharedCheck_434_; 
v_a_365_ = lean_ctor_get(v___x_364_, 0);
v_isSharedCheck_434_ = !lean_is_exclusive(v___x_364_);
if (v_isSharedCheck_434_ == 0)
{
v___x_367_ = v___x_364_;
v_isShared_368_ = v_isSharedCheck_434_;
goto v_resetjp_366_;
}
else
{
lean_inc(v_a_365_);
lean_dec(v___x_364_);
v___x_367_ = lean_box(0);
v_isShared_368_ = v_isSharedCheck_434_;
goto v_resetjp_366_;
}
v_resetjp_366_:
{
switch(lean_obj_tag(v_a_365_))
{
case 1:
{
lean_object* v_lit_369_; lean_object* v_proof_370_; lean_object* v___x_372_; uint8_t v_isShared_373_; uint8_t v_isSharedCheck_402_; 
v_lit_369_ = lean_ctor_get(v_a_365_, 1);
v_proof_370_ = lean_ctor_get(v_a_365_, 2);
v_isSharedCheck_402_ = !lean_is_exclusive(v_a_365_);
if (v_isSharedCheck_402_ == 0)
{
lean_object* v_unused_403_; 
v_unused_403_ = lean_ctor_get(v_a_365_, 0);
lean_dec(v_unused_403_);
v___x_372_ = v_a_365_;
v_isShared_373_ = v_isSharedCheck_402_;
goto v_resetjp_371_;
}
else
{
lean_inc(v_proof_370_);
lean_inc(v_lit_369_);
lean_dec(v_a_365_);
v___x_372_ = lean_box(0);
v_isShared_373_ = v_isSharedCheck_402_;
goto v_resetjp_371_;
}
v_resetjp_371_:
{
lean_object* v___x_374_; lean_object* v___x_375_; uint8_t v___x_376_; 
v___x_374_ = lp_batteries_Lean_Expr_natLit_x21(v_lit_369_);
v___x_375_ = lean_unsigned_to_nat(0u);
v___x_376_ = lean_nat_dec_eq(v___x_374_, v___x_375_);
lean_dec(v___x_374_);
if (v___x_376_ == 0)
{
lean_object* v___x_377_; lean_object* v_snd_378_; lean_object* v_fst_379_; lean_object* v_fst_380_; lean_object* v_snd_381_; lean_object* v___x_382_; lean_object* v___x_384_; 
lean_del_object(v___x_372_);
v___x_377_ = lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core(v_arg_351_, v_fst_362_, v_fst_361_, v_snd_363_, v_arg_349_, v_lit_369_, v_proof_370_);
lean_dec(v_fst_361_);
v_snd_378_ = lean_ctor_get(v___x_377_, 1);
lean_inc(v_snd_378_);
v_fst_379_ = lean_ctor_get(v___x_377_, 0);
lean_inc(v_fst_379_);
lean_dec_ref(v___x_377_);
v_fst_380_ = lean_ctor_get(v_snd_378_, 0);
lean_inc(v_fst_380_);
v_snd_381_ = lean_ctor_get(v_snd_378_, 1);
lean_inc(v_snd_381_);
lean_dec(v_snd_378_);
v___x_382_ = lp_mathlib_Mathlib_Meta_NormNum_Result_isInt(v_u_331_, v_00_u03b1_332_, v_e_333_, v___y_358_, v_fst_380_, v_fst_379_, v_snd_381_);
lean_dec(v_fst_379_);
lean_dec(v_fst_380_);
if (v_isShared_368_ == 0)
{
lean_ctor_set(v___x_367_, 0, v___x_382_);
v___x_384_ = v___x_367_;
goto v_reusejp_383_;
}
else
{
lean_object* v_reuseFailAlloc_385_; 
v_reuseFailAlloc_385_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_385_, 0, v___x_382_);
v___x_384_ = v_reuseFailAlloc_385_;
goto v_reusejp_383_;
}
v_reusejp_383_:
{
return v___x_384_;
}
}
else
{
lean_object* v___x_386_; lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___x_397_; 
lean_dec_ref(v_lit_369_);
lean_dec(v_fst_361_);
lean_dec_ref(v_e_333_);
lean_dec(v_u_331_);
v___x_386_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__23, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__23_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__23);
v___x_387_ = l_Lean_Expr_app___override(v___x_386_, v_00_u03b1_332_);
v___x_388_ = l_Lean_Expr_app___override(v___x_387_, v___y_358_);
v___x_389_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__25, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__25_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__25);
v___x_390_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__28, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__28_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__28);
v___x_391_ = l_Lean_Expr_app___override(v___x_390_, v_arg_351_);
v___x_392_ = l_Lean_Expr_app___override(v___x_391_, v_arg_349_);
v___x_393_ = l_Lean_Expr_app___override(v___x_392_, v_fst_362_);
v___x_394_ = l_Lean_Expr_app___override(v___x_393_, v_snd_363_);
v___x_395_ = l_Lean_Expr_app___override(v___x_394_, v_proof_370_);
if (v_isShared_373_ == 0)
{
lean_ctor_set(v___x_372_, 2, v___x_395_);
lean_ctor_set(v___x_372_, 1, v___x_389_);
lean_ctor_set(v___x_372_, 0, v___x_388_);
v___x_397_ = v___x_372_;
goto v_reusejp_396_;
}
else
{
lean_object* v_reuseFailAlloc_401_; 
v_reuseFailAlloc_401_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_401_, 0, v___x_388_);
lean_ctor_set(v_reuseFailAlloc_401_, 1, v___x_389_);
lean_ctor_set(v_reuseFailAlloc_401_, 2, v___x_395_);
v___x_397_ = v_reuseFailAlloc_401_;
goto v_reusejp_396_;
}
v_reusejp_396_:
{
lean_object* v___x_399_; 
if (v_isShared_368_ == 0)
{
lean_ctor_set(v___x_367_, 0, v___x_397_);
v___x_399_ = v___x_367_;
goto v_reusejp_398_;
}
else
{
lean_object* v_reuseFailAlloc_400_; 
v_reuseFailAlloc_400_ = lean_alloc_ctor(0, 1, 0);
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
}
case 2:
{
lean_object* v_lit_404_; lean_object* v_proof_405_; lean_object* v___x_406_; lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v___x_410_; lean_object* v___x_411_; lean_object* v___x_412_; lean_object* v_snd_413_; lean_object* v_fst_414_; lean_object* v_fst_415_; lean_object* v_snd_416_; lean_object* v___x_417_; lean_object* v___x_418_; lean_object* v___x_419_; lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v___x_427_; lean_object* v___x_428_; lean_object* v___x_430_; 
v_lit_404_ = lean_ctor_get(v_a_365_, 1);
lean_inc_ref_n(v_lit_404_, 2);
v_proof_405_ = lean_ctor_get(v_a_365_, 2);
lean_inc_ref(v_proof_405_);
lean_dec_ref_known(v_a_365_, 3);
v___x_406_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__37, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__37_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__37);
lean_inc_ref_n(v_arg_349_, 2);
v___x_407_ = l_Lean_Expr_app___override(v___x_406_, v_arg_349_);
v___x_408_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__40, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__40_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__40);
v___x_409_ = l_Lean_Expr_app___override(v___x_408_, v_arg_349_);
v___x_410_ = l_Lean_Expr_app___override(v___x_409_, v_lit_404_);
v___x_411_ = l_Lean_Expr_app___override(v___x_410_, v_proof_405_);
lean_inc_ref(v_arg_351_);
v___x_412_ = lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core(v_arg_351_, v_fst_362_, v_fst_361_, v_snd_363_, v___x_407_, v_lit_404_, v___x_411_);
lean_dec(v_fst_361_);
v_snd_413_ = lean_ctor_get(v___x_412_, 1);
lean_inc(v_snd_413_);
v_fst_414_ = lean_ctor_get(v___x_412_, 0);
lean_inc(v_fst_414_);
lean_dec_ref(v___x_412_);
v_fst_415_ = lean_ctor_get(v_snd_413_, 0);
lean_inc(v_fst_415_);
v_snd_416_ = lean_ctor_get(v_snd_413_, 1);
lean_inc(v_snd_416_);
lean_dec(v_snd_413_);
v___x_417_ = lean_int_neg(v_fst_414_);
lean_dec(v_fst_414_);
v___x_418_ = lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit(v___x_417_);
v___x_419_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__9, &lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__9_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__9);
lean_inc_ref_n(v___x_418_, 2);
v___x_420_ = l_Lean_Expr_app___override(v___x_419_, v___x_418_);
v___x_421_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__43, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__43_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__43);
v___x_422_ = l_Lean_Expr_app___override(v___x_421_, v_arg_351_);
v___x_423_ = l_Lean_Expr_app___override(v___x_422_, v_arg_349_);
v___x_424_ = l_Lean_Expr_app___override(v___x_423_, v_fst_415_);
v___x_425_ = l_Lean_Expr_app___override(v___x_424_, v___x_418_);
v___x_426_ = l_Lean_Expr_app___override(v___x_425_, v_snd_416_);
v___x_427_ = l_Lean_Expr_app___override(v___x_426_, v___x_420_);
v___x_428_ = lp_mathlib_Mathlib_Meta_NormNum_Result_isInt(v_u_331_, v_00_u03b1_332_, v_e_333_, v___y_358_, v___x_418_, v___x_417_, v___x_427_);
lean_dec(v___x_417_);
lean_dec_ref(v___x_418_);
if (v_isShared_368_ == 0)
{
lean_ctor_set(v___x_367_, 0, v___x_428_);
v___x_430_ = v___x_367_;
goto v_reusejp_429_;
}
else
{
lean_object* v_reuseFailAlloc_431_; 
v_reuseFailAlloc_431_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_431_, 0, v___x_428_);
v___x_430_ = v_reuseFailAlloc_431_;
goto v_reusejp_429_;
}
v_reusejp_429_:
{
return v___x_430_;
}
}
default: 
{
lean_object* v___x_432_; lean_object* v___x_433_; 
lean_del_object(v___x_367_);
lean_dec(v_a_365_);
lean_dec(v_snd_363_);
lean_dec(v_fst_362_);
lean_dec(v_fst_361_);
lean_dec_ref(v___y_358_);
lean_dec_ref(v_arg_351_);
lean_dec_ref(v_arg_349_);
lean_dec_ref(v_e_333_);
lean_dec_ref(v_00_u03b1_332_);
lean_dec(v_u_331_);
v___x_432_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1);
v___x_433_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__0___redArg(v___x_432_, v___y_334_, v___y_335_, v___y_336_, v___y_337_);
return v___x_433_;
}
}
}
}
else
{
lean_dec(v_snd_363_);
lean_dec(v_fst_362_);
lean_dec(v_fst_361_);
lean_dec_ref(v___y_358_);
lean_dec_ref(v_arg_351_);
lean_dec_ref(v_arg_349_);
lean_dec_ref(v_e_333_);
lean_dec_ref(v_00_u03b1_332_);
lean_dec(v_u_331_);
return v___x_364_;
}
}
v___jp_435_:
{
lean_object* v___x_436_; 
lean_inc_ref(v_arg_351_);
v___x_436_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v___x_352_, v___x_353_, v_arg_351_, v___x_356_, v___y_334_, v___y_335_, v___y_336_, v___y_337_);
if (lean_obj_tag(v___x_436_) == 0)
{
lean_object* v_a_437_; lean_object* v___x_438_; lean_object* v___x_439_; 
v_a_437_ = lean_ctor_get(v___x_436_, 0);
lean_inc(v_a_437_);
lean_dec_ref_known(v___x_436_, 1);
v___x_438_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__46, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__46_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__46);
lean_inc_ref(v_arg_351_);
v___x_439_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toInt(v___x_352_, v___x_353_, v_arg_351_, v___x_438_, v_a_437_);
if (lean_obj_tag(v___x_439_) == 0)
{
lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v_a_442_; lean_object* v___x_444_; uint8_t v_isShared_445_; uint8_t v_isSharedCheck_449_; 
lean_dec_ref(v_arg_351_);
lean_dec_ref(v_arg_349_);
lean_dec_ref(v_e_333_);
lean_dec_ref(v_00_u03b1_332_);
lean_dec(v_u_331_);
v___x_440_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1);
v___x_441_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__0___redArg(v___x_440_, v___y_334_, v___y_335_, v___y_336_, v___y_337_);
v_a_442_ = lean_ctor_get(v___x_441_, 0);
v_isSharedCheck_449_ = !lean_is_exclusive(v___x_441_);
if (v_isSharedCheck_449_ == 0)
{
v___x_444_ = v___x_441_;
v_isShared_445_ = v_isSharedCheck_449_;
goto v_resetjp_443_;
}
else
{
lean_inc(v_a_442_);
lean_dec(v___x_441_);
v___x_444_ = lean_box(0);
v_isShared_445_ = v_isSharedCheck_449_;
goto v_resetjp_443_;
}
v_resetjp_443_:
{
lean_object* v___x_447_; 
if (v_isShared_445_ == 0)
{
v___x_447_ = v___x_444_;
goto v_reusejp_446_;
}
else
{
lean_object* v_reuseFailAlloc_448_; 
v_reuseFailAlloc_448_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_448_, 0, v_a_442_);
v___x_447_ = v_reuseFailAlloc_448_;
goto v_reusejp_446_;
}
v_reusejp_446_:
{
return v___x_447_;
}
}
}
else
{
lean_object* v_val_450_; 
v_val_450_ = lean_ctor_get(v___x_439_, 0);
lean_inc(v_val_450_);
lean_dec_ref_known(v___x_439_, 1);
v___y_358_ = v___x_438_;
v_a_359_ = v_val_450_;
goto v___jp_357_;
}
}
else
{
lean_dec_ref(v_arg_351_);
lean_dec_ref(v_arg_349_);
lean_dec_ref(v_e_333_);
lean_dec_ref(v_00_u03b1_332_);
lean_dec(v_u_331_);
return v___x_436_;
}
}
}
else
{
lean_dec_ref_known(v_a_340_, 2);
lean_dec_ref(v_fn_348_);
lean_dec_ref(v_e_333_);
lean_dec_ref(v_00_u03b1_332_);
lean_dec(v_u_331_);
v___y_342_ = v___y_334_;
v___y_343_ = v___y_335_;
v___y_344_ = v___y_336_;
v___y_345_ = v___y_337_;
goto v___jp_341_;
}
}
else
{
lean_dec(v_a_340_);
lean_dec_ref(v_e_333_);
lean_dec_ref(v_00_u03b1_332_);
lean_dec(v_u_331_);
v___y_342_ = v___y_334_;
v___y_343_ = v___y_335_;
v___y_344_ = v___y_336_;
v___y_345_ = v___y_337_;
goto v___jp_341_;
}
v___jp_341_:
{
lean_object* v___x_346_; lean_object* v___x_347_; 
v___x_346_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1);
v___x_347_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__0___redArg(v___x_346_, v___y_342_, v___y_343_, v___y_344_, v___y_345_);
return v___x_347_;
}
}
else
{
lean_object* v_a_472_; lean_object* v___x_474_; uint8_t v_isShared_475_; uint8_t v_isSharedCheck_479_; 
lean_dec_ref(v_e_333_);
lean_dec_ref(v_00_u03b1_332_);
lean_dec(v_u_331_);
v_a_472_ = lean_ctor_get(v___x_339_, 0);
v_isSharedCheck_479_ = !lean_is_exclusive(v___x_339_);
if (v_isSharedCheck_479_ == 0)
{
v___x_474_ = v___x_339_;
v_isShared_475_ = v_isSharedCheck_479_;
goto v_resetjp_473_;
}
else
{
lean_inc(v_a_472_);
lean_dec(v___x_339_);
v___x_474_ = lean_box(0);
v_isShared_475_ = v_isSharedCheck_479_;
goto v_resetjp_473_;
}
v_resetjp_473_:
{
lean_object* v___x_477_; 
if (v_isShared_475_ == 0)
{
v___x_477_ = v___x_474_;
goto v_reusejp_476_;
}
else
{
lean_object* v_reuseFailAlloc_478_; 
v_reuseFailAlloc_478_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_478_, 0, v_a_472_);
v___x_477_ = v_reuseFailAlloc_478_;
goto v_reusejp_476_;
}
v_reusejp_476_:
{
return v___x_477_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___boxed(lean_object* v_u_480_, lean_object* v_00_u03b1_481_, lean_object* v_e_482_, lean_object* v___y_483_, lean_object* v___y_484_, lean_object* v___y_485_, lean_object* v___y_486_, lean_object* v___y_487_){
_start:
{
lean_object* v_res_488_; 
v_res_488_ = lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1(v_u_480_, v_00_u03b1_481_, v_e_482_, v___y_483_, v___y_484_, v___y_485_, v___y_486_);
lean_dec(v___y_486_);
lean_dec_ref(v___y_485_);
lean_dec(v___y_484_);
lean_dec_ref(v___y_483_);
return v_res_488_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__0(lean_object* v_00_u03b1_501_, lean_object* v_msg_502_, lean_object* v___y_503_, lean_object* v___y_504_, lean_object* v___y_505_, lean_object* v___y_506_){
_start:
{
lean_object* v___x_508_; 
v___x_508_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__0___redArg(v_msg_502_, v___y_503_, v___y_504_, v___y_505_, v___y_506_);
return v___x_508_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__0___boxed(lean_object* v_00_u03b1_509_, lean_object* v_msg_510_, lean_object* v___y_511_, lean_object* v___y_512_, lean_object* v___y_513_, lean_object* v___y_514_, lean_object* v___y_515_){
_start:
{
lean_object* v_res_516_; 
v_res_516_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__0(v_00_u03b1_509_, v_msg_510_, v___y_511_, v___y_512_, v___y_513_, v___y_514_);
lean_dec(v___y_514_);
lean_dec_ref(v___y_513_);
lean_dec(v___y_512_);
lean_dec_ref(v___y_511_);
return v_res_516_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntMod_go_core___closed__2(void){
_start:
{
lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; 
v___x_523_ = lean_box(0);
v___x_524_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntMod_go_core___closed__1));
v___x_525_ = l_Lean_Expr_const___override(v___x_524_, v___x_523_);
return v___x_525_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntMod_go_core(lean_object* v_a_526_, lean_object* v_na_527_, lean_object* v_za_528_, lean_object* v_pa_529_, lean_object* v_b_530_, lean_object* v_nb_531_, lean_object* v_pb_532_){
_start:
{
lean_object* v_b_533_; lean_object* v___x_534_; lean_object* v_q_535_; lean_object* v_nq_536_; lean_object* v_r_537_; lean_object* v_nr_538_; lean_object* v_m_539_; lean_object* v_nm_540_; lean_object* v___x_541_; lean_object* v_pf_u2081_542_; lean_object* v_pf_u2082_543_; lean_object* v_pf_u2083_544_; lean_object* v___x_545_; lean_object* v___x_546_; lean_object* v___x_547_; lean_object* v___x_548_; lean_object* v___x_549_; lean_object* v___x_550_; lean_object* v___x_551_; lean_object* v___x_552_; lean_object* v___x_553_; lean_object* v___x_554_; lean_object* v___x_555_; lean_object* v___x_556_; lean_object* v___x_557_; lean_object* v___x_558_; 
v_b_533_ = lp_batteries_Lean_Expr_natLit_x21(v_nb_531_);
v___x_534_ = lean_nat_to_int(v_b_533_);
v_q_535_ = lean_int_ediv(v_za_528_, v___x_534_);
v_nq_536_ = lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit(v_q_535_);
v_r_537_ = lp_mathlib_Int_natMod(v_za_528_, v___x_534_);
v_nr_538_ = l_Lean_mkRawNatLit(v_r_537_);
v_m_539_ = lean_int_mul(v_q_535_, v___x_534_);
lean_dec(v___x_534_);
lean_dec(v_q_535_);
v_nm_540_ = lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit(v_m_539_);
lean_dec(v_m_539_);
v___x_541_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__9, &lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__9_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__9);
lean_inc_ref(v_nm_540_);
v_pf_u2081_542_ = l_Lean_Expr_app___override(v___x_541_, v_nm_540_);
lean_inc_ref(v_na_527_);
v_pf_u2082_543_ = l_Lean_Expr_app___override(v___x_541_, v_na_527_);
v_pf_u2083_544_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__17, &lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__17_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__17);
v___x_545_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntMod_go_core___closed__2, &lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntMod_go_core___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntMod_go_core___closed__2);
v___x_546_ = l_Lean_Expr_app___override(v___x_545_, v_a_526_);
v___x_547_ = l_Lean_Expr_app___override(v___x_546_, v_b_530_);
v___x_548_ = l_Lean_Expr_app___override(v___x_547_, v_nq_536_);
v___x_549_ = l_Lean_Expr_app___override(v___x_548_, v_nm_540_);
v___x_550_ = l_Lean_Expr_app___override(v___x_549_, v_na_527_);
v___x_551_ = l_Lean_Expr_app___override(v___x_550_, v_nb_531_);
lean_inc_ref(v_nr_538_);
v___x_552_ = l_Lean_Expr_app___override(v___x_551_, v_nr_538_);
v___x_553_ = l_Lean_Expr_app___override(v___x_552_, v_pa_529_);
v___x_554_ = l_Lean_Expr_app___override(v___x_553_, v_pb_532_);
v___x_555_ = l_Lean_Expr_app___override(v___x_554_, v_pf_u2081_542_);
v___x_556_ = l_Lean_Expr_app___override(v___x_555_, v_pf_u2082_543_);
v___x_557_ = l_Lean_Expr_app___override(v___x_556_, v_pf_u2083_544_);
v___x_558_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_558_, 0, v_nr_538_);
lean_ctor_set(v___x_558_, 1, v___x_557_);
return v___x_558_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntMod_go_core___boxed(lean_object* v_a_559_, lean_object* v_na_560_, lean_object* v_za_561_, lean_object* v_pa_562_, lean_object* v_b_563_, lean_object* v_nb_564_, lean_object* v_pb_565_){
_start:
{
lean_object* v_res_566_; 
v_res_566_ = lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntMod_go_core(v_a_559_, v_na_560_, v_za_561_, v_pa_562_, v_b_563_, v_nb_564_, v_pb_565_);
lean_dec(v_za_561_);
return v_res_566_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__0(void){
_start:
{
lean_object* v___x_567_; lean_object* v___x_568_; lean_object* v___x_569_; 
v___x_567_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5, &lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5);
v___x_568_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__23, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__23_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__23);
v___x_569_ = l_Lean_Expr_app___override(v___x_568_, v___x_567_);
return v___x_569_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__1(void){
_start:
{
lean_object* v___x_570_; lean_object* v___x_571_; lean_object* v___x_572_; 
v___x_570_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__46, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__46_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__46);
v___x_571_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__0, &lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__0_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__0);
v___x_572_ = l_Lean_Expr_app___override(v___x_571_, v___x_570_);
return v___x_572_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__5(void){
_start:
{
lean_object* v___x_578_; lean_object* v___x_579_; lean_object* v___x_580_; 
v___x_578_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__7));
v___x_579_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__4));
v___x_580_ = l_Lean_Expr_const___override(v___x_579_, v___x_578_);
return v___x_580_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__6(void){
_start:
{
lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_583_; 
v___x_581_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5, &lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5);
v___x_582_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__5, &lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__5_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__5);
v___x_583_ = l_Lean_Expr_app___override(v___x_582_, v___x_581_);
return v___x_583_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__7(void){
_start:
{
lean_object* v___x_584_; lean_object* v___x_585_; lean_object* v___x_586_; 
v___x_584_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5, &lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5);
v___x_585_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__6, &lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__6_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__6);
v___x_586_ = l_Lean_Expr_app___override(v___x_585_, v___x_584_);
return v___x_586_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__8(void){
_start:
{
lean_object* v___x_587_; lean_object* v___x_588_; lean_object* v___x_589_; 
v___x_587_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5, &lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5);
v___x_588_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__7, &lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__7_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__7);
v___x_589_ = l_Lean_Expr_app___override(v___x_588_, v___x_587_);
return v___x_589_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__11(void){
_start:
{
lean_object* v___x_593_; lean_object* v___x_594_; lean_object* v___x_595_; 
v___x_593_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__5));
v___x_594_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__10));
v___x_595_ = l_Lean_Expr_const___override(v___x_594_, v___x_593_);
return v___x_595_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__12(void){
_start:
{
lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v___x_598_; 
v___x_596_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5, &lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5);
v___x_597_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__11, &lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__11_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__11);
v___x_598_ = l_Lean_Expr_app___override(v___x_597_, v___x_596_);
return v___x_598_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__15(void){
_start:
{
lean_object* v___x_603_; lean_object* v___x_604_; lean_object* v___x_605_; 
v___x_603_ = lean_box(0);
v___x_604_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__14));
v___x_605_ = l_Lean_Expr_const___override(v___x_604_, v___x_603_);
return v___x_605_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__16(void){
_start:
{
lean_object* v___x_606_; lean_object* v___x_607_; lean_object* v___x_608_; 
v___x_606_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__15, &lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__15_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__15);
v___x_607_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__12, &lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__12_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__12);
v___x_608_ = l_Lean_Expr_app___override(v___x_607_, v___x_606_);
return v___x_608_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__17(void){
_start:
{
lean_object* v___x_609_; lean_object* v___x_610_; lean_object* v___x_611_; 
v___x_609_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__16, &lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__16_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__16);
v___x_610_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__8, &lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__8_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__8);
v___x_611_ = l_Lean_Expr_app___override(v___x_610_, v___x_609_);
return v___x_611_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__20(void){
_start:
{
lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_620_; 
v___x_618_ = lean_box(0);
v___x_619_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__19));
v___x_620_ = l_Lean_Expr_const___override(v___x_619_, v___x_618_);
return v___x_620_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__23(void){
_start:
{
lean_object* v___x_627_; lean_object* v___x_628_; lean_object* v___x_629_; 
v___x_627_ = lean_box(0);
v___x_628_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__22));
v___x_629_ = l_Lean_Expr_const___override(v___x_628_, v___x_627_);
return v___x_629_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go(lean_object* v_a_630_, lean_object* v_na_631_, lean_object* v_za_632_, lean_object* v_pa_633_, lean_object* v_b_634_, lean_object* v_x_635_){
_start:
{
switch(lean_obj_tag(v_x_635_))
{
case 1:
{
lean_object* v_lit_636_; lean_object* v_proof_637_; lean_object* v___x_639_; uint8_t v_isShared_640_; uint8_t v_isSharedCheck_666_; 
v_lit_636_ = lean_ctor_get(v_x_635_, 1);
v_proof_637_ = lean_ctor_get(v_x_635_, 2);
v_isSharedCheck_666_ = !lean_is_exclusive(v_x_635_);
if (v_isSharedCheck_666_ == 0)
{
lean_object* v_unused_667_; 
v_unused_667_ = lean_ctor_get(v_x_635_, 0);
lean_dec(v_unused_667_);
v___x_639_ = v_x_635_;
v_isShared_640_ = v_isSharedCheck_666_;
goto v_resetjp_638_;
}
else
{
lean_inc(v_proof_637_);
lean_inc(v_lit_636_);
lean_dec(v_x_635_);
v___x_639_ = lean_box(0);
v_isShared_640_ = v_isSharedCheck_666_;
goto v_resetjp_638_;
}
v_resetjp_638_:
{
lean_object* v___x_641_; lean_object* v___x_642_; uint8_t v___x_643_; 
v___x_641_ = lp_batteries_Lean_Expr_natLit_x21(v_lit_636_);
v___x_642_ = lean_unsigned_to_nat(0u);
v___x_643_ = lean_nat_dec_eq(v___x_641_, v___x_642_);
lean_dec(v___x_641_);
if (v___x_643_ == 0)
{
lean_object* v___x_644_; lean_object* v_fst_645_; lean_object* v_snd_646_; lean_object* v___x_647_; lean_object* v___x_649_; 
v___x_644_ = lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntMod_go_core(v_a_630_, v_na_631_, v_za_632_, v_pa_633_, v_b_634_, v_lit_636_, v_proof_637_);
v_fst_645_ = lean_ctor_get(v___x_644_, 0);
lean_inc(v_fst_645_);
v_snd_646_ = lean_ctor_get(v___x_644_, 1);
lean_inc(v_snd_646_);
lean_dec_ref(v___x_644_);
v___x_647_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__1);
if (v_isShared_640_ == 0)
{
lean_ctor_set(v___x_639_, 2, v_snd_646_);
lean_ctor_set(v___x_639_, 1, v_fst_645_);
lean_ctor_set(v___x_639_, 0, v___x_647_);
v___x_649_ = v___x_639_;
goto v_reusejp_648_;
}
else
{
lean_object* v_reuseFailAlloc_651_; 
v_reuseFailAlloc_651_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_651_, 0, v___x_647_);
lean_ctor_set(v_reuseFailAlloc_651_, 1, v_fst_645_);
lean_ctor_set(v_reuseFailAlloc_651_, 2, v_snd_646_);
v___x_649_ = v_reuseFailAlloc_651_;
goto v_reusejp_648_;
}
v_reusejp_648_:
{
lean_object* v___x_650_; 
v___x_650_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_650_, 0, v___x_649_);
return v___x_650_;
}
}
else
{
lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v___x_654_; lean_object* v___x_655_; lean_object* v___x_656_; lean_object* v___x_657_; lean_object* v___x_658_; lean_object* v___x_659_; lean_object* v___x_660_; lean_object* v___x_661_; lean_object* v___x_662_; lean_object* v___x_663_; lean_object* v___x_664_; lean_object* v___x_665_; 
lean_del_object(v___x_639_);
lean_dec_ref(v_lit_636_);
v___x_652_ = lean_box(0);
v___x_653_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5, &lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5);
v___x_654_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__17, &lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__17_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__17);
lean_inc_ref(v_a_630_);
v___x_655_ = l_Lean_Expr_app___override(v___x_654_, v_a_630_);
lean_inc_ref(v_b_634_);
v___x_656_ = l_Lean_Expr_app___override(v___x_655_, v_b_634_);
v___x_657_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__46, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__46_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__46);
v___x_658_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__20, &lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__20_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__20);
v___x_659_ = l_Lean_Expr_app___override(v___x_658_, v_a_630_);
v___x_660_ = l_Lean_Expr_app___override(v___x_659_, v_b_634_);
lean_inc_ref(v_na_631_);
v___x_661_ = l_Lean_Expr_app___override(v___x_660_, v_na_631_);
v___x_662_ = l_Lean_Expr_app___override(v___x_661_, v_pa_633_);
v___x_663_ = l_Lean_Expr_app___override(v___x_662_, v_proof_637_);
v___x_664_ = lp_mathlib_Mathlib_Meta_NormNum_Result_isInt(v___x_652_, v___x_653_, v___x_656_, v___x_657_, v_na_631_, v_za_632_, v___x_663_);
lean_dec_ref(v_na_631_);
v___x_665_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_665_, 0, v___x_664_);
return v___x_665_;
}
}
}
case 2:
{
lean_object* v_inst_668_; lean_object* v_lit_669_; lean_object* v_proof_670_; lean_object* v___x_672_; uint8_t v_isShared_673_; uint8_t v_isSharedCheck_694_; 
v_inst_668_ = lean_ctor_get(v_x_635_, 0);
v_lit_669_ = lean_ctor_get(v_x_635_, 1);
v_proof_670_ = lean_ctor_get(v_x_635_, 2);
v_isSharedCheck_694_ = !lean_is_exclusive(v_x_635_);
if (v_isSharedCheck_694_ == 0)
{
v___x_672_ = v_x_635_;
v_isShared_673_ = v_isSharedCheck_694_;
goto v_resetjp_671_;
}
else
{
lean_inc(v_proof_670_);
lean_inc(v_lit_669_);
lean_inc(v_inst_668_);
lean_dec(v_x_635_);
v___x_672_ = lean_box(0);
v_isShared_673_ = v_isSharedCheck_694_;
goto v_resetjp_671_;
}
v_resetjp_671_:
{
lean_object* v___x_674_; lean_object* v___x_675_; lean_object* v___x_676_; lean_object* v___x_677_; lean_object* v___x_678_; lean_object* v___x_679_; lean_object* v___x_680_; lean_object* v_fst_681_; lean_object* v_snd_682_; lean_object* v___x_683_; lean_object* v___x_684_; lean_object* v___x_685_; lean_object* v___x_686_; lean_object* v___x_687_; lean_object* v___x_688_; lean_object* v___x_689_; lean_object* v___x_691_; 
v___x_674_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__37, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__37_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__37);
lean_inc_ref_n(v_b_634_, 2);
v___x_675_ = l_Lean_Expr_app___override(v___x_674_, v_b_634_);
v___x_676_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__40, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__40_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__40);
v___x_677_ = l_Lean_Expr_app___override(v___x_676_, v_b_634_);
lean_inc_ref(v_lit_669_);
v___x_678_ = l_Lean_Expr_app___override(v___x_677_, v_lit_669_);
v___x_679_ = l_Lean_Expr_app___override(v___x_678_, v_proof_670_);
lean_inc_ref(v_a_630_);
v___x_680_ = lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntMod_go_core(v_a_630_, v_na_631_, v_za_632_, v_pa_633_, v___x_675_, v_lit_669_, v___x_679_);
v_fst_681_ = lean_ctor_get(v___x_680_, 0);
lean_inc_n(v_fst_681_, 2);
v_snd_682_ = lean_ctor_get(v___x_680_, 1);
lean_inc(v_snd_682_);
lean_dec_ref(v___x_680_);
v___x_683_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__0, &lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__0_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__0);
v___x_684_ = l_Lean_Expr_app___override(v___x_683_, v_inst_668_);
v___x_685_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__23, &lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__23_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__23);
v___x_686_ = l_Lean_Expr_app___override(v___x_685_, v_a_630_);
v___x_687_ = l_Lean_Expr_app___override(v___x_686_, v_b_634_);
v___x_688_ = l_Lean_Expr_app___override(v___x_687_, v_fst_681_);
v___x_689_ = l_Lean_Expr_app___override(v___x_688_, v_snd_682_);
if (v_isShared_673_ == 0)
{
lean_ctor_set_tag(v___x_672_, 1);
lean_ctor_set(v___x_672_, 2, v___x_689_);
lean_ctor_set(v___x_672_, 1, v_fst_681_);
lean_ctor_set(v___x_672_, 0, v___x_684_);
v___x_691_ = v___x_672_;
goto v_reusejp_690_;
}
else
{
lean_object* v_reuseFailAlloc_693_; 
v_reuseFailAlloc_693_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_693_, 0, v___x_684_);
lean_ctor_set(v_reuseFailAlloc_693_, 1, v_fst_681_);
lean_ctor_set(v_reuseFailAlloc_693_, 2, v___x_689_);
v___x_691_ = v_reuseFailAlloc_693_;
goto v_reusejp_690_;
}
v_reusejp_690_:
{
lean_object* v___x_692_; 
v___x_692_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_692_, 0, v___x_691_);
return v___x_692_;
}
}
}
default: 
{
lean_object* v___x_695_; 
lean_dec_ref(v_x_635_);
lean_dec_ref(v_b_634_);
lean_dec_ref(v_pa_633_);
lean_dec_ref(v_na_631_);
lean_dec_ref(v_a_630_);
v___x_695_ = lean_box(0);
return v___x_695_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___boxed(lean_object* v_a_696_, lean_object* v_na_697_, lean_object* v_za_698_, lean_object* v_pa_699_, lean_object* v_b_700_, lean_object* v_x_701_){
_start:
{
lean_object* v_res_702_; 
v_res_702_ = lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go(v_a_696_, v_na_697_, v_za_698_, v_pa_699_, v_b_700_, v_x_701_);
lean_dec(v_za_698_);
return v_res_702_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod___lam__1(lean_object* v_u_703_, lean_object* v_00_u03b1_704_, lean_object* v_e_705_, lean_object* v___y_706_, lean_object* v___y_707_, lean_object* v___y_708_, lean_object* v___y_709_){
_start:
{
lean_object* v___x_711_; 
v___x_711_ = l_Lean_Meta_whnfR(v_e_705_, v___y_706_, v___y_707_, v___y_708_, v___y_709_);
if (lean_obj_tag(v___x_711_) == 0)
{
lean_object* v_a_712_; lean_object* v___y_714_; lean_object* v___y_715_; lean_object* v___y_716_; lean_object* v___y_717_; 
v_a_712_ = lean_ctor_get(v___x_711_, 0);
lean_inc(v_a_712_);
lean_dec_ref_known(v___x_711_, 1);
if (lean_obj_tag(v_a_712_) == 5)
{
lean_object* v_fn_720_; 
v_fn_720_ = lean_ctor_get(v_a_712_, 0);
lean_inc_ref(v_fn_720_);
if (lean_obj_tag(v_fn_720_) == 5)
{
lean_object* v_arg_721_; lean_object* v_fn_722_; lean_object* v_arg_723_; lean_object* v___x_724_; lean_object* v___x_725_; lean_object* v___x_726_; lean_object* v___f_727_; uint8_t v___x_728_; lean_object* v___x_754_; 
v_arg_721_ = lean_ctor_get(v_a_712_, 1);
lean_inc_ref(v_arg_721_);
lean_dec_ref_known(v_a_712_, 2);
v_fn_722_ = lean_ctor_get(v_fn_720_, 0);
lean_inc_ref(v_fn_722_);
v_arg_723_ = lean_ctor_get(v_fn_720_, 1);
lean_inc_ref(v_arg_723_);
lean_dec_ref_known(v_fn_720_, 2);
v___x_724_ = lean_box(0);
v___x_725_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5, &lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5);
v___x_726_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__17, &lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__17_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go___closed__17);
v___f_727_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__0___boxed), 7, 2);
lean_closure_set(v___f_727_, 0, v_fn_722_);
lean_closure_set(v___f_727_, 1, v___x_726_);
v___x_728_ = 0;
v___x_754_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__1___redArg(v___f_727_, v___x_728_, v___y_706_, v___y_707_, v___y_708_, v___y_709_);
if (lean_obj_tag(v___x_754_) == 0)
{
lean_object* v_a_755_; uint8_t v___x_756_; 
v_a_755_ = lean_ctor_get(v___x_754_, 0);
lean_inc(v_a_755_);
lean_dec_ref_known(v___x_754_, 1);
v___x_756_ = lean_unbox(v_a_755_);
lean_dec(v_a_755_);
if (v___x_756_ == 0)
{
lean_object* v___x_757_; lean_object* v___x_758_; lean_object* v_a_759_; lean_object* v___x_761_; uint8_t v_isShared_762_; uint8_t v_isSharedCheck_766_; 
lean_dec_ref(v_arg_723_);
lean_dec_ref(v_arg_721_);
v___x_757_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1);
v___x_758_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__0___redArg(v___x_757_, v___y_706_, v___y_707_, v___y_708_, v___y_709_);
v_a_759_ = lean_ctor_get(v___x_758_, 0);
v_isSharedCheck_766_ = !lean_is_exclusive(v___x_758_);
if (v_isSharedCheck_766_ == 0)
{
v___x_761_ = v___x_758_;
v_isShared_762_ = v_isSharedCheck_766_;
goto v_resetjp_760_;
}
else
{
lean_inc(v_a_759_);
lean_dec(v___x_758_);
v___x_761_ = lean_box(0);
v_isShared_762_ = v_isSharedCheck_766_;
goto v_resetjp_760_;
}
v_resetjp_760_:
{
lean_object* v___x_764_; 
if (v_isShared_762_ == 0)
{
v___x_764_ = v___x_761_;
goto v_reusejp_763_;
}
else
{
lean_object* v_reuseFailAlloc_765_; 
v_reuseFailAlloc_765_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_765_, 0, v_a_759_);
v___x_764_ = v_reuseFailAlloc_765_;
goto v_reusejp_763_;
}
v_reusejp_763_:
{
return v___x_764_;
}
}
}
else
{
goto v___jp_729_;
}
}
else
{
lean_object* v_a_767_; lean_object* v___x_769_; uint8_t v_isShared_770_; uint8_t v_isSharedCheck_774_; 
lean_dec_ref(v_arg_723_);
lean_dec_ref(v_arg_721_);
v_a_767_ = lean_ctor_get(v___x_754_, 0);
v_isSharedCheck_774_ = !lean_is_exclusive(v___x_754_);
if (v_isSharedCheck_774_ == 0)
{
v___x_769_ = v___x_754_;
v_isShared_770_ = v_isSharedCheck_774_;
goto v_resetjp_768_;
}
else
{
lean_inc(v_a_767_);
lean_dec(v___x_754_);
v___x_769_ = lean_box(0);
v_isShared_770_ = v_isSharedCheck_774_;
goto v_resetjp_768_;
}
v_resetjp_768_:
{
lean_object* v___x_772_; 
if (v_isShared_770_ == 0)
{
v___x_772_ = v___x_769_;
goto v_reusejp_771_;
}
else
{
lean_object* v_reuseFailAlloc_773_; 
v_reuseFailAlloc_773_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_773_, 0, v_a_767_);
v___x_772_ = v_reuseFailAlloc_773_;
goto v_reusejp_771_;
}
v_reusejp_771_:
{
return v___x_772_;
}
}
}
v___jp_729_:
{
lean_object* v___x_730_; 
lean_inc_ref(v_arg_723_);
v___x_730_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v___x_724_, v___x_725_, v_arg_723_, v___x_728_, v___y_706_, v___y_707_, v___y_708_, v___y_709_);
if (lean_obj_tag(v___x_730_) == 0)
{
lean_object* v_a_731_; lean_object* v___x_732_; lean_object* v___x_733_; 
v_a_731_ = lean_ctor_get(v___x_730_, 0);
lean_inc(v_a_731_);
lean_dec_ref_known(v___x_730_, 1);
v___x_732_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__46, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__46_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__46);
lean_inc_ref(v_arg_723_);
v___x_733_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toInt(v___x_724_, v___x_725_, v_arg_723_, v___x_732_, v_a_731_);
if (lean_obj_tag(v___x_733_) == 1)
{
lean_object* v_val_734_; lean_object* v_snd_735_; lean_object* v_fst_736_; lean_object* v_fst_737_; lean_object* v_snd_738_; lean_object* v___x_739_; 
v_val_734_ = lean_ctor_get(v___x_733_, 0);
lean_inc(v_val_734_);
lean_dec_ref_known(v___x_733_, 1);
v_snd_735_ = lean_ctor_get(v_val_734_, 1);
lean_inc(v_snd_735_);
v_fst_736_ = lean_ctor_get(v_val_734_, 0);
lean_inc(v_fst_736_);
lean_dec(v_val_734_);
v_fst_737_ = lean_ctor_get(v_snd_735_, 0);
lean_inc(v_fst_737_);
v_snd_738_ = lean_ctor_get(v_snd_735_, 1);
lean_inc(v_snd_738_);
lean_dec(v_snd_735_);
lean_inc_ref(v_arg_721_);
v___x_739_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v___x_724_, v___x_725_, v_arg_721_, v___x_728_, v___y_706_, v___y_707_, v___y_708_, v___y_709_);
if (lean_obj_tag(v___x_739_) == 0)
{
lean_object* v_a_740_; lean_object* v___x_742_; uint8_t v_isShared_743_; uint8_t v_isSharedCheck_751_; 
v_a_740_ = lean_ctor_get(v___x_739_, 0);
v_isSharedCheck_751_ = !lean_is_exclusive(v___x_739_);
if (v_isSharedCheck_751_ == 0)
{
v___x_742_ = v___x_739_;
v_isShared_743_ = v_isSharedCheck_751_;
goto v_resetjp_741_;
}
else
{
lean_inc(v_a_740_);
lean_dec(v___x_739_);
v___x_742_ = lean_box(0);
v_isShared_743_ = v_isSharedCheck_751_;
goto v_resetjp_741_;
}
v_resetjp_741_:
{
lean_object* v___x_744_; 
v___x_744_ = lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go(v_arg_723_, v_fst_737_, v_fst_736_, v_snd_738_, v_arg_721_, v_a_740_);
lean_dec(v_fst_736_);
if (lean_obj_tag(v___x_744_) == 0)
{
lean_object* v___x_745_; lean_object* v___x_746_; 
lean_del_object(v___x_742_);
v___x_745_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1);
v___x_746_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__0___redArg(v___x_745_, v___y_706_, v___y_707_, v___y_708_, v___y_709_);
return v___x_746_;
}
else
{
lean_object* v_val_747_; lean_object* v___x_749_; 
v_val_747_ = lean_ctor_get(v___x_744_, 0);
lean_inc(v_val_747_);
lean_dec_ref_known(v___x_744_, 1);
if (v_isShared_743_ == 0)
{
lean_ctor_set(v___x_742_, 0, v_val_747_);
v___x_749_ = v___x_742_;
goto v_reusejp_748_;
}
else
{
lean_object* v_reuseFailAlloc_750_; 
v_reuseFailAlloc_750_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_750_, 0, v_val_747_);
v___x_749_ = v_reuseFailAlloc_750_;
goto v_reusejp_748_;
}
v_reusejp_748_:
{
return v___x_749_;
}
}
}
}
else
{
lean_dec(v_snd_738_);
lean_dec(v_fst_737_);
lean_dec(v_fst_736_);
lean_dec_ref(v_arg_723_);
lean_dec_ref(v_arg_721_);
return v___x_739_;
}
}
else
{
lean_object* v___x_752_; lean_object* v___x_753_; 
lean_dec(v___x_733_);
lean_dec_ref(v_arg_723_);
lean_dec_ref(v_arg_721_);
v___x_752_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1);
v___x_753_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__0___redArg(v___x_752_, v___y_706_, v___y_707_, v___y_708_, v___y_709_);
return v___x_753_;
}
}
else
{
lean_dec_ref(v_arg_723_);
lean_dec_ref(v_arg_721_);
return v___x_730_;
}
}
}
else
{
lean_dec_ref(v_fn_720_);
lean_dec_ref_known(v_a_712_, 2);
v___y_714_ = v___y_706_;
v___y_715_ = v___y_707_;
v___y_716_ = v___y_708_;
v___y_717_ = v___y_709_;
goto v___jp_713_;
}
}
else
{
lean_dec(v_a_712_);
v___y_714_ = v___y_706_;
v___y_715_ = v___y_707_;
v___y_716_ = v___y_708_;
v___y_717_ = v___y_709_;
goto v___jp_713_;
}
v___jp_713_:
{
lean_object* v___x_718_; lean_object* v___x_719_; 
v___x_718_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1);
v___x_719_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__0___redArg(v___x_718_, v___y_714_, v___y_715_, v___y_716_, v___y_717_);
return v___x_719_;
}
}
else
{
lean_object* v_a_775_; lean_object* v___x_777_; uint8_t v_isShared_778_; uint8_t v_isSharedCheck_782_; 
v_a_775_ = lean_ctor_get(v___x_711_, 0);
v_isSharedCheck_782_ = !lean_is_exclusive(v___x_711_);
if (v_isSharedCheck_782_ == 0)
{
v___x_777_ = v___x_711_;
v_isShared_778_ = v_isSharedCheck_782_;
goto v_resetjp_776_;
}
else
{
lean_inc(v_a_775_);
lean_dec(v___x_711_);
v___x_777_ = lean_box(0);
v_isShared_778_ = v_isSharedCheck_782_;
goto v_resetjp_776_;
}
v_resetjp_776_:
{
lean_object* v___x_780_; 
if (v_isShared_778_ == 0)
{
v___x_780_ = v___x_777_;
goto v_reusejp_779_;
}
else
{
lean_object* v_reuseFailAlloc_781_; 
v_reuseFailAlloc_781_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_781_, 0, v_a_775_);
v___x_780_ = v_reuseFailAlloc_781_;
goto v_reusejp_779_;
}
v_reusejp_779_:
{
return v___x_780_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod___lam__1___boxed(lean_object* v_u_783_, lean_object* v_00_u03b1_784_, lean_object* v_e_785_, lean_object* v___y_786_, lean_object* v___y_787_, lean_object* v___y_788_, lean_object* v___y_789_, lean_object* v___y_790_){
_start:
{
lean_object* v_res_791_; 
v_res_791_ = lp_mathlib_Mathlib_Meta_NormNum_evalIntMod___lam__1(v_u_783_, v_00_u03b1_784_, v_e_785_, v___y_786_, v___y_787_, v___y_788_, v___y_789_);
lean_dec(v___y_789_);
lean_dec_ref(v___y_788_);
lean_dec(v___y_787_);
lean_dec_ref(v___y_786_);
lean_dec_ref(v_00_u03b1_784_);
lean_dec(v_u_783_);
return v_res_791_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__0(void){
_start:
{
lean_object* v___x_804_; lean_object* v___x_805_; 
v___x_804_ = lean_unsigned_to_nat(0u);
v___x_805_ = lean_nat_to_int(v___x_804_);
return v___x_805_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__3(void){
_start:
{
lean_object* v___x_812_; lean_object* v___x_813_; lean_object* v___x_814_; 
v___x_812_ = lean_box(0);
v___x_813_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__2));
v___x_814_ = l_Lean_Expr_const___override(v___x_813_, v___x_812_);
return v___x_814_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__6(void){
_start:
{
lean_object* v___x_821_; lean_object* v___x_822_; lean_object* v___x_823_; 
v___x_821_ = lean_box(0);
v___x_822_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__5));
v___x_823_ = l_Lean_Expr_const___override(v___x_822_, v___x_821_);
return v___x_823_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__10(void){
_start:
{
lean_object* v___x_829_; lean_object* v___x_830_; lean_object* v___x_831_; 
v___x_829_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__5));
v___x_830_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__9));
v___x_831_ = l_Lean_Expr_const___override(v___x_830_, v___x_829_);
return v___x_831_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__11(void){
_start:
{
lean_object* v___x_832_; lean_object* v___x_833_; lean_object* v___x_834_; 
v___x_832_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5, &lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5);
v___x_833_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__10, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__10_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__10);
v___x_834_ = l_Lean_Expr_app___override(v___x_833_, v___x_832_);
return v___x_834_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__14(void){
_start:
{
lean_object* v___x_839_; lean_object* v___x_840_; lean_object* v___x_841_; 
v___x_839_ = lean_box(0);
v___x_840_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__13));
v___x_841_ = l_Lean_Expr_const___override(v___x_840_, v___x_839_);
return v___x_841_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__15(void){
_start:
{
lean_object* v___x_842_; lean_object* v___x_843_; lean_object* v___x_844_; 
v___x_842_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__14, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__14_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__14);
v___x_843_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__11, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__11_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__11);
v___x_844_ = l_Lean_Expr_app___override(v___x_843_, v___x_842_);
return v___x_844_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1(lean_object* v_u_845_, lean_object* v_00_u03b1_846_, lean_object* v_e_847_, lean_object* v___y_848_, lean_object* v___y_849_, lean_object* v___y_850_, lean_object* v___y_851_){
_start:
{
lean_object* v___x_853_; 
v___x_853_ = l_Lean_Meta_whnfR(v_e_847_, v___y_848_, v___y_849_, v___y_850_, v___y_851_);
if (lean_obj_tag(v___x_853_) == 0)
{
lean_object* v_a_854_; lean_object* v___x_856_; uint8_t v_isShared_857_; uint8_t v_isSharedCheck_974_; 
v_a_854_ = lean_ctor_get(v___x_853_, 0);
v_isSharedCheck_974_ = !lean_is_exclusive(v___x_853_);
if (v_isSharedCheck_974_ == 0)
{
v___x_856_ = v___x_853_;
v_isShared_857_ = v_isSharedCheck_974_;
goto v_resetjp_855_;
}
else
{
lean_inc(v_a_854_);
lean_dec(v___x_853_);
v___x_856_ = lean_box(0);
v_isShared_857_ = v_isSharedCheck_974_;
goto v_resetjp_855_;
}
v_resetjp_855_:
{
lean_object* v___y_859_; lean_object* v___y_860_; lean_object* v___y_861_; lean_object* v___y_862_; 
if (lean_obj_tag(v_a_854_) == 5)
{
lean_object* v_fn_865_; 
v_fn_865_ = lean_ctor_get(v_a_854_, 0);
lean_inc_ref(v_fn_865_);
if (lean_obj_tag(v_fn_865_) == 5)
{
lean_object* v_arg_866_; lean_object* v_fn_867_; lean_object* v_arg_868_; lean_object* v___x_869_; lean_object* v___x_870_; lean_object* v___y_872_; lean_object* v___y_873_; lean_object* v___y_874_; lean_object* v_a_875_; lean_object* v___x_913_; lean_object* v___f_914_; uint8_t v___x_915_; lean_object* v___y_917_; lean_object* v_a_918_; lean_object* v___x_953_; 
v_arg_866_ = lean_ctor_get(v_a_854_, 1);
lean_inc_ref(v_arg_866_);
lean_dec_ref_known(v_a_854_, 2);
v_fn_867_ = lean_ctor_get(v_fn_865_, 0);
lean_inc_ref(v_fn_867_);
v_arg_868_ = lean_ctor_get(v_fn_865_, 1);
lean_inc_ref(v_arg_868_);
lean_dec_ref_known(v_fn_865_, 2);
v___x_869_ = lean_box(0);
v___x_870_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5, &lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__5);
v___x_913_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__15, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__15_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__15);
v___f_914_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__0___boxed), 7, 2);
lean_closure_set(v___f_914_, 0, v_fn_867_);
lean_closure_set(v___f_914_, 1, v___x_913_);
v___x_915_ = 0;
v___x_953_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__1___redArg(v___f_914_, v___x_915_, v___y_848_, v___y_849_, v___y_850_, v___y_851_);
if (lean_obj_tag(v___x_953_) == 0)
{
lean_object* v_a_954_; uint8_t v___x_955_; 
v_a_954_ = lean_ctor_get(v___x_953_, 0);
lean_inc(v_a_954_);
lean_dec_ref_known(v___x_953_, 1);
v___x_955_ = lean_unbox(v_a_954_);
lean_dec(v_a_954_);
if (v___x_955_ == 0)
{
lean_object* v___x_956_; lean_object* v___x_957_; lean_object* v_a_958_; lean_object* v___x_960_; uint8_t v_isShared_961_; uint8_t v_isSharedCheck_965_; 
lean_dec_ref(v_arg_868_);
lean_dec_ref(v_arg_866_);
lean_del_object(v___x_856_);
v___x_956_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1);
v___x_957_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__0___redArg(v___x_956_, v___y_848_, v___y_849_, v___y_850_, v___y_851_);
v_a_958_ = lean_ctor_get(v___x_957_, 0);
v_isSharedCheck_965_ = !lean_is_exclusive(v___x_957_);
if (v_isSharedCheck_965_ == 0)
{
v___x_960_ = v___x_957_;
v_isShared_961_ = v_isSharedCheck_965_;
goto v_resetjp_959_;
}
else
{
lean_inc(v_a_958_);
lean_dec(v___x_957_);
v___x_960_ = lean_box(0);
v_isShared_961_ = v_isSharedCheck_965_;
goto v_resetjp_959_;
}
v_resetjp_959_:
{
lean_object* v___x_963_; 
if (v_isShared_961_ == 0)
{
v___x_963_ = v___x_960_;
goto v_reusejp_962_;
}
else
{
lean_object* v_reuseFailAlloc_964_; 
v_reuseFailAlloc_964_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_964_, 0, v_a_958_);
v___x_963_ = v_reuseFailAlloc_964_;
goto v_reusejp_962_;
}
v_reusejp_962_:
{
return v___x_963_;
}
}
}
else
{
goto v___jp_937_;
}
}
else
{
lean_object* v_a_966_; lean_object* v___x_968_; uint8_t v_isShared_969_; uint8_t v_isSharedCheck_973_; 
lean_dec_ref(v_arg_868_);
lean_dec_ref(v_arg_866_);
lean_del_object(v___x_856_);
v_a_966_ = lean_ctor_get(v___x_953_, 0);
v_isSharedCheck_973_ = !lean_is_exclusive(v___x_953_);
if (v_isSharedCheck_973_ == 0)
{
v___x_968_ = v___x_953_;
v_isShared_969_ = v_isSharedCheck_973_;
goto v_resetjp_967_;
}
else
{
lean_inc(v_a_966_);
lean_dec(v___x_953_);
v___x_968_ = lean_box(0);
v_isShared_969_ = v_isSharedCheck_973_;
goto v_resetjp_967_;
}
v_resetjp_967_:
{
lean_object* v___x_971_; 
if (v_isShared_969_ == 0)
{
v___x_971_ = v___x_968_;
goto v_reusejp_970_;
}
else
{
lean_object* v_reuseFailAlloc_972_; 
v_reuseFailAlloc_972_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_972_, 0, v_a_966_);
v___x_971_ = v_reuseFailAlloc_972_;
goto v_reusejp_970_;
}
v_reusejp_970_:
{
return v___x_971_;
}
}
}
v___jp_871_:
{
lean_object* v_snd_876_; lean_object* v_fst_877_; lean_object* v_fst_878_; lean_object* v_snd_879_; lean_object* v___x_880_; lean_object* v___x_881_; uint8_t v___x_882_; 
v_snd_876_ = lean_ctor_get(v_a_875_, 1);
lean_inc(v_snd_876_);
v_fst_877_ = lean_ctor_get(v_a_875_, 0);
lean_inc(v_fst_877_);
lean_dec_ref(v_a_875_);
v_fst_878_ = lean_ctor_get(v_snd_876_, 0);
lean_inc(v_fst_878_);
v_snd_879_ = lean_ctor_get(v_snd_876_, 1);
lean_inc(v_snd_879_);
lean_dec(v_snd_876_);
v___x_880_ = lean_int_emod(v_fst_877_, v___y_873_);
v___x_881_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__0, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__0_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__0);
v___x_882_ = lean_int_dec_eq(v___x_880_, v___x_881_);
lean_dec(v___x_880_);
if (v___x_882_ == 0)
{
lean_object* v___x_883_; lean_object* v___x_884_; lean_object* v___x_885_; lean_object* v___x_886_; lean_object* v___x_887_; lean_object* v___x_888_; lean_object* v___x_889_; lean_object* v___x_890_; lean_object* v___x_891_; lean_object* v___x_892_; lean_object* v___x_894_; 
lean_dec(v_fst_877_);
lean_dec(v___y_873_);
v___x_883_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__17, &lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__17_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__17);
v___x_884_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__3, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__3_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__3);
v___x_885_ = l_Lean_Expr_app___override(v___x_884_, v_arg_868_);
v___x_886_ = l_Lean_Expr_app___override(v___x_885_, v_arg_866_);
v___x_887_ = l_Lean_Expr_app___override(v___x_886_, v___y_874_);
v___x_888_ = l_Lean_Expr_app___override(v___x_887_, v_fst_878_);
v___x_889_ = l_Lean_Expr_app___override(v___x_888_, v___y_872_);
v___x_890_ = l_Lean_Expr_app___override(v___x_889_, v_snd_879_);
v___x_891_ = l_Lean_Expr_app___override(v___x_890_, v___x_883_);
v___x_892_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_892_, 0, v___x_891_);
lean_ctor_set_uint8(v___x_892_, sizeof(void*)*1, v___x_882_);
if (v_isShared_857_ == 0)
{
lean_ctor_set(v___x_856_, 0, v___x_892_);
v___x_894_ = v___x_856_;
goto v_reusejp_893_;
}
else
{
lean_object* v_reuseFailAlloc_895_; 
v_reuseFailAlloc_895_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_895_, 0, v___x_892_);
v___x_894_ = v_reuseFailAlloc_895_;
goto v_reusejp_893_;
}
v_reusejp_893_:
{
return v___x_894_;
}
}
else
{
lean_object* v___x_896_; lean_object* v___x_897_; lean_object* v___x_898_; lean_object* v___x_899_; lean_object* v___x_900_; lean_object* v___x_901_; lean_object* v___x_902_; lean_object* v___x_903_; lean_object* v___x_904_; lean_object* v___x_905_; lean_object* v___x_906_; lean_object* v___x_907_; lean_object* v___x_908_; lean_object* v___x_909_; lean_object* v___x_911_; 
v___x_896_ = lean_int_ediv(v_fst_877_, v___y_873_);
lean_dec(v___y_873_);
lean_dec(v_fst_877_);
v___x_897_ = lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit(v___x_896_);
lean_dec(v___x_896_);
v___x_898_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__6, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__6_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___closed__6);
v___x_899_ = l_Lean_Expr_app___override(v___x_898_, v_arg_868_);
v___x_900_ = l_Lean_Expr_app___override(v___x_899_, v_arg_866_);
v___x_901_ = l_Lean_Expr_app___override(v___x_900_, v___y_874_);
lean_inc(v_fst_878_);
v___x_902_ = l_Lean_Expr_app___override(v___x_901_, v_fst_878_);
v___x_903_ = l_Lean_Expr_app___override(v___x_902_, v___x_897_);
v___x_904_ = l_Lean_Expr_app___override(v___x_903_, v___y_872_);
v___x_905_ = l_Lean_Expr_app___override(v___x_904_, v_snd_879_);
v___x_906_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__9, &lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__9_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_DivMod_0__Mathlib_Meta_NormNum_evalIntDiv_core___closed__9);
v___x_907_ = l_Lean_Expr_app___override(v___x_906_, v_fst_878_);
v___x_908_ = l_Lean_Expr_app___override(v___x_905_, v___x_907_);
v___x_909_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_909_, 0, v___x_908_);
lean_ctor_set_uint8(v___x_909_, sizeof(void*)*1, v___x_882_);
if (v_isShared_857_ == 0)
{
lean_ctor_set(v___x_856_, 0, v___x_909_);
v___x_911_ = v___x_856_;
goto v_reusejp_910_;
}
else
{
lean_object* v_reuseFailAlloc_912_; 
v_reuseFailAlloc_912_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_912_, 0, v___x_909_);
v___x_911_ = v_reuseFailAlloc_912_;
goto v_reusejp_910_;
}
v_reusejp_910_:
{
return v___x_911_;
}
}
}
v___jp_916_:
{
lean_object* v_snd_919_; lean_object* v_fst_920_; lean_object* v_fst_921_; lean_object* v_snd_922_; lean_object* v___x_923_; 
v_snd_919_ = lean_ctor_get(v_a_918_, 1);
lean_inc(v_snd_919_);
v_fst_920_ = lean_ctor_get(v_a_918_, 0);
lean_inc(v_fst_920_);
lean_dec_ref(v_a_918_);
v_fst_921_ = lean_ctor_get(v_snd_919_, 0);
lean_inc(v_fst_921_);
v_snd_922_ = lean_ctor_get(v_snd_919_, 1);
lean_inc(v_snd_922_);
lean_dec(v_snd_919_);
lean_inc_ref(v_arg_866_);
v___x_923_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v___x_869_, v___x_870_, v_arg_866_, v___x_915_, v___y_848_, v___y_849_, v___y_850_, v___y_851_);
if (lean_obj_tag(v___x_923_) == 0)
{
lean_object* v_a_924_; lean_object* v___x_925_; 
v_a_924_ = lean_ctor_get(v___x_923_, 0);
lean_inc(v_a_924_);
lean_dec_ref_known(v___x_923_, 1);
lean_inc_ref(v_arg_866_);
v___x_925_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toInt(v___x_869_, v___x_870_, v_arg_866_, v___y_917_, v_a_924_);
if (lean_obj_tag(v___x_925_) == 0)
{
lean_object* v___x_926_; lean_object* v___x_927_; lean_object* v_a_928_; lean_object* v___x_930_; uint8_t v_isShared_931_; uint8_t v_isSharedCheck_935_; 
lean_dec(v_snd_922_);
lean_dec(v_fst_921_);
lean_dec(v_fst_920_);
lean_dec_ref(v_arg_868_);
lean_dec_ref(v_arg_866_);
lean_del_object(v___x_856_);
v___x_926_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1);
v___x_927_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__0___redArg(v___x_926_, v___y_848_, v___y_849_, v___y_850_, v___y_851_);
v_a_928_ = lean_ctor_get(v___x_927_, 0);
v_isSharedCheck_935_ = !lean_is_exclusive(v___x_927_);
if (v_isSharedCheck_935_ == 0)
{
v___x_930_ = v___x_927_;
v_isShared_931_ = v_isSharedCheck_935_;
goto v_resetjp_929_;
}
else
{
lean_inc(v_a_928_);
lean_dec(v___x_927_);
v___x_930_ = lean_box(0);
v_isShared_931_ = v_isSharedCheck_935_;
goto v_resetjp_929_;
}
v_resetjp_929_:
{
lean_object* v___x_933_; 
if (v_isShared_931_ == 0)
{
v___x_933_ = v___x_930_;
goto v_reusejp_932_;
}
else
{
lean_object* v_reuseFailAlloc_934_; 
v_reuseFailAlloc_934_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_934_, 0, v_a_928_);
v___x_933_ = v_reuseFailAlloc_934_;
goto v_reusejp_932_;
}
v_reusejp_932_:
{
return v___x_933_;
}
}
}
else
{
lean_object* v_val_936_; 
v_val_936_ = lean_ctor_get(v___x_925_, 0);
lean_inc(v_val_936_);
lean_dec_ref_known(v___x_925_, 1);
v___y_872_ = v_snd_922_;
v___y_873_ = v_fst_920_;
v___y_874_ = v_fst_921_;
v_a_875_ = v_val_936_;
goto v___jp_871_;
}
}
else
{
lean_dec(v_snd_922_);
lean_dec(v_fst_921_);
lean_dec(v_fst_920_);
lean_dec_ref(v___y_917_);
lean_dec_ref(v_arg_868_);
lean_dec_ref(v_arg_866_);
lean_del_object(v___x_856_);
return v___x_923_;
}
}
v___jp_937_:
{
lean_object* v___x_938_; 
lean_inc_ref(v_arg_868_);
v___x_938_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v___x_869_, v___x_870_, v_arg_868_, v___x_915_, v___y_848_, v___y_849_, v___y_850_, v___y_851_);
if (lean_obj_tag(v___x_938_) == 0)
{
lean_object* v_a_939_; lean_object* v___x_940_; lean_object* v___x_941_; 
v_a_939_ = lean_ctor_get(v___x_938_, 0);
lean_inc(v_a_939_);
lean_dec_ref_known(v___x_938_, 1);
v___x_940_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__46, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__46_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__46);
lean_inc_ref(v_arg_868_);
v___x_941_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toInt(v___x_869_, v___x_870_, v_arg_868_, v___x_940_, v_a_939_);
if (lean_obj_tag(v___x_941_) == 0)
{
lean_object* v___x_942_; lean_object* v___x_943_; lean_object* v_a_944_; lean_object* v___x_946_; uint8_t v_isShared_947_; uint8_t v_isSharedCheck_951_; 
lean_dec_ref(v_arg_868_);
lean_dec_ref(v_arg_866_);
lean_del_object(v___x_856_);
v___x_942_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1);
v___x_943_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__0___redArg(v___x_942_, v___y_848_, v___y_849_, v___y_850_, v___y_851_);
v_a_944_ = lean_ctor_get(v___x_943_, 0);
v_isSharedCheck_951_ = !lean_is_exclusive(v___x_943_);
if (v_isSharedCheck_951_ == 0)
{
v___x_946_ = v___x_943_;
v_isShared_947_ = v_isSharedCheck_951_;
goto v_resetjp_945_;
}
else
{
lean_inc(v_a_944_);
lean_dec(v___x_943_);
v___x_946_ = lean_box(0);
v_isShared_947_ = v_isSharedCheck_951_;
goto v_resetjp_945_;
}
v_resetjp_945_:
{
lean_object* v___x_949_; 
if (v_isShared_947_ == 0)
{
v___x_949_ = v___x_946_;
goto v_reusejp_948_;
}
else
{
lean_object* v_reuseFailAlloc_950_; 
v_reuseFailAlloc_950_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_950_, 0, v_a_944_);
v___x_949_ = v_reuseFailAlloc_950_;
goto v_reusejp_948_;
}
v_reusejp_948_:
{
return v___x_949_;
}
}
}
else
{
lean_object* v_val_952_; 
v_val_952_ = lean_ctor_get(v___x_941_, 0);
lean_inc(v_val_952_);
lean_dec_ref_known(v___x_941_, 1);
v___y_917_ = v___x_940_;
v_a_918_ = v_val_952_;
goto v___jp_916_;
}
}
else
{
lean_dec_ref(v_arg_868_);
lean_dec_ref(v_arg_866_);
lean_del_object(v___x_856_);
return v___x_938_;
}
}
}
else
{
lean_dec_ref(v_fn_865_);
lean_dec_ref_known(v_a_854_, 2);
lean_del_object(v___x_856_);
v___y_859_ = v___y_848_;
v___y_860_ = v___y_849_;
v___y_861_ = v___y_850_;
v___y_862_ = v___y_851_;
goto v___jp_858_;
}
}
else
{
lean_del_object(v___x_856_);
lean_dec(v_a_854_);
v___y_859_ = v___y_848_;
v___y_860_ = v___y_849_;
v___y_861_ = v___y_850_;
v___y_862_ = v___y_851_;
goto v___jp_858_;
}
v___jp_858_:
{
lean_object* v___x_863_; lean_object* v___x_864_; 
v___x_863_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalIntDiv___lam__1___closed__1);
v___x_864_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalIntDiv_spec__0___redArg(v___x_863_, v___y_859_, v___y_860_, v___y_861_, v___y_862_);
return v___x_864_;
}
}
}
else
{
lean_object* v_a_975_; lean_object* v___x_977_; uint8_t v_isShared_978_; uint8_t v_isSharedCheck_982_; 
v_a_975_ = lean_ctor_get(v___x_853_, 0);
v_isSharedCheck_982_ = !lean_is_exclusive(v___x_853_);
if (v_isSharedCheck_982_ == 0)
{
v___x_977_ = v___x_853_;
v_isShared_978_ = v_isSharedCheck_982_;
goto v_resetjp_976_;
}
else
{
lean_inc(v_a_975_);
lean_dec(v___x_853_);
v___x_977_ = lean_box(0);
v_isShared_978_ = v_isSharedCheck_982_;
goto v_resetjp_976_;
}
v_resetjp_976_:
{
lean_object* v___x_980_; 
if (v_isShared_978_ == 0)
{
v___x_980_ = v___x_977_;
goto v_reusejp_979_;
}
else
{
lean_object* v_reuseFailAlloc_981_; 
v_reuseFailAlloc_981_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_981_, 0, v_a_975_);
v___x_980_ = v_reuseFailAlloc_981_;
goto v_reusejp_979_;
}
v_reusejp_979_:
{
return v___x_980_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1___boxed(lean_object* v_u_983_, lean_object* v_00_u03b1_984_, lean_object* v_e_985_, lean_object* v___y_986_, lean_object* v___y_987_, lean_object* v___y_988_, lean_object* v___y_989_, lean_object* v___y_990_){
_start:
{
lean_object* v_res_991_; 
v_res_991_ = lp_mathlib_Mathlib_Meta_NormNum_evalIntDvd___lam__1(v_u_983_, v_00_u03b1_984_, v_e_985_, v___y_986_, v___y_987_, v___y_988_, v___y_989_);
lean_dec(v___y_989_);
lean_dec_ref(v___y_988_);
lean_dec(v___y_987_);
lean_dec_ref(v___y_986_);
lean_dec_ref(v_00_u03b1_984_);
lean_dec(v_u_983_);
return v_res_991_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Ineq(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NormNum_DivMod(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Ineq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_NormNum_DivMod(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_NormNum_Ineq(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_NormNum_DivMod(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_NormNum_Ineq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NormNum_DivMod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_NormNum_DivMod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_NormNum_DivMod(builtin);
}
#ifdef __cplusplus
}
#endif
