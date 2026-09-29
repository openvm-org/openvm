// Lean compiler output
// Module: Mathlib.Tactic.NormNum.OfScientific
// Imports: public import Init public meta import Init public import Mathlib.Tactic.NormNum.Basic public import Mathlib.Data.Rat.Cast.Defs public import Mathlib.Tactic.Positivity.Basic public import Mathlib.Tactic.SetLike
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
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_Level_succ___override(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_Expr_lit___override(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_whnfR(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferDivisionSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_Lean_Expr_natLit_x21(lean_object*);
lean_object* lean_nat_pow(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* l_Lean_mkRawNatLit(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_derive(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toNNRat_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalOfScientific_spec__1___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalOfScientific_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalOfScientific_spec__1(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalOfScientific_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__1(uint8_t, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_evalOfScientific_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_evalOfScientific_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalOfScientific_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalOfScientific_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__2_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__4;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "OfScientific"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "ofScientific"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__5_value),LEAN_SCALAR_PTR_LITERAL(1, 219, 72, 84, 44, 38, 226, 47)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__6_value),LEAN_SCALAR_PTR_LITERAL(101, 32, 126, 239, 82, 155, 222, 105)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "NNRatCast"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toOfScientific"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__8_value),LEAN_SCALAR_PTR_LITERAL(61, 217, 231, 3, 133, 178, 168, 33)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__9_value),LEAN_SCALAR_PTR_LITERAL(206, 167, 158, 37, 116, 41, 100, 16)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "DivisionSemiring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "toNNRatCast"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__11_value),LEAN_SCALAR_PTR_LITERAL(70, 208, 64, 71, 63, 26, 215, 130)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__13_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__12_value),LEAN_SCALAR_PTR_LITERAL(91, 148, 22, 247, 181, 204, 251, 118)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Bool"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "false"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__14_value),LEAN_SCALAR_PTR_LITERAL(250, 44, 198, 216, 184, 195, 199, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__16_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__15_value),LEAN_SCALAR_PTR_LITERAL(117, 151, 161, 190, 111, 237, 188, 218)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__16_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__17;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "true"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__14_value),LEAN_SCALAR_PTR_LITERAL(250, 44, 198, 216, 184, 195, 199, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__19_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__18_value),LEAN_SCALAR_PTR_LITERAL(22, 245, 194, 28, 184, 9, 113, 128)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__19_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__20;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "nonexhaustive match"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__21_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__22;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "AddCommMonoidWithOne"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__23_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "toAddMonoidWithOne"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__24_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__25_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__23_value),LEAN_SCALAR_PTR_LITERAL(126, 216, 146, 120, 99, 62, 20, 70)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__25_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__24_value),LEAN_SCALAR_PTR_LITERAL(172, 33, 204, 185, 213, 137, 110, 97)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__25_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "NonAssocSemiring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__26_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "toAddCommMonoidWithOne"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__27_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__28_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__26_value),LEAN_SCALAR_PTR_LITERAL(46, 119, 91, 198, 213, 11, 55, 139)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__28_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__27_value),LEAN_SCALAR_PTR_LITERAL(2, 121, 193, 151, 116, 56, 170, 8)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__28_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Semiring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__29_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "toNonAssocSemiring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__30_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__31_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__29_value),LEAN_SCALAR_PTR_LITERAL(37, 127, 172, 14, 25, 240, 239, 179)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__31_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__30_value),LEAN_SCALAR_PTR_LITERAL(146, 92, 66, 67, 127, 202, 60, 223)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__31_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "toSemiring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__32_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__33_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__11_value),LEAN_SCALAR_PTR_LITERAL(70, 208, 64, 71, 63, 26, 215, 130)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__33_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__32_value),LEAN_SCALAR_PTR_LITERAL(241, 35, 131, 210, 203, 216, 146, 177)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__33_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__34_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__35_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "NormNum"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__36_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "isNat_ofScientific_of_false"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__37_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__38_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__34_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__38_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__38_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__35_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__38_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__38_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__36_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__38_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__37_value),LEAN_SCALAR_PTR_LITERAL(165, 38, 134, 17, 184, 46, 233, 222)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__38_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__39 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__39_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "refl"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__40 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__40_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__41_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__39_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__41_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__40_value),LEAN_SCALAR_PTR_LITERAL(72, 6, 107, 181, 0, 125, 21, 187)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__41 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__41_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__42_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__42;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__43_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__43;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__44_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__44;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__45_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__45;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "NNRat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__46 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__46_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "cast"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__47 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__47_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__48_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__46_value),LEAN_SCALAR_PTR_LITERAL(208, 217, 98, 171, 152, 255, 249, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__48_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__47_value),LEAN_SCALAR_PTR_LITERAL(60, 54, 121, 2, 219, 220, 7, 222)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__48 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__48_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "divNat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__49 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__49_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__50_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__46_value),LEAN_SCALAR_PTR_LITERAL(208, 217, 98, 171, 152, 255, 249, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__50_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__49_value),LEAN_SCALAR_PTR_LITERAL(69, 235, 187, 189, 129, 129, 90, 4)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__50 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__50_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__51_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__51;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HPow"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__52 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__52_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hPow"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__53 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__53_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__54_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__52_value),LEAN_SCALAR_PTR_LITERAL(155, 188, 136, 200, 106, 253, 76, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__54_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__53_value),LEAN_SCALAR_PTR_LITERAL(32, 63, 208, 57, 56, 184, 164, 144)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__54 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__54_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__55 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__55_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__56_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__55_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__56 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__56_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__56_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__57 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__57_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__58_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__58;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__59_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__59;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__60_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__60;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__61_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__61;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__62_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instHPow"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__62 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__62_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__63_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__62_value),LEAN_SCALAR_PTR_LITERAL(213, 197, 76, 235, 199, 0, 254, 199)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__63 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__63_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__64_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__64;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__65_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__65;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__66_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__66;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__67_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "NPow"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__67 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__67_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__68_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toPow"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__68 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__68_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__69_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__67_value),LEAN_SCALAR_PTR_LITERAL(39, 79, 240, 225, 164, 207, 253, 237)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__69_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__69_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__68_value),LEAN_SCALAR_PTR_LITERAL(56, 108, 173, 227, 4, 14, 173, 115)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__69 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__69_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__70_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__70;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__71_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__71;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__72_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Monoid"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__72 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__72_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__73_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "toNPow"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__73 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__73_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__74_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__72_value),LEAN_SCALAR_PTR_LITERAL(162, 147, 2, 115, 233, 179, 113, 5)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__74_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__74_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__73_value),LEAN_SCALAR_PTR_LITERAL(224, 31, 132, 245, 47, 70, 119, 231)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__74 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__74_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__75_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__75;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__76_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__76;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__77_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "instMonoid"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__77 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__77_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__78_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__2_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__78_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__78_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__77_value),LEAN_SCALAR_PTR_LITERAL(180, 16, 78, 191, 137, 146, 192, 44)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__78 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__78_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__79_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__79;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__80_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__80;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__81_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__81;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__82_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__82;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__83_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__83;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__84_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "OfNat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__84 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__84_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__85_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ofNat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__85 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__85_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__86_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__84_value),LEAN_SCALAR_PTR_LITERAL(135, 241, 166, 108, 243, 216, 193, 244)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__86_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__86_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__85_value),LEAN_SCALAR_PTR_LITERAL(2, 108, 58, 34, 100, 49, 50, 216)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__86 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__86_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__87_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__87;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__88_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__88;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__89_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(10) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__89 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__89_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__90_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__90;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__91_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__91;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__92_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "instOfNatNat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__92 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__92_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__93_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__92_value),LEAN_SCALAR_PTR_LITERAL(217, 8, 172, 44, 179, 254, 147, 95)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__93 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__93_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__94_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__94;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__95_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__95;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__96_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__96;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__97_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__97;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__98_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "isNNRat_ofScientific_of_true"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__98 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__98_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__99_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__34_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__99_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__99_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__35_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__99_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__99_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__36_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__99_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__99_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__98_value),LEAN_SCALAR_PTR_LITERAL(59, 254, 192, 117, 91, 9, 0, 29)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__99 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__99_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___boxed, .m_arity = 9, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))} };
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "evalOfScientific"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__34_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__35_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___closed__2_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__36_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___closed__1_value),LEAN_SCALAR_PTR_LITERAL(83, 46, 140, 119, 6, 122, 192, 86)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___closed__2_value),LEAN_SCALAR_PTR_LITERAL(1, 1, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___closed__3_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalOfScientific_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalOfScientific_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalOfScientific_spec__1___redArg(lean_object* v_k_1_, uint8_t v_allowLevelAssignments_2_, lean_object* v___y_3_, lean_object* v___y_4_, lean_object* v___y_5_, lean_object* v___y_6_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), v_allowLevelAssignments_2_, v_k_1_, v___y_3_, v___y_4_, v___y_5_, v___y_6_);
if (lean_obj_tag(v___x_8_) == 0)
{
lean_object* v_a_9_; lean_object* v___x_11_; uint8_t v_isShared_12_; uint8_t v_isSharedCheck_16_; 
v_a_9_ = lean_ctor_get(v___x_8_, 0);
v_isSharedCheck_16_ = !lean_is_exclusive(v___x_8_);
if (v_isSharedCheck_16_ == 0)
{
v___x_11_ = v___x_8_;
v_isShared_12_ = v_isSharedCheck_16_;
goto v_resetjp_10_;
}
else
{
lean_inc(v_a_9_);
lean_dec(v___x_8_);
v___x_11_ = lean_box(0);
v_isShared_12_ = v_isSharedCheck_16_;
goto v_resetjp_10_;
}
v_resetjp_10_:
{
lean_object* v___x_14_; 
if (v_isShared_12_ == 0)
{
v___x_14_ = v___x_11_;
goto v_reusejp_13_;
}
else
{
lean_object* v_reuseFailAlloc_15_; 
v_reuseFailAlloc_15_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_15_, 0, v_a_9_);
v___x_14_ = v_reuseFailAlloc_15_;
goto v_reusejp_13_;
}
v_reusejp_13_:
{
return v___x_14_;
}
}
}
else
{
lean_object* v_a_17_; lean_object* v___x_19_; uint8_t v_isShared_20_; uint8_t v_isSharedCheck_24_; 
v_a_17_ = lean_ctor_get(v___x_8_, 0);
v_isSharedCheck_24_ = !lean_is_exclusive(v___x_8_);
if (v_isSharedCheck_24_ == 0)
{
v___x_19_ = v___x_8_;
v_isShared_20_ = v_isSharedCheck_24_;
goto v_resetjp_18_;
}
else
{
lean_inc(v_a_17_);
lean_dec(v___x_8_);
v___x_19_ = lean_box(0);
v_isShared_20_ = v_isSharedCheck_24_;
goto v_resetjp_18_;
}
v_resetjp_18_:
{
lean_object* v___x_22_; 
if (v_isShared_20_ == 0)
{
v___x_22_ = v___x_19_;
goto v_reusejp_21_;
}
else
{
lean_object* v_reuseFailAlloc_23_; 
v_reuseFailAlloc_23_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_23_, 0, v_a_17_);
v___x_22_ = v_reuseFailAlloc_23_;
goto v_reusejp_21_;
}
v_reusejp_21_:
{
return v___x_22_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalOfScientific_spec__1___redArg___boxed(lean_object* v_k_25_, lean_object* v_allowLevelAssignments_26_, lean_object* v___y_27_, lean_object* v___y_28_, lean_object* v___y_29_, lean_object* v___y_30_, lean_object* v___y_31_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_32_; lean_object* v_res_33_; 
v_allowLevelAssignments_boxed_32_ = lean_unbox(v_allowLevelAssignments_26_);
v_res_33_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalOfScientific_spec__1___redArg(v_k_25_, v_allowLevelAssignments_boxed_32_, v___y_27_, v___y_28_, v___y_29_, v___y_30_);
lean_dec(v___y_30_);
lean_dec_ref(v___y_29_);
lean_dec(v___y_28_);
lean_dec_ref(v___y_27_);
return v_res_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalOfScientific_spec__1(lean_object* v_00_u03b1_34_, lean_object* v_k_35_, uint8_t v_allowLevelAssignments_36_, lean_object* v___y_37_, lean_object* v___y_38_, lean_object* v___y_39_, lean_object* v___y_40_){
_start:
{
lean_object* v___x_42_; 
v___x_42_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalOfScientific_spec__1___redArg(v_k_35_, v_allowLevelAssignments_36_, v___y_37_, v___y_38_, v___y_39_, v___y_40_);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalOfScientific_spec__1___boxed(lean_object* v_00_u03b1_43_, lean_object* v_k_44_, lean_object* v_allowLevelAssignments_45_, lean_object* v___y_46_, lean_object* v___y_47_, lean_object* v___y_48_, lean_object* v___y_49_, lean_object* v___y_50_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_51_; lean_object* v_res_52_; 
v_allowLevelAssignments_boxed_51_ = lean_unbox(v_allowLevelAssignments_45_);
v_res_52_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalOfScientific_spec__1(v_00_u03b1_43_, v_k_44_, v_allowLevelAssignments_boxed_51_, v___y_46_, v___y_47_, v___y_48_, v___y_49_);
lean_dec(v___y_49_);
lean_dec_ref(v___y_48_);
lean_dec(v___y_47_);
lean_dec_ref(v___y_46_);
return v_res_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__0(lean_object* v_fn_53_, lean_object* v___x_54_, lean_object* v___y_55_, lean_object* v___y_56_, lean_object* v___y_57_, lean_object* v___y_58_){
_start:
{
lean_object* v___x_60_; 
v___x_60_ = l_Lean_Meta_isExprDefEq(v_fn_53_, v___x_54_, v___y_55_, v___y_56_, v___y_57_, v___y_58_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__0___boxed(lean_object* v_fn_61_, lean_object* v___x_62_, lean_object* v___y_63_, lean_object* v___y_64_, lean_object* v___y_65_, lean_object* v___y_66_, lean_object* v___y_67_){
_start:
{
lean_object* v_res_68_; 
v_res_68_ = lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__0(v_fn_61_, v___x_62_, v___y_63_, v___y_64_, v___y_65_, v___y_66_);
lean_dec(v___y_66_);
lean_dec_ref(v___y_65_);
lean_dec(v___y_64_);
lean_dec_ref(v___y_63_);
return v_res_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__1(uint8_t v___x_69_, lean_object* v___x_70_, lean_object* v_arg_71_, uint8_t v___x_72_, uint8_t v___x_73_, lean_object* v___y_74_, lean_object* v___y_75_, lean_object* v___y_76_, lean_object* v___y_77_){
_start:
{
lean_object* v_keyedConfig_79_; uint8_t v_trackZetaDelta_80_; lean_object* v_zetaDeltaSet_81_; lean_object* v_lctx_82_; lean_object* v_localInstances_83_; lean_object* v_defEqCtx_x3f_84_; lean_object* v_synthPendingDepth_85_; lean_object* v_customCanUnfoldPredicate_x3f_86_; uint8_t v_univApprox_87_; uint8_t v_inTypeClassResolution_88_; uint8_t v_cacheInferType_89_; lean_object* v___x_91_; uint8_t v_isShared_92_; uint8_t v_isSharedCheck_112_; 
v_keyedConfig_79_ = lean_ctor_get(v___y_74_, 0);
v_trackZetaDelta_80_ = lean_ctor_get_uint8(v___y_74_, sizeof(void*)*7);
v_zetaDeltaSet_81_ = lean_ctor_get(v___y_74_, 1);
v_lctx_82_ = lean_ctor_get(v___y_74_, 2);
v_localInstances_83_ = lean_ctor_get(v___y_74_, 3);
v_defEqCtx_x3f_84_ = lean_ctor_get(v___y_74_, 4);
v_synthPendingDepth_85_ = lean_ctor_get(v___y_74_, 5);
v_customCanUnfoldPredicate_x3f_86_ = lean_ctor_get(v___y_74_, 6);
v_univApprox_87_ = lean_ctor_get_uint8(v___y_74_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_88_ = lean_ctor_get_uint8(v___y_74_, sizeof(void*)*7 + 2);
v_cacheInferType_89_ = lean_ctor_get_uint8(v___y_74_, sizeof(void*)*7 + 3);
v_isSharedCheck_112_ = !lean_is_exclusive(v___y_74_);
if (v_isSharedCheck_112_ == 0)
{
v___x_91_ = v___y_74_;
v_isShared_92_ = v_isSharedCheck_112_;
goto v_resetjp_90_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_86_);
lean_inc(v_synthPendingDepth_85_);
lean_inc(v_defEqCtx_x3f_84_);
lean_inc(v_localInstances_83_);
lean_inc(v_lctx_82_);
lean_inc(v_zetaDeltaSet_81_);
lean_inc(v_keyedConfig_79_);
lean_dec(v___y_74_);
v___x_91_ = lean_box(0);
v_isShared_92_ = v_isSharedCheck_112_;
goto v_resetjp_90_;
}
v_resetjp_90_:
{
lean_object* v___x_93_; lean_object* v___x_95_; 
v___x_93_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_69_, v_keyedConfig_79_);
if (v_isShared_92_ == 0)
{
lean_ctor_set(v___x_91_, 0, v___x_93_);
v___x_95_ = v___x_91_;
goto v_reusejp_94_;
}
else
{
lean_object* v_reuseFailAlloc_111_; 
v_reuseFailAlloc_111_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_111_, 0, v___x_93_);
lean_ctor_set(v_reuseFailAlloc_111_, 1, v_zetaDeltaSet_81_);
lean_ctor_set(v_reuseFailAlloc_111_, 2, v_lctx_82_);
lean_ctor_set(v_reuseFailAlloc_111_, 3, v_localInstances_83_);
lean_ctor_set(v_reuseFailAlloc_111_, 4, v_defEqCtx_x3f_84_);
lean_ctor_set(v_reuseFailAlloc_111_, 5, v_synthPendingDepth_85_);
lean_ctor_set(v_reuseFailAlloc_111_, 6, v_customCanUnfoldPredicate_x3f_86_);
lean_ctor_set_uint8(v_reuseFailAlloc_111_, sizeof(void*)*7, v_trackZetaDelta_80_);
lean_ctor_set_uint8(v_reuseFailAlloc_111_, sizeof(void*)*7 + 1, v_univApprox_87_);
lean_ctor_set_uint8(v_reuseFailAlloc_111_, sizeof(void*)*7 + 2, v_inTypeClassResolution_88_);
lean_ctor_set_uint8(v_reuseFailAlloc_111_, sizeof(void*)*7 + 3, v_cacheInferType_89_);
v___x_95_ = v_reuseFailAlloc_111_;
goto v_reusejp_94_;
}
v_reusejp_94_:
{
lean_object* v___x_96_; 
v___x_96_ = l_Lean_Meta_isExprDefEq(v___x_70_, v_arg_71_, v___x_95_, v___y_75_, v___y_76_, v___y_77_);
lean_dec_ref(v___x_95_);
if (lean_obj_tag(v___x_96_) == 0)
{
lean_object* v_a_97_; lean_object* v___x_99_; uint8_t v_isShared_100_; uint8_t v_isSharedCheck_110_; 
v_a_97_ = lean_ctor_get(v___x_96_, 0);
v_isSharedCheck_110_ = !lean_is_exclusive(v___x_96_);
if (v_isSharedCheck_110_ == 0)
{
v___x_99_ = v___x_96_;
v_isShared_100_ = v_isSharedCheck_110_;
goto v_resetjp_98_;
}
else
{
lean_inc(v_a_97_);
lean_dec(v___x_96_);
v___x_99_ = lean_box(0);
v_isShared_100_ = v_isSharedCheck_110_;
goto v_resetjp_98_;
}
v_resetjp_98_:
{
uint8_t v___x_101_; 
v___x_101_ = lean_unbox(v_a_97_);
lean_dec(v_a_97_);
if (v___x_101_ == 0)
{
lean_object* v___x_102_; lean_object* v___x_104_; 
v___x_102_ = lean_box(v___x_72_);
if (v_isShared_100_ == 0)
{
lean_ctor_set(v___x_99_, 0, v___x_102_);
v___x_104_ = v___x_99_;
goto v_reusejp_103_;
}
else
{
lean_object* v_reuseFailAlloc_105_; 
v_reuseFailAlloc_105_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_105_, 0, v___x_102_);
v___x_104_ = v_reuseFailAlloc_105_;
goto v_reusejp_103_;
}
v_reusejp_103_:
{
return v___x_104_;
}
}
else
{
lean_object* v___x_106_; lean_object* v___x_108_; 
v___x_106_ = lean_box(v___x_73_);
if (v_isShared_100_ == 0)
{
lean_ctor_set(v___x_99_, 0, v___x_106_);
v___x_108_ = v___x_99_;
goto v_reusejp_107_;
}
else
{
lean_object* v_reuseFailAlloc_109_; 
v_reuseFailAlloc_109_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_109_, 0, v___x_106_);
v___x_108_ = v_reuseFailAlloc_109_;
goto v_reusejp_107_;
}
v_reusejp_107_:
{
return v___x_108_;
}
}
}
}
else
{
return v___x_96_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__1___boxed(lean_object* v___x_113_, lean_object* v___x_114_, lean_object* v_arg_115_, lean_object* v___x_116_, lean_object* v___x_117_, lean_object* v___y_118_, lean_object* v___y_119_, lean_object* v___y_120_, lean_object* v___y_121_, lean_object* v___y_122_){
_start:
{
uint8_t v___x_9195__boxed_123_; uint8_t v___x_9198__boxed_124_; uint8_t v___x_9199__boxed_125_; lean_object* v_res_126_; 
v___x_9195__boxed_123_ = lean_unbox(v___x_113_);
v___x_9198__boxed_124_ = lean_unbox(v___x_116_);
v___x_9199__boxed_125_ = lean_unbox(v___x_117_);
v_res_126_ = lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__1(v___x_9195__boxed_123_, v___x_114_, v_arg_115_, v___x_9198__boxed_124_, v___x_9199__boxed_125_, v___y_118_, v___y_119_, v___y_120_, v___y_121_);
lean_dec(v___y_121_);
lean_dec_ref(v___y_120_);
lean_dec(v___y_119_);
return v_res_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_evalOfScientific_spec__0_spec__0(lean_object* v_msgData_127_, lean_object* v___y_128_, lean_object* v___y_129_, lean_object* v___y_130_, lean_object* v___y_131_){
_start:
{
lean_object* v___x_133_; lean_object* v_env_134_; lean_object* v___x_135_; lean_object* v_mctx_136_; lean_object* v_lctx_137_; lean_object* v_options_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; 
v___x_133_ = lean_st_ref_get(v___y_131_);
v_env_134_ = lean_ctor_get(v___x_133_, 0);
lean_inc_ref(v_env_134_);
lean_dec(v___x_133_);
v___x_135_ = lean_st_ref_get(v___y_129_);
v_mctx_136_ = lean_ctor_get(v___x_135_, 0);
lean_inc_ref(v_mctx_136_);
lean_dec(v___x_135_);
v_lctx_137_ = lean_ctor_get(v___y_128_, 2);
v_options_138_ = lean_ctor_get(v___y_130_, 2);
lean_inc_ref(v_options_138_);
lean_inc_ref(v_lctx_137_);
v___x_139_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_139_, 0, v_env_134_);
lean_ctor_set(v___x_139_, 1, v_mctx_136_);
lean_ctor_set(v___x_139_, 2, v_lctx_137_);
lean_ctor_set(v___x_139_, 3, v_options_138_);
v___x_140_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_140_, 0, v___x_139_);
lean_ctor_set(v___x_140_, 1, v_msgData_127_);
v___x_141_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_141_, 0, v___x_140_);
return v___x_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_evalOfScientific_spec__0_spec__0___boxed(lean_object* v_msgData_142_, lean_object* v___y_143_, lean_object* v___y_144_, lean_object* v___y_145_, lean_object* v___y_146_, lean_object* v___y_147_){
_start:
{
lean_object* v_res_148_; 
v_res_148_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_evalOfScientific_spec__0_spec__0(v_msgData_142_, v___y_143_, v___y_144_, v___y_145_, v___y_146_);
lean_dec(v___y_146_);
lean_dec_ref(v___y_145_);
lean_dec(v___y_144_);
lean_dec_ref(v___y_143_);
return v_res_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalOfScientific_spec__0___redArg(lean_object* v_msg_149_, lean_object* v___y_150_, lean_object* v___y_151_, lean_object* v___y_152_, lean_object* v___y_153_){
_start:
{
lean_object* v_ref_155_; lean_object* v___x_156_; lean_object* v_a_157_; lean_object* v___x_159_; uint8_t v_isShared_160_; uint8_t v_isSharedCheck_165_; 
v_ref_155_ = lean_ctor_get(v___y_152_, 5);
v___x_156_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_evalOfScientific_spec__0_spec__0(v_msg_149_, v___y_150_, v___y_151_, v___y_152_, v___y_153_);
v_a_157_ = lean_ctor_get(v___x_156_, 0);
v_isSharedCheck_165_ = !lean_is_exclusive(v___x_156_);
if (v_isSharedCheck_165_ == 0)
{
v___x_159_ = v___x_156_;
v_isShared_160_ = v_isSharedCheck_165_;
goto v_resetjp_158_;
}
else
{
lean_inc(v_a_157_);
lean_dec(v___x_156_);
v___x_159_ = lean_box(0);
v_isShared_160_ = v_isSharedCheck_165_;
goto v_resetjp_158_;
}
v_resetjp_158_:
{
lean_object* v___x_161_; lean_object* v___x_163_; 
lean_inc(v_ref_155_);
v___x_161_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_161_, 0, v_ref_155_);
lean_ctor_set(v___x_161_, 1, v_a_157_);
if (v_isShared_160_ == 0)
{
lean_ctor_set_tag(v___x_159_, 1);
lean_ctor_set(v___x_159_, 0, v___x_161_);
v___x_163_ = v___x_159_;
goto v_reusejp_162_;
}
else
{
lean_object* v_reuseFailAlloc_164_; 
v_reuseFailAlloc_164_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_164_, 0, v___x_161_);
v___x_163_ = v_reuseFailAlloc_164_;
goto v_reusejp_162_;
}
v_reusejp_162_:
{
return v___x_163_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalOfScientific_spec__0___redArg___boxed(lean_object* v_msg_166_, lean_object* v___y_167_, lean_object* v___y_168_, lean_object* v___y_169_, lean_object* v___y_170_, lean_object* v___y_171_){
_start:
{
lean_object* v_res_172_; 
v_res_172_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalOfScientific_spec__0___redArg(v_msg_166_, v___y_167_, v___y_168_, v___y_169_, v___y_170_);
lean_dec(v___y_170_);
lean_dec_ref(v___y_169_);
lean_dec(v___y_168_);
lean_dec_ref(v___y_167_);
return v_res_172_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__1(void){
_start:
{
lean_object* v___x_174_; lean_object* v___x_175_; 
v___x_174_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__0));
v___x_175_ = l_Lean_stringToMessageData(v___x_174_);
return v___x_175_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__4(void){
_start:
{
lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; 
v___x_179_ = lean_box(0);
v___x_180_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__3));
v___x_181_ = l_Lean_Expr_const___override(v___x_180_, v___x_179_);
return v___x_181_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__17(void){
_start:
{
lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; 
v___x_202_ = lean_box(0);
v___x_203_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__16));
v___x_204_ = l_Lean_Expr_const___override(v___x_203_, v___x_202_);
return v___x_204_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__20(void){
_start:
{
lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; 
v___x_209_ = lean_box(0);
v___x_210_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__19));
v___x_211_ = l_Lean_Expr_const___override(v___x_210_, v___x_209_);
return v___x_211_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__22(void){
_start:
{
lean_object* v___x_213_; lean_object* v___x_214_; 
v___x_213_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__21));
v___x_214_ = l_Lean_stringToMessageData(v___x_213_);
return v___x_214_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__42(void){
_start:
{
lean_object* v___x_248_; lean_object* v___x_249_; 
v___x_248_ = lean_box(0);
v___x_249_ = l_Lean_Level_succ___override(v___x_248_);
return v___x_249_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__43(void){
_start:
{
lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; 
v___x_250_ = lean_box(0);
v___x_251_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__42, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__42_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__42);
v___x_252_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_252_, 0, v___x_251_);
lean_ctor_set(v___x_252_, 1, v___x_250_);
return v___x_252_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__44(void){
_start:
{
lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; 
v___x_253_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__43, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__43_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__43);
v___x_254_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__41));
v___x_255_ = l_Lean_Expr_const___override(v___x_254_, v___x_253_);
return v___x_255_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__45(void){
_start:
{
lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; 
v___x_256_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__4, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__4_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__4);
v___x_257_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__44, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__44_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__44);
v___x_258_ = l_Lean_Expr_app___override(v___x_257_, v___x_256_);
return v___x_258_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__51(void){
_start:
{
lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; 
v___x_268_ = lean_box(0);
v___x_269_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__50));
v___x_270_ = l_Lean_Expr_const___override(v___x_269_, v___x_268_);
return v___x_270_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__58(void){
_start:
{
lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; 
v___x_285_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__57));
v___x_286_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__54));
v___x_287_ = l_Lean_Expr_const___override(v___x_286_, v___x_285_);
return v___x_287_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__59(void){
_start:
{
lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_290_; 
v___x_288_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__4, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__4_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__4);
v___x_289_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__58, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__58_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__58);
v___x_290_ = l_Lean_Expr_app___override(v___x_289_, v___x_288_);
return v___x_290_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__60(void){
_start:
{
lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; 
v___x_291_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__4, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__4_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__4);
v___x_292_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__59, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__59_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__59);
v___x_293_ = l_Lean_Expr_app___override(v___x_292_, v___x_291_);
return v___x_293_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__61(void){
_start:
{
lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; 
v___x_294_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__4, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__4_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__4);
v___x_295_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__60, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__60_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__60);
v___x_296_ = l_Lean_Expr_app___override(v___x_295_, v___x_294_);
return v___x_296_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__64(void){
_start:
{
lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; 
v___x_300_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__56));
v___x_301_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__63));
v___x_302_ = l_Lean_Expr_const___override(v___x_301_, v___x_300_);
return v___x_302_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__65(void){
_start:
{
lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; 
v___x_303_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__4, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__4_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__4);
v___x_304_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__64, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__64_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__64);
v___x_305_ = l_Lean_Expr_app___override(v___x_304_, v___x_303_);
return v___x_305_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__66(void){
_start:
{
lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; 
v___x_306_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__4, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__4_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__4);
v___x_307_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__65, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__65_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__65);
v___x_308_ = l_Lean_Expr_app___override(v___x_307_, v___x_306_);
return v___x_308_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__70(void){
_start:
{
lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; 
v___x_314_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__55));
v___x_315_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__69));
v___x_316_ = l_Lean_Expr_const___override(v___x_315_, v___x_314_);
return v___x_316_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__71(void){
_start:
{
lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; 
v___x_317_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__4, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__4_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__4);
v___x_318_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__70, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__70_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__70);
v___x_319_ = l_Lean_Expr_app___override(v___x_318_, v___x_317_);
return v___x_319_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__75(void){
_start:
{
lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; 
v___x_325_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__55));
v___x_326_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__74));
v___x_327_ = l_Lean_Expr_const___override(v___x_326_, v___x_325_);
return v___x_327_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__76(void){
_start:
{
lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; 
v___x_328_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__4, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__4_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__4);
v___x_329_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__75, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__75_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__75);
v___x_330_ = l_Lean_Expr_app___override(v___x_329_, v___x_328_);
return v___x_330_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__79(void){
_start:
{
lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v___x_337_; 
v___x_335_ = lean_box(0);
v___x_336_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__78));
v___x_337_ = l_Lean_Expr_const___override(v___x_336_, v___x_335_);
return v___x_337_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__80(void){
_start:
{
lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v___x_340_; 
v___x_338_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__79, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__79_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__79);
v___x_339_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__76, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__76_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__76);
v___x_340_ = l_Lean_Expr_app___override(v___x_339_, v___x_338_);
return v___x_340_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__81(void){
_start:
{
lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; 
v___x_341_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__80, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__80_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__80);
v___x_342_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__71, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__71_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__71);
v___x_343_ = l_Lean_Expr_app___override(v___x_342_, v___x_341_);
return v___x_343_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__82(void){
_start:
{
lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; 
v___x_344_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__81, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__81_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__81);
v___x_345_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__66, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__66_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__66);
v___x_346_ = l_Lean_Expr_app___override(v___x_345_, v___x_344_);
return v___x_346_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__83(void){
_start:
{
lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; 
v___x_347_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__82, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__82_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__82);
v___x_348_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__61, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__61_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__61);
v___x_349_ = l_Lean_Expr_app___override(v___x_348_, v___x_347_);
return v___x_349_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__87(void){
_start:
{
lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; 
v___x_355_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__55));
v___x_356_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__86));
v___x_357_ = l_Lean_Expr_const___override(v___x_356_, v___x_355_);
return v___x_357_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__88(void){
_start:
{
lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; 
v___x_358_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__4, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__4_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__4);
v___x_359_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__87, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__87_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__87);
v___x_360_ = l_Lean_Expr_app___override(v___x_359_, v___x_358_);
return v___x_360_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__90(void){
_start:
{
lean_object* v___x_363_; lean_object* v___x_364_; 
v___x_363_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__89));
v___x_364_ = l_Lean_Expr_lit___override(v___x_363_);
return v___x_364_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__91(void){
_start:
{
lean_object* v___x_365_; lean_object* v___x_366_; lean_object* v___x_367_; 
v___x_365_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__90, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__90_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__90);
v___x_366_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__88, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__88_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__88);
v___x_367_ = l_Lean_Expr_app___override(v___x_366_, v___x_365_);
return v___x_367_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__94(void){
_start:
{
lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; 
v___x_371_ = lean_box(0);
v___x_372_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__93));
v___x_373_ = l_Lean_Expr_const___override(v___x_372_, v___x_371_);
return v___x_373_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__95(void){
_start:
{
lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; 
v___x_374_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__90, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__90_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__90);
v___x_375_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__94, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__94_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__94);
v___x_376_ = l_Lean_Expr_app___override(v___x_375_, v___x_374_);
return v___x_376_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__96(void){
_start:
{
lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v___x_379_; 
v___x_377_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__95, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__95_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__95);
v___x_378_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__91, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__91_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__91);
v___x_379_ = l_Lean_Expr_app___override(v___x_378_, v___x_377_);
return v___x_379_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__97(void){
_start:
{
lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_382_; 
v___x_380_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__96, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__96_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__96);
v___x_381_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__83, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__83_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__83);
v___x_382_ = l_Lean_Expr_app___override(v___x_381_, v___x_380_);
return v___x_382_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3(uint8_t v___x_389_, lean_object* v_u_390_, lean_object* v_00_u03b1_391_, lean_object* v_e_392_, lean_object* v___y_393_, lean_object* v___y_394_, lean_object* v___y_395_, lean_object* v___y_396_){
_start:
{
lean_object* v___x_398_; 
v___x_398_ = l_Lean_Meta_whnfR(v_e_392_, v___y_393_, v___y_394_, v___y_395_, v___y_396_);
if (lean_obj_tag(v___x_398_) == 0)
{
lean_object* v_a_399_; lean_object* v___y_401_; lean_object* v___y_402_; lean_object* v___y_403_; lean_object* v___y_404_; 
v_a_399_ = lean_ctor_get(v___x_398_, 0);
lean_inc(v_a_399_);
lean_dec_ref_known(v___x_398_, 1);
if (lean_obj_tag(v_a_399_) == 5)
{
lean_object* v_fn_407_; 
v_fn_407_ = lean_ctor_get(v_a_399_, 0);
lean_inc_ref(v_fn_407_);
if (lean_obj_tag(v_fn_407_) == 5)
{
lean_object* v_fn_408_; 
v_fn_408_ = lean_ctor_get(v_fn_407_, 0);
lean_inc_ref(v_fn_408_);
if (lean_obj_tag(v_fn_408_) == 5)
{
lean_object* v_arg_409_; lean_object* v_arg_410_; lean_object* v_fn_411_; lean_object* v_arg_412_; lean_object* v___x_413_; 
v_arg_409_ = lean_ctor_get(v_a_399_, 1);
lean_inc_ref(v_arg_409_);
lean_dec_ref_known(v_a_399_, 2);
v_arg_410_ = lean_ctor_get(v_fn_407_, 1);
lean_inc_ref(v_arg_410_);
lean_dec_ref_known(v_fn_407_, 2);
v_fn_411_ = lean_ctor_get(v_fn_408_, 0);
lean_inc_ref(v_fn_411_);
v_arg_412_ = lean_ctor_get(v_fn_408_, 1);
lean_inc_ref(v_arg_412_);
lean_dec_ref_known(v_fn_408_, 2);
lean_inc_ref(v_00_u03b1_391_);
lean_inc(v_u_390_);
v___x_413_ = lp_mathlib_Mathlib_Meta_NormNum_inferDivisionSemiring(v_u_390_, v_00_u03b1_391_, v___y_393_, v___y_394_, v___y_395_, v___y_396_);
if (lean_obj_tag(v___x_413_) == 0)
{
lean_object* v_a_414_; lean_object* v___x_415_; lean_object* v___x_416_; lean_object* v___x_417_; lean_object* v___x_418_; lean_object* v___x_419_; lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v___x_427_; lean_object* v___x_428_; lean_object* v___x_429_; lean_object* v___f_430_; uint8_t v___x_431_; lean_object* v___x_432_; 
v_a_414_ = lean_ctor_get(v___x_413_, 0);
lean_inc_n(v_a_414_, 2);
lean_dec_ref_known(v___x_413_, 1);
v___x_415_ = lean_box(0);
v___x_416_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__4, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__4_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__4);
v___x_417_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__7));
lean_inc(v_u_390_);
v___x_418_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_418_, 0, v_u_390_);
lean_ctor_set(v___x_418_, 1, v___x_415_);
lean_inc_ref_n(v___x_418_, 3);
v___x_419_ = l_Lean_Expr_const___override(v___x_417_, v___x_418_);
lean_inc_ref_n(v_00_u03b1_391_, 3);
v___x_420_ = l_Lean_Expr_app___override(v___x_419_, v_00_u03b1_391_);
v___x_421_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__10));
v___x_422_ = l_Lean_Expr_const___override(v___x_421_, v___x_418_);
v___x_423_ = l_Lean_Expr_app___override(v___x_422_, v_00_u03b1_391_);
v___x_424_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__13));
v___x_425_ = l_Lean_Expr_const___override(v___x_424_, v___x_418_);
v___x_426_ = l_Lean_Expr_app___override(v___x_425_, v_00_u03b1_391_);
v___x_427_ = l_Lean_Expr_app___override(v___x_426_, v_a_414_);
lean_inc_ref(v___x_427_);
v___x_428_ = l_Lean_Expr_app___override(v___x_423_, v___x_427_);
v___x_429_ = l_Lean_Expr_app___override(v___x_420_, v___x_428_);
v___f_430_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__0___boxed), 7, 2);
lean_closure_set(v___f_430_, 0, v_fn_411_);
lean_closure_set(v___f_430_, 1, v___x_429_);
v___x_431_ = 0;
v___x_432_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalOfScientific_spec__1___redArg(v___f_430_, v___x_431_, v___y_393_, v___y_394_, v___y_395_, v___y_396_);
if (lean_obj_tag(v___x_432_) == 0)
{
lean_object* v_a_433_; uint8_t v___x_578_; 
v_a_433_ = lean_ctor_get(v___x_432_, 0);
lean_inc(v_a_433_);
lean_dec_ref_known(v___x_432_, 1);
v___x_578_ = lean_unbox(v_a_433_);
lean_dec(v_a_433_);
if (v___x_578_ == 0)
{
lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v_a_581_; lean_object* v___x_583_; uint8_t v_isShared_584_; uint8_t v_isSharedCheck_588_; 
lean_dec_ref(v___x_427_);
lean_dec_ref_known(v___x_418_, 2);
lean_dec(v_a_414_);
lean_dec_ref(v_arg_412_);
lean_dec_ref(v_arg_410_);
lean_dec_ref(v_arg_409_);
lean_dec_ref(v_00_u03b1_391_);
lean_dec(v_u_390_);
v___x_579_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__1);
v___x_580_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalOfScientific_spec__0___redArg(v___x_579_, v___y_393_, v___y_394_, v___y_395_, v___y_396_);
v_a_581_ = lean_ctor_get(v___x_580_, 0);
v_isSharedCheck_588_ = !lean_is_exclusive(v___x_580_);
if (v_isSharedCheck_588_ == 0)
{
v___x_583_ = v___x_580_;
v_isShared_584_ = v_isSharedCheck_588_;
goto v_resetjp_582_;
}
else
{
lean_inc(v_a_581_);
lean_dec(v___x_580_);
v___x_583_ = lean_box(0);
v_isShared_584_ = v_isSharedCheck_588_;
goto v_resetjp_582_;
}
v_resetjp_582_:
{
lean_object* v___x_586_; 
if (v_isShared_584_ == 0)
{
v___x_586_ = v___x_583_;
goto v_reusejp_585_;
}
else
{
lean_object* v_reuseFailAlloc_587_; 
v_reuseFailAlloc_587_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_587_, 0, v_a_581_);
v___x_586_ = v_reuseFailAlloc_587_;
goto v_reusejp_585_;
}
v_reusejp_585_:
{
return v___x_586_;
}
}
}
else
{
goto v___jp_434_;
}
v___jp_434_:
{
lean_object* v___x_435_; lean_object* v___x_436_; uint8_t v___x_437_; lean_object* v___x_438_; lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___f_441_; lean_object* v___x_442_; 
v___x_435_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__17, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__17_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__17);
v___x_436_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__20, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__20_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__20);
v___x_437_ = 2;
v___x_438_ = lean_box(v___x_437_);
v___x_439_ = lean_box(v___x_431_);
v___x_440_ = lean_box(v___x_389_);
lean_inc_ref(v_arg_410_);
v___f_441_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__1___boxed), 10, 5);
lean_closure_set(v___f_441_, 0, v___x_438_);
lean_closure_set(v___f_441_, 1, v___x_436_);
lean_closure_set(v___f_441_, 2, v_arg_410_);
lean_closure_set(v___f_441_, 3, v___x_439_);
lean_closure_set(v___f_441_, 4, v___x_440_);
v___x_442_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalOfScientific_spec__1___redArg(v___f_441_, v___x_431_, v___y_393_, v___y_394_, v___y_395_, v___y_396_);
if (lean_obj_tag(v___x_442_) == 0)
{
lean_object* v_a_443_; uint8_t v___x_444_; 
v_a_443_ = lean_ctor_get(v___x_442_, 0);
lean_inc(v_a_443_);
lean_dec_ref_known(v___x_442_, 1);
v___x_444_ = lean_unbox(v_a_443_);
lean_dec(v_a_443_);
if (v___x_444_ == 0)
{
lean_object* v___x_445_; lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v___f_448_; lean_object* v___x_449_; 
lean_dec_ref(v___x_427_);
lean_dec(v_u_390_);
v___x_445_ = lean_box(v___x_437_);
v___x_446_ = lean_box(v___x_431_);
v___x_447_ = lean_box(v___x_389_);
v___f_448_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__1___boxed), 10, 5);
lean_closure_set(v___f_448_, 0, v___x_445_);
lean_closure_set(v___f_448_, 1, v___x_435_);
lean_closure_set(v___f_448_, 2, v_arg_410_);
lean_closure_set(v___f_448_, 3, v___x_446_);
lean_closure_set(v___f_448_, 4, v___x_447_);
v___x_449_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalOfScientific_spec__1___redArg(v___f_448_, v___x_431_, v___y_393_, v___y_394_, v___y_395_, v___y_396_);
if (lean_obj_tag(v___x_449_) == 0)
{
lean_object* v_a_450_; uint8_t v___x_451_; 
v_a_450_ = lean_ctor_get(v___x_449_, 0);
lean_inc(v_a_450_);
lean_dec_ref_known(v___x_449_, 1);
v___x_451_ = lean_unbox(v_a_450_);
lean_dec(v_a_450_);
if (v___x_451_ == 0)
{
lean_object* v___x_452_; lean_object* v___x_453_; 
lean_dec_ref_known(v___x_418_, 2);
lean_dec(v_a_414_);
lean_dec_ref(v_arg_412_);
lean_dec_ref(v_arg_409_);
lean_dec_ref(v_00_u03b1_391_);
v___x_452_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__22, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__22_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__22);
v___x_453_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalOfScientific_spec__0___redArg(v___x_452_, v___y_393_, v___y_394_, v___y_395_, v___y_396_);
return v___x_453_;
}
else
{
lean_object* v___x_454_; lean_object* v___x_455_; 
v___x_454_ = lean_box(0);
lean_inc_ref(v_arg_412_);
v___x_455_ = lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg(v___x_454_, v___x_416_, v_arg_412_, v___y_393_, v___y_394_, v___y_395_, v___y_396_);
if (lean_obj_tag(v___x_455_) == 0)
{
lean_object* v_a_456_; lean_object* v_fst_457_; lean_object* v_snd_458_; lean_object* v___x_459_; 
v_a_456_ = lean_ctor_get(v___x_455_, 0);
lean_inc(v_a_456_);
lean_dec_ref_known(v___x_455_, 1);
v_fst_457_ = lean_ctor_get(v_a_456_, 0);
lean_inc(v_fst_457_);
v_snd_458_ = lean_ctor_get(v_a_456_, 1);
lean_inc(v_snd_458_);
lean_dec(v_a_456_);
lean_inc_ref(v_arg_409_);
v___x_459_ = lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg(v___x_454_, v___x_416_, v_arg_409_, v___y_393_, v___y_394_, v___y_395_, v___y_396_);
if (lean_obj_tag(v___x_459_) == 0)
{
lean_object* v_a_460_; lean_object* v___x_462_; uint8_t v_isShared_463_; uint8_t v_isSharedCheck_506_; 
v_a_460_ = lean_ctor_get(v___x_459_, 0);
v_isSharedCheck_506_ = !lean_is_exclusive(v___x_459_);
if (v_isSharedCheck_506_ == 0)
{
v___x_462_ = v___x_459_;
v_isShared_463_ = v_isSharedCheck_506_;
goto v_resetjp_461_;
}
else
{
lean_inc(v_a_460_);
lean_dec(v___x_459_);
v___x_462_ = lean_box(0);
v_isShared_463_ = v_isSharedCheck_506_;
goto v_resetjp_461_;
}
v_resetjp_461_:
{
lean_object* v_fst_464_; lean_object* v_snd_465_; lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; lean_object* v___x_475_; lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v___x_494_; lean_object* v___x_495_; lean_object* v___x_496_; lean_object* v___x_497_; lean_object* v___x_498_; lean_object* v___x_499_; lean_object* v___x_500_; lean_object* v___x_501_; lean_object* v___x_502_; lean_object* v___x_504_; 
v_fst_464_ = lean_ctor_get(v_a_460_, 0);
lean_inc(v_fst_464_);
v_snd_465_ = lean_ctor_get(v_a_460_, 1);
lean_inc(v_snd_465_);
lean_dec(v_a_460_);
v___x_466_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__25));
v___x_467_ = lp_batteries_Lean_Expr_natLit_x21(v_fst_457_);
v___x_468_ = lp_batteries_Lean_Expr_natLit_x21(v_fst_464_);
v___x_469_ = lean_unsigned_to_nat(10u);
v___x_470_ = lean_nat_pow(v___x_469_, v___x_468_);
lean_dec(v___x_468_);
v___x_471_ = lean_nat_mul(v___x_467_, v___x_470_);
lean_dec(v___x_470_);
lean_dec(v___x_467_);
v___x_472_ = l_Lean_mkRawNatLit(v___x_471_);
lean_inc_ref_n(v___x_418_, 4);
v___x_473_ = l_Lean_Expr_const___override(v___x_466_, v___x_418_);
lean_inc_ref_n(v_00_u03b1_391_, 4);
v___x_474_ = l_Lean_Expr_app___override(v___x_473_, v_00_u03b1_391_);
v___x_475_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__28));
v___x_476_ = l_Lean_Expr_const___override(v___x_475_, v___x_418_);
v___x_477_ = l_Lean_Expr_app___override(v___x_476_, v_00_u03b1_391_);
v___x_478_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__31));
v___x_479_ = l_Lean_Expr_const___override(v___x_478_, v___x_418_);
v___x_480_ = l_Lean_Expr_app___override(v___x_479_, v_00_u03b1_391_);
v___x_481_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__33));
v___x_482_ = l_Lean_Expr_const___override(v___x_481_, v___x_418_);
v___x_483_ = l_Lean_Expr_app___override(v___x_482_, v_00_u03b1_391_);
lean_inc(v_a_414_);
v___x_484_ = l_Lean_Expr_app___override(v___x_483_, v_a_414_);
v___x_485_ = l_Lean_Expr_app___override(v___x_480_, v___x_484_);
v___x_486_ = l_Lean_Expr_app___override(v___x_477_, v___x_485_);
v___x_487_ = l_Lean_Expr_app___override(v___x_474_, v___x_486_);
v___x_488_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__38));
v___x_489_ = l_Lean_Expr_const___override(v___x_488_, v___x_418_);
v___x_490_ = l_Lean_Expr_app___override(v___x_489_, v_00_u03b1_391_);
v___x_491_ = l_Lean_Expr_app___override(v___x_490_, v_a_414_);
v___x_492_ = l_Lean_Expr_app___override(v___x_491_, v_arg_412_);
v___x_493_ = l_Lean_Expr_app___override(v___x_492_, v_arg_409_);
v___x_494_ = l_Lean_Expr_app___override(v___x_493_, v_fst_457_);
v___x_495_ = l_Lean_Expr_app___override(v___x_494_, v_fst_464_);
lean_inc_ref_n(v___x_472_, 2);
v___x_496_ = l_Lean_Expr_app___override(v___x_495_, v___x_472_);
v___x_497_ = l_Lean_Expr_app___override(v___x_496_, v_snd_458_);
v___x_498_ = l_Lean_Expr_app___override(v___x_497_, v_snd_465_);
v___x_499_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__45, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__45_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__45);
v___x_500_ = l_Lean_Expr_app___override(v___x_499_, v___x_472_);
v___x_501_ = l_Lean_Expr_app___override(v___x_498_, v___x_500_);
v___x_502_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_502_, 0, v___x_487_);
lean_ctor_set(v___x_502_, 1, v___x_472_);
lean_ctor_set(v___x_502_, 2, v___x_501_);
if (v_isShared_463_ == 0)
{
lean_ctor_set(v___x_462_, 0, v___x_502_);
v___x_504_ = v___x_462_;
goto v_reusejp_503_;
}
else
{
lean_object* v_reuseFailAlloc_505_; 
v_reuseFailAlloc_505_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_505_, 0, v___x_502_);
v___x_504_ = v_reuseFailAlloc_505_;
goto v_reusejp_503_;
}
v_reusejp_503_:
{
return v___x_504_;
}
}
}
else
{
lean_object* v_a_507_; lean_object* v___x_509_; uint8_t v_isShared_510_; uint8_t v_isSharedCheck_514_; 
lean_dec(v_snd_458_);
lean_dec(v_fst_457_);
lean_dec_ref_known(v___x_418_, 2);
lean_dec(v_a_414_);
lean_dec_ref(v_arg_412_);
lean_dec_ref(v_arg_409_);
lean_dec_ref(v_00_u03b1_391_);
v_a_507_ = lean_ctor_get(v___x_459_, 0);
v_isSharedCheck_514_ = !lean_is_exclusive(v___x_459_);
if (v_isSharedCheck_514_ == 0)
{
v___x_509_ = v___x_459_;
v_isShared_510_ = v_isSharedCheck_514_;
goto v_resetjp_508_;
}
else
{
lean_inc(v_a_507_);
lean_dec(v___x_459_);
v___x_509_ = lean_box(0);
v_isShared_510_ = v_isSharedCheck_514_;
goto v_resetjp_508_;
}
v_resetjp_508_:
{
lean_object* v___x_512_; 
if (v_isShared_510_ == 0)
{
v___x_512_ = v___x_509_;
goto v_reusejp_511_;
}
else
{
lean_object* v_reuseFailAlloc_513_; 
v_reuseFailAlloc_513_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_513_, 0, v_a_507_);
v___x_512_ = v_reuseFailAlloc_513_;
goto v_reusejp_511_;
}
v_reusejp_511_:
{
return v___x_512_;
}
}
}
}
else
{
lean_object* v_a_515_; lean_object* v___x_517_; uint8_t v_isShared_518_; uint8_t v_isSharedCheck_522_; 
lean_dec_ref_known(v___x_418_, 2);
lean_dec(v_a_414_);
lean_dec_ref(v_arg_412_);
lean_dec_ref(v_arg_409_);
lean_dec_ref(v_00_u03b1_391_);
v_a_515_ = lean_ctor_get(v___x_455_, 0);
v_isSharedCheck_522_ = !lean_is_exclusive(v___x_455_);
if (v_isSharedCheck_522_ == 0)
{
v___x_517_ = v___x_455_;
v_isShared_518_ = v_isSharedCheck_522_;
goto v_resetjp_516_;
}
else
{
lean_inc(v_a_515_);
lean_dec(v___x_455_);
v___x_517_ = lean_box(0);
v_isShared_518_ = v_isSharedCheck_522_;
goto v_resetjp_516_;
}
v_resetjp_516_:
{
lean_object* v___x_520_; 
if (v_isShared_518_ == 0)
{
v___x_520_ = v___x_517_;
goto v_reusejp_519_;
}
else
{
lean_object* v_reuseFailAlloc_521_; 
v_reuseFailAlloc_521_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_521_, 0, v_a_515_);
v___x_520_ = v_reuseFailAlloc_521_;
goto v_reusejp_519_;
}
v_reusejp_519_:
{
return v___x_520_;
}
}
}
}
}
else
{
lean_object* v_a_523_; lean_object* v___x_525_; uint8_t v_isShared_526_; uint8_t v_isSharedCheck_530_; 
lean_dec_ref_known(v___x_418_, 2);
lean_dec(v_a_414_);
lean_dec_ref(v_arg_412_);
lean_dec_ref(v_arg_409_);
lean_dec_ref(v_00_u03b1_391_);
v_a_523_ = lean_ctor_get(v___x_449_, 0);
v_isSharedCheck_530_ = !lean_is_exclusive(v___x_449_);
if (v_isSharedCheck_530_ == 0)
{
v___x_525_ = v___x_449_;
v_isShared_526_ = v_isSharedCheck_530_;
goto v_resetjp_524_;
}
else
{
lean_inc(v_a_523_);
lean_dec(v___x_449_);
v___x_525_ = lean_box(0);
v_isShared_526_ = v_isSharedCheck_530_;
goto v_resetjp_524_;
}
v_resetjp_524_:
{
lean_object* v___x_528_; 
if (v_isShared_526_ == 0)
{
v___x_528_ = v___x_525_;
goto v_reusejp_527_;
}
else
{
lean_object* v_reuseFailAlloc_529_; 
v_reuseFailAlloc_529_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_529_, 0, v_a_523_);
v___x_528_ = v_reuseFailAlloc_529_;
goto v_reusejp_527_;
}
v_reusejp_527_:
{
return v___x_528_;
}
}
}
}
else
{
lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; 
lean_dec_ref(v_arg_410_);
v___x_531_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__48));
lean_inc_ref(v___x_418_);
v___x_532_ = l_Lean_Expr_const___override(v___x_531_, v___x_418_);
lean_inc_ref_n(v_00_u03b1_391_, 2);
v___x_533_ = l_Lean_Expr_app___override(v___x_532_, v_00_u03b1_391_);
v___x_534_ = l_Lean_Expr_app___override(v___x_533_, v___x_427_);
v___x_535_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__51, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__51_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__51);
lean_inc_ref(v_arg_412_);
v___x_536_ = l_Lean_Expr_app___override(v___x_535_, v_arg_412_);
v___x_537_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__97, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__97_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__97);
lean_inc_ref(v_arg_409_);
v___x_538_ = l_Lean_Expr_app___override(v___x_537_, v_arg_409_);
v___x_539_ = l_Lean_Expr_app___override(v___x_536_, v___x_538_);
v___x_540_ = l_Lean_Expr_app___override(v___x_534_, v___x_539_);
lean_inc_ref(v___x_540_);
lean_inc(v_u_390_);
v___x_541_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v_u_390_, v_00_u03b1_391_, v___x_540_, v___x_431_, v___y_393_, v___y_394_, v___y_395_, v___y_396_);
if (lean_obj_tag(v___x_541_) == 0)
{
lean_object* v_a_542_; lean_object* v___x_544_; uint8_t v_isShared_545_; uint8_t v_isSharedCheck_569_; 
v_a_542_ = lean_ctor_get(v___x_541_, 0);
v_isSharedCheck_569_ = !lean_is_exclusive(v___x_541_);
if (v_isSharedCheck_569_ == 0)
{
v___x_544_ = v___x_541_;
v_isShared_545_ = v_isSharedCheck_569_;
goto v_resetjp_543_;
}
else
{
lean_inc(v_a_542_);
lean_dec(v___x_541_);
v___x_544_ = lean_box(0);
v_isShared_545_ = v_isSharedCheck_569_;
goto v_resetjp_543_;
}
v_resetjp_543_:
{
lean_object* v___x_546_; 
lean_inc(v_a_414_);
lean_inc_ref(v_00_u03b1_391_);
v___x_546_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toNNRat_x27(v_u_390_, v_00_u03b1_391_, v___x_540_, v_a_414_, v_a_542_);
if (lean_obj_tag(v___x_546_) == 1)
{
lean_object* v_val_547_; lean_object* v_snd_548_; lean_object* v_snd_549_; lean_object* v_fst_550_; lean_object* v_fst_551_; lean_object* v_fst_552_; lean_object* v_snd_553_; lean_object* v___x_554_; lean_object* v___x_555_; lean_object* v___x_556_; lean_object* v___x_557_; lean_object* v___x_558_; lean_object* v___x_559_; lean_object* v___x_560_; lean_object* v___x_561_; lean_object* v___x_562_; lean_object* v___x_563_; lean_object* v___x_565_; 
v_val_547_ = lean_ctor_get(v___x_546_, 0);
lean_inc(v_val_547_);
lean_dec_ref_known(v___x_546_, 1);
v_snd_548_ = lean_ctor_get(v_val_547_, 1);
lean_inc(v_snd_548_);
v_snd_549_ = lean_ctor_get(v_snd_548_, 1);
lean_inc(v_snd_549_);
v_fst_550_ = lean_ctor_get(v_val_547_, 0);
lean_inc(v_fst_550_);
lean_dec(v_val_547_);
v_fst_551_ = lean_ctor_get(v_snd_548_, 0);
lean_inc_n(v_fst_551_, 2);
lean_dec(v_snd_548_);
v_fst_552_ = lean_ctor_get(v_snd_549_, 0);
lean_inc_n(v_fst_552_, 2);
v_snd_553_ = lean_ctor_get(v_snd_549_, 1);
lean_inc(v_snd_553_);
lean_dec(v_snd_549_);
v___x_554_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__99));
v___x_555_ = l_Lean_Expr_const___override(v___x_554_, v___x_418_);
v___x_556_ = l_Lean_Expr_app___override(v___x_555_, v_00_u03b1_391_);
lean_inc(v_a_414_);
v___x_557_ = l_Lean_Expr_app___override(v___x_556_, v_a_414_);
v___x_558_ = l_Lean_Expr_app___override(v___x_557_, v_arg_412_);
v___x_559_ = l_Lean_Expr_app___override(v___x_558_, v_arg_409_);
v___x_560_ = l_Lean_Expr_app___override(v___x_559_, v_fst_551_);
v___x_561_ = l_Lean_Expr_app___override(v___x_560_, v_fst_552_);
v___x_562_ = l_Lean_Expr_app___override(v___x_561_, v_snd_553_);
v___x_563_ = lean_alloc_ctor(3, 5, 0);
lean_ctor_set(v___x_563_, 0, v_a_414_);
lean_ctor_set(v___x_563_, 1, v_fst_550_);
lean_ctor_set(v___x_563_, 2, v_fst_551_);
lean_ctor_set(v___x_563_, 3, v_fst_552_);
lean_ctor_set(v___x_563_, 4, v___x_562_);
if (v_isShared_545_ == 0)
{
lean_ctor_set(v___x_544_, 0, v___x_563_);
v___x_565_ = v___x_544_;
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
lean_object* v___x_567_; lean_object* v___x_568_; 
lean_dec(v___x_546_);
lean_del_object(v___x_544_);
lean_dec_ref_known(v___x_418_, 2);
lean_dec(v_a_414_);
lean_dec_ref(v_arg_412_);
lean_dec_ref(v_arg_409_);
lean_dec_ref(v_00_u03b1_391_);
v___x_567_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__1);
v___x_568_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalOfScientific_spec__0___redArg(v___x_567_, v___y_393_, v___y_394_, v___y_395_, v___y_396_);
return v___x_568_;
}
}
}
else
{
lean_dec_ref(v___x_540_);
lean_dec_ref_known(v___x_418_, 2);
lean_dec(v_a_414_);
lean_dec_ref(v_arg_412_);
lean_dec_ref(v_arg_409_);
lean_dec_ref(v_00_u03b1_391_);
lean_dec(v_u_390_);
return v___x_541_;
}
}
}
else
{
lean_object* v_a_570_; lean_object* v___x_572_; uint8_t v_isShared_573_; uint8_t v_isSharedCheck_577_; 
lean_dec_ref(v___x_427_);
lean_dec_ref_known(v___x_418_, 2);
lean_dec(v_a_414_);
lean_dec_ref(v_arg_412_);
lean_dec_ref(v_arg_410_);
lean_dec_ref(v_arg_409_);
lean_dec_ref(v_00_u03b1_391_);
lean_dec(v_u_390_);
v_a_570_ = lean_ctor_get(v___x_442_, 0);
v_isSharedCheck_577_ = !lean_is_exclusive(v___x_442_);
if (v_isSharedCheck_577_ == 0)
{
v___x_572_ = v___x_442_;
v_isShared_573_ = v_isSharedCheck_577_;
goto v_resetjp_571_;
}
else
{
lean_inc(v_a_570_);
lean_dec(v___x_442_);
v___x_572_ = lean_box(0);
v_isShared_573_ = v_isSharedCheck_577_;
goto v_resetjp_571_;
}
v_resetjp_571_:
{
lean_object* v___x_575_; 
if (v_isShared_573_ == 0)
{
v___x_575_ = v___x_572_;
goto v_reusejp_574_;
}
else
{
lean_object* v_reuseFailAlloc_576_; 
v_reuseFailAlloc_576_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_576_, 0, v_a_570_);
v___x_575_ = v_reuseFailAlloc_576_;
goto v_reusejp_574_;
}
v_reusejp_574_:
{
return v___x_575_;
}
}
}
}
}
else
{
lean_object* v_a_589_; lean_object* v___x_591_; uint8_t v_isShared_592_; uint8_t v_isSharedCheck_596_; 
lean_dec_ref(v___x_427_);
lean_dec_ref_known(v___x_418_, 2);
lean_dec(v_a_414_);
lean_dec_ref(v_arg_412_);
lean_dec_ref(v_arg_410_);
lean_dec_ref(v_arg_409_);
lean_dec_ref(v_00_u03b1_391_);
lean_dec(v_u_390_);
v_a_589_ = lean_ctor_get(v___x_432_, 0);
v_isSharedCheck_596_ = !lean_is_exclusive(v___x_432_);
if (v_isSharedCheck_596_ == 0)
{
v___x_591_ = v___x_432_;
v_isShared_592_ = v_isSharedCheck_596_;
goto v_resetjp_590_;
}
else
{
lean_inc(v_a_589_);
lean_dec(v___x_432_);
v___x_591_ = lean_box(0);
v_isShared_592_ = v_isSharedCheck_596_;
goto v_resetjp_590_;
}
v_resetjp_590_:
{
lean_object* v___x_594_; 
if (v_isShared_592_ == 0)
{
v___x_594_ = v___x_591_;
goto v_reusejp_593_;
}
else
{
lean_object* v_reuseFailAlloc_595_; 
v_reuseFailAlloc_595_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_595_, 0, v_a_589_);
v___x_594_ = v_reuseFailAlloc_595_;
goto v_reusejp_593_;
}
v_reusejp_593_:
{
return v___x_594_;
}
}
}
}
else
{
lean_object* v_a_597_; lean_object* v___x_599_; uint8_t v_isShared_600_; uint8_t v_isSharedCheck_604_; 
lean_dec_ref(v_arg_412_);
lean_dec_ref(v_fn_411_);
lean_dec_ref(v_arg_410_);
lean_dec_ref(v_arg_409_);
lean_dec_ref(v_00_u03b1_391_);
lean_dec(v_u_390_);
v_a_597_ = lean_ctor_get(v___x_413_, 0);
v_isSharedCheck_604_ = !lean_is_exclusive(v___x_413_);
if (v_isSharedCheck_604_ == 0)
{
v___x_599_ = v___x_413_;
v_isShared_600_ = v_isSharedCheck_604_;
goto v_resetjp_598_;
}
else
{
lean_inc(v_a_597_);
lean_dec(v___x_413_);
v___x_599_ = lean_box(0);
v_isShared_600_ = v_isSharedCheck_604_;
goto v_resetjp_598_;
}
v_resetjp_598_:
{
lean_object* v___x_602_; 
if (v_isShared_600_ == 0)
{
v___x_602_ = v___x_599_;
goto v_reusejp_601_;
}
else
{
lean_object* v_reuseFailAlloc_603_; 
v_reuseFailAlloc_603_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_603_, 0, v_a_597_);
v___x_602_ = v_reuseFailAlloc_603_;
goto v_reusejp_601_;
}
v_reusejp_601_:
{
return v___x_602_;
}
}
}
}
else
{
lean_dec_ref_known(v_fn_407_, 2);
lean_dec_ref(v_fn_408_);
lean_dec_ref_known(v_a_399_, 2);
lean_dec_ref(v_00_u03b1_391_);
lean_dec(v_u_390_);
v___y_401_ = v___y_393_;
v___y_402_ = v___y_394_;
v___y_403_ = v___y_395_;
v___y_404_ = v___y_396_;
goto v___jp_400_;
}
}
else
{
lean_dec_ref(v_fn_407_);
lean_dec_ref_known(v_a_399_, 2);
lean_dec_ref(v_00_u03b1_391_);
lean_dec(v_u_390_);
v___y_401_ = v___y_393_;
v___y_402_ = v___y_394_;
v___y_403_ = v___y_395_;
v___y_404_ = v___y_396_;
goto v___jp_400_;
}
}
else
{
lean_dec(v_a_399_);
lean_dec_ref(v_00_u03b1_391_);
lean_dec(v_u_390_);
v___y_401_ = v___y_393_;
v___y_402_ = v___y_394_;
v___y_403_ = v___y_395_;
v___y_404_ = v___y_396_;
goto v___jp_400_;
}
v___jp_400_:
{
lean_object* v___x_405_; lean_object* v___x_406_; 
v___x_405_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___closed__1);
v___x_406_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalOfScientific_spec__0___redArg(v___x_405_, v___y_401_, v___y_402_, v___y_403_, v___y_404_);
return v___x_406_;
}
}
else
{
lean_object* v_a_605_; lean_object* v___x_607_; uint8_t v_isShared_608_; uint8_t v_isSharedCheck_612_; 
lean_dec_ref(v_00_u03b1_391_);
lean_dec(v_u_390_);
v_a_605_ = lean_ctor_get(v___x_398_, 0);
v_isSharedCheck_612_ = !lean_is_exclusive(v___x_398_);
if (v_isSharedCheck_612_ == 0)
{
v___x_607_ = v___x_398_;
v_isShared_608_ = v_isSharedCheck_612_;
goto v_resetjp_606_;
}
else
{
lean_inc(v_a_605_);
lean_dec(v___x_398_);
v___x_607_ = lean_box(0);
v_isShared_608_ = v_isSharedCheck_612_;
goto v_resetjp_606_;
}
v_resetjp_606_:
{
lean_object* v___x_610_; 
if (v_isShared_608_ == 0)
{
v___x_610_ = v___x_607_;
goto v_reusejp_609_;
}
else
{
lean_object* v_reuseFailAlloc_611_; 
v_reuseFailAlloc_611_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_611_, 0, v_a_605_);
v___x_610_ = v_reuseFailAlloc_611_;
goto v_reusejp_609_;
}
v_reusejp_609_:
{
return v___x_610_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3___boxed(lean_object* v___x_613_, lean_object* v_u_614_, lean_object* v_00_u03b1_615_, lean_object* v_e_616_, lean_object* v___y_617_, lean_object* v___y_618_, lean_object* v___y_619_, lean_object* v___y_620_, lean_object* v___y_621_){
_start:
{
uint8_t v___x_9774__boxed_622_; lean_object* v_res_623_; 
v___x_9774__boxed_622_ = lean_unbox(v___x_613_);
v_res_623_ = lp_mathlib_Mathlib_Meta_NormNum_evalOfScientific___lam__3(v___x_9774__boxed_622_, v_u_614_, v_00_u03b1_615_, v_e_616_, v___y_617_, v___y_618_, v___y_619_, v___y_620_);
lean_dec(v___y_620_);
lean_dec_ref(v___y_619_);
lean_dec(v___y_618_);
lean_dec_ref(v___y_617_);
return v_res_623_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalOfScientific_spec__0(lean_object* v_00_u03b1_638_, lean_object* v_msg_639_, lean_object* v___y_640_, lean_object* v___y_641_, lean_object* v___y_642_, lean_object* v___y_643_){
_start:
{
lean_object* v___x_645_; 
v___x_645_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalOfScientific_spec__0___redArg(v_msg_639_, v___y_640_, v___y_641_, v___y_642_, v___y_643_);
return v___x_645_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalOfScientific_spec__0___boxed(lean_object* v_00_u03b1_646_, lean_object* v_msg_647_, lean_object* v___y_648_, lean_object* v___y_649_, lean_object* v___y_650_, lean_object* v___y_651_, lean_object* v___y_652_){
_start:
{
lean_object* v_res_653_; 
v_res_653_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalOfScientific_spec__0(v_00_u03b1_646_, v_msg_647_, v___y_648_, v___y_649_, v___y_650_, v___y_651_);
lean_dec(v___y_651_);
lean_dec_ref(v___y_650_);
lean_dec(v___y_649_);
lean_dec_ref(v___y_648_);
return v_res_653_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Rat_Cast_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Positivity_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_SetLike(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NormNum_OfScientific(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Rat_Cast_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Positivity_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_SetLike(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_NormNum_OfScientific(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Tactic_NormNum_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Rat_Cast_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Positivity_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_SetLike(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_NormNum_OfScientific(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_NormNum_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Rat_Cast_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Positivity_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_SetLike(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NormNum_OfScientific(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_NormNum_OfScientific(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_NormNum_OfScientific(builtin);
}
#ifdef __cplusplus
}
#endif
