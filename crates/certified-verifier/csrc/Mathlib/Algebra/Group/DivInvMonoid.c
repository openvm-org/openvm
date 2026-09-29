// Lean compiler output
// Module: Mathlib.Algebra.Group.DivInvMonoid
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Monoid public import Mathlib.Data.Int.Notation public import Mathlib.Tactic.CrossRefAttribute public import Mathlib.Tactic.OfNat
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
lean_object* lean_nat_to_int(lean_object*);
uint8_t lean_int_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_abs(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
static lean_once_cell_t lp_mathlib_zpowRec___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_zpowRec___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_zpowRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zpowRec___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zpowRec(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zpowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zsmulRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zsmulRec___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zsmulRec(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zsmulRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivInvMonoid_div_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivInvMonoid_div_x27___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivInvMonoid_div_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivInvMonoid_div_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZPow_toPow___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZPow_toPow___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZPow_toPow(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZSMul_toSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZSMul_toSMul(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZPow_ofPow___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZPow_ofPow___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZPow_ofPow(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZSMul_ofSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZSMul_ofSMul(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__0 = (const lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__0_value;
static const lean_string_object lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__1 = (const lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__1_value;
static const lean_string_object lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__2 = (const lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__2_value;
static const lean_string_object lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__3 = (const lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__3_value;
static const lean_ctor_object lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__4_value_aux_0),((lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__4_value_aux_1),((lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__4_value_aux_2),((lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__4 = (const lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__4_value;
static const lean_array_object lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__5 = (const lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__5_value;
static const lean_string_object lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__6 = (const lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__6_value;
static const lean_ctor_object lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__7_value_aux_0),((lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__7_value_aux_1),((lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__7_value_aux_2),((lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__6_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__7 = (const lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__7_value;
static const lean_string_object lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__8 = (const lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__8_value;
static const lean_ctor_object lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__9 = (const lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__9_value;
static const lean_string_object lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "intros"};
static const lean_object* lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__10 = (const lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__10_value;
static const lean_ctor_object lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__11_value_aux_0),((lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__11_value_aux_1),((lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__11_value_aux_2),((lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__10_value),LEAN_SCALAR_PTR_LITERAL(26, 175, 18, 116, 252, 50, 128, 45)}};
static const lean_object* lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__11 = (const lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__11_value;
static lean_once_cell_t lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__12;
static lean_once_cell_t lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__13;
static const lean_ctor_object lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__9_value),((lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__5_value)}};
static const lean_object* lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__14 = (const lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__14_value;
static lean_once_cell_t lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__15;
static lean_once_cell_t lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__16;
static lean_once_cell_t lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__17;
static const lean_string_object lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ";"};
static const lean_object* lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__18 = (const lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__18_value;
static lean_once_cell_t lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__19;
static lean_once_cell_t lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__20;
static const lean_string_object lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticRfl"};
static const lean_object* lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__21 = (const lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__21_value;
static const lean_ctor_object lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__22_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__22_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__22_value_aux_0),((lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__22_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__22_value_aux_1),((lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__22_value_aux_2),((lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__21_value),LEAN_SCALAR_PTR_LITERAL(201, 188, 173, 198, 169, 252, 183, 45)}};
static const lean_object* lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__22 = (const lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__22_value;
static const lean_string_object lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "rfl"};
static const lean_object* lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__23 = (const lean_object*)&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__23_value;
static lean_once_cell_t lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__24;
static lean_once_cell_t lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__25;
static lean_once_cell_t lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__26;
static lean_once_cell_t lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__27;
static lean_once_cell_t lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__28;
static lean_once_cell_t lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__29;
static lean_once_cell_t lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__30;
static lean_once_cell_t lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__31;
static lean_once_cell_t lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__32_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__32;
LEAN_EXPORT lean_object* lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_DivInvMonoid_zpow__zero_x27___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_DivInvMonoid_zpow__succ_x27___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_DivInvMonoid_zpow__neg_x27___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_SubNegMonoid_sub_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubNegMonoid_sub_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubNegMonoid_sub__eq__add__neg___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_SubNegMonoid_zsmul__zero_x27___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_SubNegMonoid_zsmul__succ_x27___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_SubNegMonoid_zsmul__neg_x27___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubtractionMonoid_toInvolutiveNeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubtractionMonoid_toInvolutiveNeg___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubtractionMonoid_toInvolutiveNeg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubtractionMonoid_toInvolutiveNeg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionMonoid_toInvolutiveInv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionMonoid_toInvolutiveInv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionMonoid_toInvolutiveInv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionMonoid_toInvolutiveInv___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubtractionCommMonoid_toAddCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubtractionCommMonoid_toAddCommMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubtractionCommMonoid_toAddCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubtractionCommMonoid_toAddCommMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionCommMonoid_toCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionCommMonoid_toCommMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionCommMonoid_toCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionCommMonoid_toCommMonoid___boxed(lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_zpowRec___redArg___closed__0(void){
_start:
{
lean_object* v_natZero_1_; lean_object* v_intZero_2_; 
v_natZero_1_ = lean_unsigned_to_nat(0u);
v_intZero_2_ = lean_nat_to_int(v_natZero_1_);
return v_intZero_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zpowRec___redArg(lean_object* v_inst_3_, lean_object* v_npow_4_, lean_object* v_x_5_, lean_object* v_x_6_){
_start:
{
lean_object* v_intZero_7_; uint8_t v_isNeg_8_; 
v_intZero_7_ = lean_obj_once(&lp_mathlib_zpowRec___redArg___closed__0, &lp_mathlib_zpowRec___redArg___closed__0_once, _init_lp_mathlib_zpowRec___redArg___closed__0);
v_isNeg_8_ = lean_int_dec_lt(v_x_5_, v_intZero_7_);
if (v_isNeg_8_ == 0)
{
lean_object* v_a_9_; lean_object* v___x_10_; 
lean_dec(v_inst_3_);
v_a_9_ = lean_nat_abs(v_x_5_);
v___x_10_ = lean_apply_2(v_npow_4_, v_a_9_, v_x_6_);
return v___x_10_;
}
else
{
lean_object* v_abs_11_; lean_object* v_one_12_; lean_object* v_a_13_; lean_object* v___x_14_; lean_object* v___x_15_; lean_object* v___x_16_; 
v_abs_11_ = lean_nat_abs(v_x_5_);
v_one_12_ = lean_unsigned_to_nat(1u);
v_a_13_ = lean_nat_sub(v_abs_11_, v_one_12_);
lean_dec(v_abs_11_);
v___x_14_ = lean_nat_add(v_a_13_, v_one_12_);
lean_dec(v_a_13_);
v___x_15_ = lean_apply_2(v_npow_4_, v___x_14_, v_x_6_);
v___x_16_ = lean_apply_1(v_inst_3_, v___x_15_);
return v___x_16_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_zpowRec___redArg___boxed(lean_object* v_inst_17_, lean_object* v_npow_18_, lean_object* v_x_19_, lean_object* v_x_20_){
_start:
{
lean_object* v_res_21_; 
v_res_21_ = lp_mathlib_zpowRec___redArg(v_inst_17_, v_npow_18_, v_x_19_, v_x_20_);
lean_dec(v_x_19_);
return v_res_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zpowRec(lean_object* v_G_22_, lean_object* v_inst_23_, lean_object* v_inst_24_, lean_object* v_inst_25_, lean_object* v_npow_26_, lean_object* v_x_27_, lean_object* v_x_28_){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lp_mathlib_zpowRec___redArg(v_inst_25_, v_npow_26_, v_x_27_, v_x_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zpowRec___boxed(lean_object* v_G_30_, lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_npow_34_, lean_object* v_x_35_, lean_object* v_x_36_){
_start:
{
lean_object* v_res_37_; 
v_res_37_ = lp_mathlib_zpowRec(v_G_30_, v_inst_31_, v_inst_32_, v_inst_33_, v_npow_34_, v_x_35_, v_x_36_);
lean_dec(v_x_35_);
lean_dec(v_inst_32_);
lean_dec(v_inst_31_);
return v_res_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zsmulRec___redArg(lean_object* v_inst_38_, lean_object* v_nsmul_39_, lean_object* v_x_40_, lean_object* v_x_41_){
_start:
{
lean_object* v_intZero_42_; uint8_t v_isNeg_43_; 
v_intZero_42_ = lean_obj_once(&lp_mathlib_zpowRec___redArg___closed__0, &lp_mathlib_zpowRec___redArg___closed__0_once, _init_lp_mathlib_zpowRec___redArg___closed__0);
v_isNeg_43_ = lean_int_dec_lt(v_x_40_, v_intZero_42_);
if (v_isNeg_43_ == 0)
{
lean_object* v_a_44_; lean_object* v___x_45_; 
lean_dec(v_inst_38_);
v_a_44_ = lean_nat_abs(v_x_40_);
v___x_45_ = lean_apply_2(v_nsmul_39_, v_a_44_, v_x_41_);
return v___x_45_;
}
else
{
lean_object* v_abs_46_; lean_object* v_one_47_; lean_object* v_a_48_; lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; 
v_abs_46_ = lean_nat_abs(v_x_40_);
v_one_47_ = lean_unsigned_to_nat(1u);
v_a_48_ = lean_nat_sub(v_abs_46_, v_one_47_);
lean_dec(v_abs_46_);
v___x_49_ = lean_nat_add(v_a_48_, v_one_47_);
lean_dec(v_a_48_);
v___x_50_ = lean_apply_2(v_nsmul_39_, v___x_49_, v_x_41_);
v___x_51_ = lean_apply_1(v_inst_38_, v___x_50_);
return v___x_51_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_zsmulRec___redArg___boxed(lean_object* v_inst_52_, lean_object* v_nsmul_53_, lean_object* v_x_54_, lean_object* v_x_55_){
_start:
{
lean_object* v_res_56_; 
v_res_56_ = lp_mathlib_zsmulRec___redArg(v_inst_52_, v_nsmul_53_, v_x_54_, v_x_55_);
lean_dec(v_x_54_);
return v_res_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zsmulRec(lean_object* v_G_57_, lean_object* v_inst_58_, lean_object* v_inst_59_, lean_object* v_inst_60_, lean_object* v_nsmul_61_, lean_object* v_x_62_, lean_object* v_x_63_){
_start:
{
lean_object* v___x_64_; 
v___x_64_ = lp_mathlib_zsmulRec___redArg(v_inst_60_, v_nsmul_61_, v_x_62_, v_x_63_);
return v___x_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zsmulRec___boxed(lean_object* v_G_65_, lean_object* v_inst_66_, lean_object* v_inst_67_, lean_object* v_inst_68_, lean_object* v_nsmul_69_, lean_object* v_x_70_, lean_object* v_x_71_){
_start:
{
lean_object* v_res_72_; 
v_res_72_ = lp_mathlib_zsmulRec(v_G_65_, v_inst_66_, v_inst_67_, v_inst_68_, v_nsmul_69_, v_x_70_, v_x_71_);
lean_dec(v_x_70_);
lean_dec(v_inst_67_);
lean_dec(v_inst_66_);
return v_res_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivInvMonoid_div_x27___redArg(lean_object* v_inst_73_, lean_object* v_inst_74_, lean_object* v_a_75_, lean_object* v_b_76_){
_start:
{
lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v_toMul_79_; lean_object* v___x_80_; lean_object* v___x_81_; 
v___x_77_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_73_);
v___x_78_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_77_);
v_toMul_79_ = lean_ctor_get(v___x_78_, 1);
lean_inc(v_toMul_79_);
lean_dec_ref(v___x_78_);
v___x_80_ = lean_apply_1(v_inst_74_, v_b_76_);
v___x_81_ = lean_apply_2(v_toMul_79_, v_a_75_, v___x_80_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivInvMonoid_div_x27___redArg___boxed(lean_object* v_inst_82_, lean_object* v_inst_83_, lean_object* v_a_84_, lean_object* v_b_85_){
_start:
{
lean_object* v_res_86_; 
v_res_86_ = lp_mathlib_DivInvMonoid_div_x27___redArg(v_inst_82_, v_inst_83_, v_a_84_, v_b_85_);
lean_dec_ref(v_inst_82_);
return v_res_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivInvMonoid_div_x27(lean_object* v_G_87_, lean_object* v_inst_88_, lean_object* v_inst_89_, lean_object* v_a_90_, lean_object* v_b_91_){
_start:
{
lean_object* v___x_92_; 
v___x_92_ = lp_mathlib_DivInvMonoid_div_x27___redArg(v_inst_88_, v_inst_89_, v_a_90_, v_b_91_);
return v___x_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivInvMonoid_div_x27___boxed(lean_object* v_G_93_, lean_object* v_inst_94_, lean_object* v_inst_95_, lean_object* v_a_96_, lean_object* v_b_97_){
_start:
{
lean_object* v_res_98_; 
v_res_98_ = lp_mathlib_DivInvMonoid_div_x27(v_G_93_, v_inst_94_, v_inst_95_, v_a_96_, v_b_97_);
lean_dec_ref(v_inst_94_);
return v_res_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZPow_toPow___redArg___lam__0(lean_object* v_inst_99_, lean_object* v_x_100_, lean_object* v_n_101_){
_start:
{
lean_object* v___x_102_; 
v___x_102_ = lean_apply_2(v_inst_99_, v_n_101_, v_x_100_);
return v___x_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZPow_toPow___redArg(lean_object* v_inst_103_){
_start:
{
lean_object* v___f_104_; 
v___f_104_ = lean_alloc_closure((void*)(lp_mathlib_ZPow_toPow___redArg___lam__0), 3, 1);
lean_closure_set(v___f_104_, 0, v_inst_103_);
return v___f_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZPow_toPow(lean_object* v_M_105_, lean_object* v_inst_106_){
_start:
{
lean_object* v___f_107_; 
v___f_107_ = lean_alloc_closure((void*)(lp_mathlib_ZPow_toPow___redArg___lam__0), 3, 1);
lean_closure_set(v___f_107_, 0, v_inst_106_);
return v___f_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZSMul_toSMul___redArg___lam__0(lean_object* v_inst_108_, lean_object* v_n_109_, lean_object* v_x_110_){
_start:
{
lean_object* v___x_111_; 
v___x_111_ = lean_apply_2(v_inst_108_, v_n_109_, v_x_110_);
return v___x_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZSMul_toSMul___redArg(lean_object* v_inst_112_){
_start:
{
lean_object* v___f_113_; 
v___f_113_ = lean_alloc_closure((void*)(lp_mathlib_ZSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_113_, 0, v_inst_112_);
return v___f_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZSMul_toSMul(lean_object* v_M_114_, lean_object* v_inst_115_){
_start:
{
lean_object* v___f_116_; 
v___f_116_ = lean_alloc_closure((void*)(lp_mathlib_ZSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_116_, 0, v_inst_115_);
return v___f_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZPow_ofPow___redArg___lam__0(lean_object* v_inst_117_, lean_object* v_n_118_, lean_object* v_x_119_){
_start:
{
lean_object* v___x_120_; 
v___x_120_ = lean_apply_2(v_inst_117_, v_x_119_, v_n_118_);
return v___x_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZPow_ofPow___redArg(lean_object* v_inst_121_){
_start:
{
lean_object* v___f_122_; 
v___f_122_ = lean_alloc_closure((void*)(lp_mathlib_ZPow_ofPow___redArg___lam__0), 3, 1);
lean_closure_set(v___f_122_, 0, v_inst_121_);
return v___f_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZPow_ofPow(lean_object* v_M_123_, lean_object* v_inst_124_){
_start:
{
lean_object* v___f_125_; 
v___f_125_ = lean_alloc_closure((void*)(lp_mathlib_ZPow_ofPow___redArg___lam__0), 3, 1);
lean_closure_set(v___f_125_, 0, v_inst_124_);
return v___f_125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZSMul_ofSMul___redArg(lean_object* v_inst_126_){
_start:
{
lean_object* v___f_127_; 
v___f_127_ = lean_alloc_closure((void*)(lp_mathlib_ZSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_127_, 0, v_inst_126_);
return v___f_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZSMul_ofSMul(lean_object* v_M_128_, lean_object* v_inst_129_){
_start:
{
lean_object* v___f_130_; 
v___f_130_ = lean_alloc_closure((void*)(lp_mathlib_ZSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_130_, 0, v_inst_129_);
return v___f_130_;
}
}
static lean_object* _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__12(void){
_start:
{
lean_object* v___x_157_; lean_object* v___x_158_; 
v___x_157_ = ((lean_object*)(lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__10));
v___x_158_ = l_Lean_mkAtom(v___x_157_);
return v___x_158_;
}
}
static lean_object* _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__13(void){
_start:
{
lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; 
v___x_159_ = lean_obj_once(&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__12, &lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__12_once, _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__12);
v___x_160_ = ((lean_object*)(lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__5));
v___x_161_ = lean_array_push(v___x_160_, v___x_159_);
return v___x_161_;
}
}
static lean_object* _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__15(void){
_start:
{
lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; 
v___x_166_ = ((lean_object*)(lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__14));
v___x_167_ = lean_obj_once(&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__13, &lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__13_once, _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__13);
v___x_168_ = lean_array_push(v___x_167_, v___x_166_);
return v___x_168_;
}
}
static lean_object* _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__16(void){
_start:
{
lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; 
v___x_169_ = lean_obj_once(&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__15, &lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__15_once, _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__15);
v___x_170_ = ((lean_object*)(lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__11));
v___x_171_ = lean_box(2);
v___x_172_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_172_, 0, v___x_171_);
lean_ctor_set(v___x_172_, 1, v___x_170_);
lean_ctor_set(v___x_172_, 2, v___x_169_);
return v___x_172_;
}
}
static lean_object* _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__17(void){
_start:
{
lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; 
v___x_173_ = lean_obj_once(&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__16, &lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__16_once, _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__16);
v___x_174_ = ((lean_object*)(lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__5));
v___x_175_ = lean_array_push(v___x_174_, v___x_173_);
return v___x_175_;
}
}
static lean_object* _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__19(void){
_start:
{
lean_object* v___x_177_; lean_object* v___x_178_; 
v___x_177_ = ((lean_object*)(lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__18));
v___x_178_ = l_Lean_mkAtom(v___x_177_);
return v___x_178_;
}
}
static lean_object* _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__20(void){
_start:
{
lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; 
v___x_179_ = lean_obj_once(&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__19, &lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__19_once, _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__19);
v___x_180_ = lean_obj_once(&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__17, &lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__17_once, _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__17);
v___x_181_ = lean_array_push(v___x_180_, v___x_179_);
return v___x_181_;
}
}
static lean_object* _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__24(void){
_start:
{
lean_object* v___x_189_; lean_object* v___x_190_; 
v___x_189_ = ((lean_object*)(lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__23));
v___x_190_ = l_Lean_mkAtom(v___x_189_);
return v___x_190_;
}
}
static lean_object* _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__25(void){
_start:
{
lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; 
v___x_191_ = lean_obj_once(&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__24, &lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__24_once, _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__24);
v___x_192_ = ((lean_object*)(lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__5));
v___x_193_ = lean_array_push(v___x_192_, v___x_191_);
return v___x_193_;
}
}
static lean_object* _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__26(void){
_start:
{
lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; 
v___x_194_ = lean_obj_once(&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__25, &lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__25_once, _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__25);
v___x_195_ = ((lean_object*)(lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__22));
v___x_196_ = lean_box(2);
v___x_197_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_197_, 0, v___x_196_);
lean_ctor_set(v___x_197_, 1, v___x_195_);
lean_ctor_set(v___x_197_, 2, v___x_194_);
return v___x_197_;
}
}
static lean_object* _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__27(void){
_start:
{
lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; 
v___x_198_ = lean_obj_once(&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__26, &lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__26_once, _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__26);
v___x_199_ = lean_obj_once(&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__20, &lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__20_once, _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__20);
v___x_200_ = lean_array_push(v___x_199_, v___x_198_);
return v___x_200_;
}
}
static lean_object* _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__28(void){
_start:
{
lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; 
v___x_201_ = lean_obj_once(&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__27, &lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__27_once, _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__27);
v___x_202_ = ((lean_object*)(lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__9));
v___x_203_ = lean_box(2);
v___x_204_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_204_, 0, v___x_203_);
lean_ctor_set(v___x_204_, 1, v___x_202_);
lean_ctor_set(v___x_204_, 2, v___x_201_);
return v___x_204_;
}
}
static lean_object* _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__29(void){
_start:
{
lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; 
v___x_205_ = lean_obj_once(&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__28, &lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__28_once, _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__28);
v___x_206_ = ((lean_object*)(lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__5));
v___x_207_ = lean_array_push(v___x_206_, v___x_205_);
return v___x_207_;
}
}
static lean_object* _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__30(void){
_start:
{
lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; 
v___x_208_ = lean_obj_once(&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__29, &lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__29_once, _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__29);
v___x_209_ = ((lean_object*)(lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__7));
v___x_210_ = lean_box(2);
v___x_211_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_211_, 0, v___x_210_);
lean_ctor_set(v___x_211_, 1, v___x_209_);
lean_ctor_set(v___x_211_, 2, v___x_208_);
return v___x_211_;
}
}
static lean_object* _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__31(void){
_start:
{
lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; 
v___x_212_ = lean_obj_once(&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__30, &lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__30_once, _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__30);
v___x_213_ = ((lean_object*)(lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__5));
v___x_214_ = lean_array_push(v___x_213_, v___x_212_);
return v___x_214_;
}
}
static lean_object* _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__32(void){
_start:
{
lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; 
v___x_215_ = lean_obj_once(&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__31, &lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__31_once, _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__31);
v___x_216_ = ((lean_object*)(lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__4));
v___x_217_ = lean_box(2);
v___x_218_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_218_, 0, v___x_217_);
lean_ctor_set(v___x_218_, 1, v___x_216_);
lean_ctor_set(v___x_218_, 2, v___x_215_);
return v___x_218_;
}
}
static lean_object* _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam(void){
_start:
{
lean_object* v___x_219_; 
v___x_219_ = lean_obj_once(&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__32, &lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__32_once, _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__32);
return v___x_219_;
}
}
static lean_object* _init_lp_mathlib_DivInvMonoid_zpow__zero_x27___autoParam(void){
_start:
{
lean_object* v___x_220_; 
v___x_220_ = lean_obj_once(&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__32, &lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__32_once, _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__32);
return v___x_220_;
}
}
static lean_object* _init_lp_mathlib_DivInvMonoid_zpow__succ_x27___autoParam(void){
_start:
{
lean_object* v___x_221_; 
v___x_221_ = lean_obj_once(&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__32, &lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__32_once, _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__32);
return v___x_221_;
}
}
static lean_object* _init_lp_mathlib_DivInvMonoid_zpow__neg_x27___autoParam(void){
_start:
{
lean_object* v___x_222_; 
v___x_222_ = lean_obj_once(&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__32, &lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__32_once, _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__32);
return v___x_222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubNegMonoid_sub_x27___redArg(lean_object* v_inst_223_, lean_object* v_inst_224_, lean_object* v_a_225_, lean_object* v_b_226_){
_start:
{
lean_object* v_toAdd_227_; lean_object* v___x_228_; lean_object* v___x_229_; 
v_toAdd_227_ = lean_ctor_get(v_inst_223_, 1);
lean_inc(v_toAdd_227_);
lean_dec_ref(v_inst_223_);
v___x_228_ = lean_apply_1(v_inst_224_, v_b_226_);
v___x_229_ = lean_apply_2(v_toAdd_227_, v_a_225_, v___x_228_);
return v___x_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubNegMonoid_sub_x27(lean_object* v_G_230_, lean_object* v_inst_231_, lean_object* v_inst_232_, lean_object* v_a_233_, lean_object* v_b_234_){
_start:
{
lean_object* v___x_235_; 
v___x_235_ = lp_mathlib_SubNegMonoid_sub_x27___redArg(v_inst_231_, v_inst_232_, v_a_233_, v_b_234_);
return v___x_235_;
}
}
static lean_object* _init_lp_mathlib_SubNegMonoid_sub__eq__add__neg___autoParam(void){
_start:
{
lean_object* v___x_236_; 
v___x_236_ = lean_obj_once(&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__32, &lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__32_once, _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__32);
return v___x_236_;
}
}
static lean_object* _init_lp_mathlib_SubNegMonoid_zsmul__zero_x27___autoParam(void){
_start:
{
lean_object* v___x_237_; 
v___x_237_ = lean_obj_once(&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__32, &lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__32_once, _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__32);
return v___x_237_;
}
}
static lean_object* _init_lp_mathlib_SubNegMonoid_zsmul__succ_x27___autoParam(void){
_start:
{
lean_object* v___x_238_; 
v___x_238_ = lean_obj_once(&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__32, &lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__32_once, _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__32);
return v___x_238_;
}
}
static lean_object* _init_lp_mathlib_SubNegMonoid_zsmul__neg_x27___autoParam(void){
_start:
{
lean_object* v___x_239_; 
v___x_239_ = lean_obj_once(&lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__32, &lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__32_once, _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam___closed__32);
return v___x_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object* v_self_240_){
_start:
{
lean_object* v_toAddMonoid_241_; lean_object* v_toNeg_242_; lean_object* v_toZero_243_; lean_object* v___x_244_; 
v_toAddMonoid_241_ = lean_ctor_get(v_self_240_, 0);
v_toNeg_242_ = lean_ctor_get(v_self_240_, 1);
v_toZero_243_ = lean_ctor_get(v_toAddMonoid_241_, 0);
lean_inc(v_toNeg_242_);
lean_inc(v_toZero_243_);
v___x_244_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_244_, 0, v_toZero_243_);
lean_ctor_set(v___x_244_, 1, v_toNeg_242_);
return v___x_244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg___boxed(lean_object* v_self_245_){
_start:
{
lean_object* v_res_246_; 
v_res_246_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_self_245_);
lean_dec_ref(v_self_245_);
return v_res_246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass(lean_object* v_G_247_, lean_object* v_self_248_){
_start:
{
lean_object* v___x_249_; 
v___x_249_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_self_248_);
return v___x_249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___boxed(lean_object* v_G_250_, lean_object* v_self_251_){
_start:
{
lean_object* v_res_252_; 
v_res_252_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass(v_G_250_, v_self_251_);
lean_dec_ref(v_self_251_);
return v_res_252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(lean_object* v_self_253_){
_start:
{
lean_object* v_toMonoid_254_; lean_object* v_toInv_255_; lean_object* v_toOne_256_; lean_object* v___x_257_; 
v_toMonoid_254_ = lean_ctor_get(v_self_253_, 0);
v_toInv_255_ = lean_ctor_get(v_self_253_, 1);
v_toOne_256_ = lean_ctor_get(v_toMonoid_254_, 0);
lean_inc(v_toInv_255_);
lean_inc(v_toOne_256_);
v___x_257_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_257_, 0, v_toOne_256_);
lean_ctor_set(v___x_257_, 1, v_toInv_255_);
return v___x_257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg___boxed(lean_object* v_self_258_){
_start:
{
lean_object* v_res_259_; 
v_res_259_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v_self_258_);
lean_dec_ref(v_self_258_);
return v_res_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass(lean_object* v_G_260_, lean_object* v_self_261_){
_start:
{
lean_object* v___x_262_; 
v___x_262_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v_self_261_);
return v___x_262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___boxed(lean_object* v_G_263_, lean_object* v_self_264_){
_start:
{
lean_object* v_res_265_; 
v_res_265_ = lp_mathlib_DivInvOneMonoid_toInvOneClass(v_G_263_, v_self_264_);
lean_dec_ref(v_self_264_);
return v_res_265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubtractionMonoid_toInvolutiveNeg___redArg(lean_object* v_self_266_){
_start:
{
lean_object* v_toNeg_267_; 
v_toNeg_267_ = lean_ctor_get(v_self_266_, 1);
lean_inc(v_toNeg_267_);
return v_toNeg_267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubtractionMonoid_toInvolutiveNeg___redArg___boxed(lean_object* v_self_268_){
_start:
{
lean_object* v_res_269_; 
v_res_269_ = lp_mathlib_SubtractionMonoid_toInvolutiveNeg___redArg(v_self_268_);
lean_dec_ref(v_self_268_);
return v_res_269_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubtractionMonoid_toInvolutiveNeg(lean_object* v_G_270_, lean_object* v_self_271_){
_start:
{
lean_object* v_toNeg_272_; 
v_toNeg_272_ = lean_ctor_get(v_self_271_, 1);
lean_inc(v_toNeg_272_);
return v_toNeg_272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubtractionMonoid_toInvolutiveNeg___boxed(lean_object* v_G_273_, lean_object* v_self_274_){
_start:
{
lean_object* v_res_275_; 
v_res_275_ = lp_mathlib_SubtractionMonoid_toInvolutiveNeg(v_G_273_, v_self_274_);
lean_dec_ref(v_self_274_);
return v_res_275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionMonoid_toInvolutiveInv___redArg(lean_object* v_self_276_){
_start:
{
lean_object* v_toInv_277_; 
v_toInv_277_ = lean_ctor_get(v_self_276_, 1);
lean_inc(v_toInv_277_);
return v_toInv_277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionMonoid_toInvolutiveInv___redArg___boxed(lean_object* v_self_278_){
_start:
{
lean_object* v_res_279_; 
v_res_279_ = lp_mathlib_DivisionMonoid_toInvolutiveInv___redArg(v_self_278_);
lean_dec_ref(v_self_278_);
return v_res_279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionMonoid_toInvolutiveInv(lean_object* v_G_280_, lean_object* v_self_281_){
_start:
{
lean_object* v_toInv_282_; 
v_toInv_282_ = lean_ctor_get(v_self_281_, 1);
lean_inc(v_toInv_282_);
return v_toInv_282_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionMonoid_toInvolutiveInv___boxed(lean_object* v_G_283_, lean_object* v_self_284_){
_start:
{
lean_object* v_res_285_; 
v_res_285_ = lp_mathlib_DivisionMonoid_toInvolutiveInv(v_G_283_, v_self_284_);
lean_dec_ref(v_self_284_);
return v_res_285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubtractionCommMonoid_toAddCommMonoid___redArg(lean_object* v_self_286_){
_start:
{
lean_object* v_toAddMonoid_287_; 
v_toAddMonoid_287_ = lean_ctor_get(v_self_286_, 0);
lean_inc_ref(v_toAddMonoid_287_);
return v_toAddMonoid_287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubtractionCommMonoid_toAddCommMonoid___redArg___boxed(lean_object* v_self_288_){
_start:
{
lean_object* v_res_289_; 
v_res_289_ = lp_mathlib_SubtractionCommMonoid_toAddCommMonoid___redArg(v_self_288_);
lean_dec_ref(v_self_288_);
return v_res_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubtractionCommMonoid_toAddCommMonoid(lean_object* v_G_290_, lean_object* v_self_291_){
_start:
{
lean_object* v_toAddMonoid_292_; 
v_toAddMonoid_292_ = lean_ctor_get(v_self_291_, 0);
lean_inc_ref(v_toAddMonoid_292_);
return v_toAddMonoid_292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubtractionCommMonoid_toAddCommMonoid___boxed(lean_object* v_G_293_, lean_object* v_self_294_){
_start:
{
lean_object* v_res_295_; 
v_res_295_ = lp_mathlib_SubtractionCommMonoid_toAddCommMonoid(v_G_293_, v_self_294_);
lean_dec_ref(v_self_294_);
return v_res_295_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionCommMonoid_toCommMonoid___redArg(lean_object* v_self_296_){
_start:
{
lean_object* v_toMonoid_297_; 
v_toMonoid_297_ = lean_ctor_get(v_self_296_, 0);
lean_inc_ref(v_toMonoid_297_);
return v_toMonoid_297_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionCommMonoid_toCommMonoid___redArg___boxed(lean_object* v_self_298_){
_start:
{
lean_object* v_res_299_; 
v_res_299_ = lp_mathlib_DivisionCommMonoid_toCommMonoid___redArg(v_self_298_);
lean_dec_ref(v_self_298_);
return v_res_299_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionCommMonoid_toCommMonoid(lean_object* v_G_300_, lean_object* v_self_301_){
_start:
{
lean_object* v_toMonoid_302_; 
v_toMonoid_302_ = lean_ctor_get(v_self_301_, 0);
lean_inc_ref(v_toMonoid_302_);
return v_toMonoid_302_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionCommMonoid_toCommMonoid___boxed(lean_object* v_G_303_, lean_object* v_self_304_){
_start:
{
lean_object* v_res_305_; 
v_res_305_ = lp_mathlib_DivisionCommMonoid_toCommMonoid(v_G_303_, v_self_304_);
lean_dec_ref(v_self_304_);
return v_res_305_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Monoid(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Notation(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_OfNat(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_DivInvMonoid(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Monoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_OfNat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_DivInvMonoid(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam = _init_lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam();
lean_mark_persistent(lp_mathlib_DivInvMonoid_div__eq__mul__inv___autoParam);
lp_mathlib_DivInvMonoid_zpow__zero_x27___autoParam = _init_lp_mathlib_DivInvMonoid_zpow__zero_x27___autoParam();
lean_mark_persistent(lp_mathlib_DivInvMonoid_zpow__zero_x27___autoParam);
lp_mathlib_DivInvMonoid_zpow__succ_x27___autoParam = _init_lp_mathlib_DivInvMonoid_zpow__succ_x27___autoParam();
lean_mark_persistent(lp_mathlib_DivInvMonoid_zpow__succ_x27___autoParam);
lp_mathlib_DivInvMonoid_zpow__neg_x27___autoParam = _init_lp_mathlib_DivInvMonoid_zpow__neg_x27___autoParam();
lean_mark_persistent(lp_mathlib_DivInvMonoid_zpow__neg_x27___autoParam);
lp_mathlib_SubNegMonoid_sub__eq__add__neg___autoParam = _init_lp_mathlib_SubNegMonoid_sub__eq__add__neg___autoParam();
lean_mark_persistent(lp_mathlib_SubNegMonoid_sub__eq__add__neg___autoParam);
lp_mathlib_SubNegMonoid_zsmul__zero_x27___autoParam = _init_lp_mathlib_SubNegMonoid_zsmul__zero_x27___autoParam();
lean_mark_persistent(lp_mathlib_SubNegMonoid_zsmul__zero_x27___autoParam);
lp_mathlib_SubNegMonoid_zsmul__succ_x27___autoParam = _init_lp_mathlib_SubNegMonoid_zsmul__succ_x27___autoParam();
lean_mark_persistent(lp_mathlib_SubNegMonoid_zsmul__succ_x27___autoParam);
lp_mathlib_SubNegMonoid_zsmul__neg_x27___autoParam = _init_lp_mathlib_SubNegMonoid_zsmul__neg_x27___autoParam();
lean_mark_persistent(lp_mathlib_SubNegMonoid_zsmul__neg_x27___autoParam);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Monoid(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Notation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_OfNat(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_DivInvMonoid(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Monoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_OfNat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_DivInvMonoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_DivInvMonoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_DivInvMonoid(builtin);
}
#ifdef __cplusplus
}
#endif
