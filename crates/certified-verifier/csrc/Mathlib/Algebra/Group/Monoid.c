// Lean compiler output
// Module: Mathlib.Algebra.Group.Monoid
// Imports: public import Init public meta import Init public import Batteries.Logic public import Mathlib.Algebra.Group.Semigroup public import Mathlib.Data.Nat.BinaryRec public import Mathlib.Data.Nat.Notation public import Mathlib.Tactic.Push.Attr
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
lean_object* lp_mathlib_Nat_binaryRec___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_nsmulRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_npowRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddZeroClass_toAddZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOneClass_toMulOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_Monoid_0__npowRec_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_Monoid_0__npowRec_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_Monoid_0__npowRec_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_Monoid_0__npowRec_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LibraryNote_forgetful__inheritance;
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go___redArg___lam__1(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_npowBinRec_go___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_npowBinRec_go___redArg___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_npowBinRec_go___redArg___closed__0 = (const lean_object*)&lp_mathlib_npowBinRec_go___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec_go___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowRec_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowRec_x27___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowRec_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowRec_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulRec_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulRec_x27___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulRec_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulRec_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_Monoid_0__npowRec_x27_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_Monoid_0__npowRec_x27_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_Monoid_0__npowRec_x27_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_Monoid_0__npowRec_x27_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowRecAuto___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowRecAuto___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowRecAuto(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulRecAuto___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulRecAuto___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulRecAuto(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowBinRecAuto___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowBinRecAuto___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowBinRecAuto(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowBinRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRecAuto___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRecAuto___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRecAuto(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NPow_toPow___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NPow_toPow___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NPow_toPow(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NSMul_toSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NSMul_toSMul(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NPow_ofPow___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NPow_ofPow___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NPow_ofPow(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NSMul_ofSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NSMul_ofSMul(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__0 = (const lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__0_value;
static const lean_string_object lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__1 = (const lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__1_value;
static const lean_string_object lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__2 = (const lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__2_value;
static const lean_string_object lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__3 = (const lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__3_value;
static const lean_ctor_object lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__4_value_aux_0),((lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__4_value_aux_1),((lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__4_value_aux_2),((lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__4 = (const lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__4_value;
static const lean_array_object lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__5 = (const lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__5_value;
static const lean_string_object lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__6 = (const lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__6_value;
static const lean_ctor_object lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__7_value_aux_0),((lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__7_value_aux_1),((lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__7_value_aux_2),((lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__6_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__7 = (const lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__7_value;
static const lean_string_object lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__8 = (const lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__8_value;
static const lean_ctor_object lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__9 = (const lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__9_value;
static const lean_string_object lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "intros"};
static const lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__10 = (const lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__10_value;
static const lean_ctor_object lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__11_value_aux_0),((lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__11_value_aux_1),((lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__11_value_aux_2),((lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__10_value),LEAN_SCALAR_PTR_LITERAL(26, 175, 18, 116, 252, 50, 128, 45)}};
static const lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__11 = (const lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__11_value;
static lean_once_cell_t lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__12;
static lean_once_cell_t lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__13;
static const lean_ctor_object lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__9_value),((lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__5_value)}};
static const lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__14 = (const lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__14_value;
static lean_once_cell_t lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__15;
static lean_once_cell_t lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__16;
static lean_once_cell_t lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__17;
static const lean_string_object lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ";"};
static const lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__18 = (const lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__18_value;
static lean_once_cell_t lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__19;
static lean_once_cell_t lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__20;
static const lean_string_object lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticRfl"};
static const lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__21 = (const lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__21_value;
static const lean_ctor_object lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__22_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__22_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__22_value_aux_0),((lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__22_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__22_value_aux_1),((lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__22_value_aux_2),((lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__21_value),LEAN_SCALAR_PTR_LITERAL(201, 188, 173, 198, 169, 252, 183, 45)}};
static const lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__22 = (const lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__22_value;
static const lean_string_object lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "rfl"};
static const lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__23 = (const lean_object*)&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__23_value;
static lean_once_cell_t lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__24;
static lean_once_cell_t lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__25;
static lean_once_cell_t lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__26;
static lean_once_cell_t lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__27;
static lean_once_cell_t lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__28;
static lean_once_cell_t lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__29;
static lean_once_cell_t lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__30;
static lean_once_cell_t lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__31;
static lean_once_cell_t lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__32_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__32;
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_nsmul__zero___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_nsmul__succ___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toAddSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toAddSemigroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toAddSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toAddSemigroup___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toAddZeroClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toAddZeroClass___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_npow__zero___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_Monoid_npow__succ___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toSemigroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toSemigroup___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toMulOneClass___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toMulOneClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toMulOneClass___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_toAddCommSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_toAddCommSemigroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_toAddCommSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_toAddCommSemigroup___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommMonoid_toCommSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommMonoid_toCommSemigroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommMonoid_toCommSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommMonoid_toCommSemigroup___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLeftCancelMonoid_toAddLeftCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLeftCancelMonoid_toAddLeftCancelSemigroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLeftCancelMonoid_toAddLeftCancelSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLeftCancelMonoid_toAddLeftCancelSemigroup___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LeftCancelMonoid_toLeftCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LeftCancelMonoid_toLeftCancelSemigroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LeftCancelMonoid_toLeftCancelSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LeftCancelMonoid_toLeftCancelSemigroup___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddRightCancelMonoid_toAddRightCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddRightCancelMonoid_toAddRightCancelSemigroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddRightCancelMonoid_toAddRightCancelSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddRightCancelMonoid_toAddRightCancelSemigroup___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightCancelMonoid_toRightCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightCancelMonoid_toRightCancelSemigroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightCancelMonoid_toRightCancelSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightCancelMonoid_toRightCancelSemigroup___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCancelMonoid_toAddRightCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCancelMonoid_toAddRightCancelMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCancelMonoid_toAddRightCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCancelMonoid_toAddRightCancelMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CancelMonoid_toRightCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CancelMonoid_toRightCancelMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CancelMonoid_toRightCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CancelMonoid_toRightCancelMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCancelCommMonoid_toAddLeftCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCancelCommMonoid_toAddLeftCancelMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCancelCommMonoid_toAddLeftCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCancelCommMonoid_toAddLeftCancelMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CancelCommMonoid_toLeftCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CancelCommMonoid_toLeftCancelMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CancelCommMonoid_toLeftCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CancelCommMonoid_toLeftCancelMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CancelCommMonoid_toCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CancelCommMonoid_toCancelMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CancelCommMonoid_toCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CancelCommMonoid_toCancelMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCancelCommMonoid_toAddCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCancelCommMonoid_toAddCancelMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCancelCommMonoid_toAddCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCancelCommMonoid_toAddCancelMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object* v_self_1_){
_start:
{
lean_object* v_toZero_2_; lean_object* v_toAdd_3_; lean_object* v___x_5_; uint8_t v_isShared_6_; uint8_t v_isSharedCheck_10_; 
v_toZero_2_ = lean_ctor_get(v_self_1_, 0);
v_toAdd_3_ = lean_ctor_get(v_self_1_, 1);
v_isSharedCheck_10_ = !lean_is_exclusive(v_self_1_);
if (v_isSharedCheck_10_ == 0)
{
v___x_5_ = v_self_1_;
v_isShared_6_ = v_isSharedCheck_10_;
goto v_resetjp_4_;
}
else
{
lean_inc(v_toAdd_3_);
lean_inc(v_toZero_2_);
lean_dec(v_self_1_);
v___x_5_ = lean_box(0);
v_isShared_6_ = v_isSharedCheck_10_;
goto v_resetjp_4_;
}
v_resetjp_4_:
{
lean_object* v___x_8_; 
if (v_isShared_6_ == 0)
{
v___x_8_ = v___x_5_;
goto v_reusejp_7_;
}
else
{
lean_object* v_reuseFailAlloc_9_; 
v_reuseFailAlloc_9_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_9_, 0, v_toZero_2_);
lean_ctor_set(v_reuseFailAlloc_9_, 1, v_toAdd_3_);
v___x_8_ = v_reuseFailAlloc_9_;
goto v_reusejp_7_;
}
v_reusejp_7_:
{
return v___x_8_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddZeroClass_toAddZero(lean_object* v_M_11_, lean_object* v_self_12_){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_self_12_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object* v_self_14_){
_start:
{
lean_object* v_toOne_15_; lean_object* v_toMul_16_; lean_object* v___x_18_; uint8_t v_isShared_19_; uint8_t v_isSharedCheck_23_; 
v_toOne_15_ = lean_ctor_get(v_self_14_, 0);
v_toMul_16_ = lean_ctor_get(v_self_14_, 1);
v_isSharedCheck_23_ = !lean_is_exclusive(v_self_14_);
if (v_isSharedCheck_23_ == 0)
{
v___x_18_ = v_self_14_;
v_isShared_19_ = v_isSharedCheck_23_;
goto v_resetjp_17_;
}
else
{
lean_inc(v_toMul_16_);
lean_inc(v_toOne_15_);
lean_dec(v_self_14_);
v___x_18_ = lean_box(0);
v_isShared_19_ = v_isSharedCheck_23_;
goto v_resetjp_17_;
}
v_resetjp_17_:
{
lean_object* v___x_21_; 
if (v_isShared_19_ == 0)
{
v___x_21_ = v___x_18_;
goto v_reusejp_20_;
}
else
{
lean_object* v_reuseFailAlloc_22_; 
v_reuseFailAlloc_22_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_22_, 0, v_toOne_15_);
lean_ctor_set(v_reuseFailAlloc_22_, 1, v_toMul_16_);
v___x_21_ = v_reuseFailAlloc_22_;
goto v_reusejp_20_;
}
v_reusejp_20_:
{
return v___x_21_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOneClass_toMulOne(lean_object* v_M_24_, lean_object* v_self_25_){
_start:
{
lean_object* v___x_26_; 
v___x_26_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_self_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_Monoid_0__npowRec_match__1_splitter___redArg(lean_object* v_x_27_, lean_object* v_x_28_, lean_object* v_h__1_29_, lean_object* v_h__2_30_){
_start:
{
lean_object* v_zero_31_; uint8_t v_isZero_32_; 
v_zero_31_ = lean_unsigned_to_nat(0u);
v_isZero_32_ = lean_nat_dec_eq(v_x_27_, v_zero_31_);
if (v_isZero_32_ == 1)
{
lean_object* v___x_33_; 
lean_dec(v_h__2_30_);
v___x_33_ = lean_apply_1(v_h__1_29_, v_x_28_);
return v___x_33_;
}
else
{
lean_object* v_one_34_; lean_object* v_n_35_; lean_object* v___x_36_; 
lean_dec(v_h__1_29_);
v_one_34_ = lean_unsigned_to_nat(1u);
v_n_35_ = lean_nat_sub(v_x_27_, v_one_34_);
v___x_36_ = lean_apply_2(v_h__2_30_, v_n_35_, v_x_28_);
return v___x_36_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_Monoid_0__npowRec_match__1_splitter___redArg___boxed(lean_object* v_x_37_, lean_object* v_x_38_, lean_object* v_h__1_39_, lean_object* v_h__2_40_){
_start:
{
lean_object* v_res_41_; 
v_res_41_ = lp_mathlib___private_Mathlib_Algebra_Group_Monoid_0__npowRec_match__1_splitter___redArg(v_x_37_, v_x_38_, v_h__1_39_, v_h__2_40_);
lean_dec(v_x_37_);
return v_res_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_Monoid_0__npowRec_match__1_splitter(lean_object* v_M_42_, lean_object* v_motive_43_, lean_object* v_x_44_, lean_object* v_x_45_, lean_object* v_h__1_46_, lean_object* v_h__2_47_){
_start:
{
lean_object* v_zero_48_; uint8_t v_isZero_49_; 
v_zero_48_ = lean_unsigned_to_nat(0u);
v_isZero_49_ = lean_nat_dec_eq(v_x_44_, v_zero_48_);
if (v_isZero_49_ == 1)
{
lean_object* v___x_50_; 
lean_dec(v_h__2_47_);
v___x_50_ = lean_apply_1(v_h__1_46_, v_x_45_);
return v___x_50_;
}
else
{
lean_object* v_one_51_; lean_object* v_n_52_; lean_object* v___x_53_; 
lean_dec(v_h__1_46_);
v_one_51_ = lean_unsigned_to_nat(1u);
v_n_52_ = lean_nat_sub(v_x_44_, v_one_51_);
v___x_53_ = lean_apply_2(v_h__2_47_, v_n_52_, v_x_45_);
return v___x_53_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_Monoid_0__npowRec_match__1_splitter___boxed(lean_object* v_M_54_, lean_object* v_motive_55_, lean_object* v_x_56_, lean_object* v_x_57_, lean_object* v_h__1_58_, lean_object* v_h__2_59_){
_start:
{
lean_object* v_res_60_; 
v_res_60_ = lp_mathlib___private_Mathlib_Algebra_Group_Monoid_0__npowRec_match__1_splitter(v_M_54_, v_motive_55_, v_x_56_, v_x_57_, v_h__1_58_, v_h__2_59_);
lean_dec(v_x_56_);
return v_res_60_;
}
}
static lean_object* _init_lp_mathlib_LibraryNote_forgetful__inheritance(void){
_start:
{
lean_object* v___x_61_; 
v___x_61_ = lean_box(0);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go___redArg___lam__0(lean_object* v_y_62_, lean_object* v_x_63_){
_start:
{
lean_inc(v_y_62_);
return v_y_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go___redArg___lam__0___boxed(lean_object* v_y_64_, lean_object* v_x_65_){
_start:
{
lean_object* v_res_66_; 
v_res_66_ = lp_mathlib_npowBinRec_go___redArg___lam__0(v_y_64_, v_x_65_);
lean_dec(v_x_65_);
lean_dec(v_y_64_);
return v_res_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go___redArg___lam__1(lean_object* v_inst_67_, uint8_t v_bn_68_, lean_object* v___n_69_, lean_object* v_fn_70_, lean_object* v_y_71_, lean_object* v_x_72_){
_start:
{
lean_object* v___y_74_; 
if (v_bn_68_ == 0)
{
v___y_74_ = v_y_71_;
goto v___jp_73_;
}
else
{
lean_object* v___x_77_; 
lean_inc(v_inst_67_);
lean_inc(v_x_72_);
v___x_77_ = lean_apply_2(v_inst_67_, v_y_71_, v_x_72_);
v___y_74_ = v___x_77_;
goto v___jp_73_;
}
v___jp_73_:
{
lean_object* v___x_75_; lean_object* v___x_76_; 
lean_inc(v_x_72_);
v___x_75_ = lean_apply_2(v_inst_67_, v_x_72_, v_x_72_);
v___x_76_ = lean_apply_2(v_fn_70_, v___y_74_, v___x_75_);
return v___x_76_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go___redArg___lam__1___boxed(lean_object* v_inst_78_, lean_object* v_bn_79_, lean_object* v___n_80_, lean_object* v_fn_81_, lean_object* v_y_82_, lean_object* v_x_83_){
_start:
{
uint8_t v_bn_boxed_84_; lean_object* v_res_85_; 
v_bn_boxed_84_ = lean_unbox(v_bn_79_);
v_res_85_ = lp_mathlib_npowBinRec_go___redArg___lam__1(v_inst_78_, v_bn_boxed_84_, v___n_80_, v_fn_81_, v_y_82_, v_x_83_);
lean_dec(v___n_80_);
return v_res_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go___redArg(lean_object* v_inst_87_, lean_object* v_k_88_, lean_object* v_a_89_, lean_object* v_a_90_){
_start:
{
lean_object* v___f_91_; lean_object* v___f_92_; lean_object* v___x_30__overap_93_; lean_object* v___x_94_; 
v___f_91_ = ((lean_object*)(lp_mathlib_npowBinRec_go___redArg___closed__0));
v___f_92_ = lean_alloc_closure((void*)(lp_mathlib_npowBinRec_go___redArg___lam__1___boxed), 6, 1);
lean_closure_set(v___f_92_, 0, v_inst_87_);
v___x_30__overap_93_ = lp_mathlib_Nat_binaryRec___redArg(v___f_91_, v___f_92_, v_k_88_);
v___x_94_ = lean_apply_2(v___x_30__overap_93_, v_a_89_, v_a_90_);
return v___x_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go___redArg___boxed(lean_object* v_inst_95_, lean_object* v_k_96_, lean_object* v_a_97_, lean_object* v_a_98_){
_start:
{
lean_object* v_res_99_; 
v_res_99_ = lp_mathlib_npowBinRec_go___redArg(v_inst_95_, v_k_96_, v_a_97_, v_a_98_);
lean_dec(v_k_96_);
return v_res_99_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go(lean_object* v_M_100_, lean_object* v_inst_101_, lean_object* v_k_102_, lean_object* v_a_103_, lean_object* v_a_104_){
_start:
{
lean_object* v___x_105_; 
v___x_105_ = lp_mathlib_npowBinRec_go___redArg(v_inst_101_, v_k_102_, v_a_103_, v_a_104_);
return v___x_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go___boxed(lean_object* v_M_106_, lean_object* v_inst_107_, lean_object* v_k_108_, lean_object* v_a_109_, lean_object* v_a_110_){
_start:
{
lean_object* v_res_111_; 
v_res_111_ = lp_mathlib_npowBinRec_go(v_M_106_, v_inst_107_, v_k_108_, v_a_109_, v_a_110_);
lean_dec(v_k_108_);
return v_res_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec_go___redArg(lean_object* v_inst_112_, lean_object* v_k_113_, lean_object* v_a_114_, lean_object* v_a_115_){
_start:
{
lean_object* v___f_116_; lean_object* v___f_117_; lean_object* v___x_30__overap_118_; lean_object* v___x_119_; 
v___f_116_ = ((lean_object*)(lp_mathlib_npowBinRec_go___redArg___closed__0));
v___f_117_ = lean_alloc_closure((void*)(lp_mathlib_npowBinRec_go___redArg___lam__1___boxed), 6, 1);
lean_closure_set(v___f_117_, 0, v_inst_112_);
v___x_30__overap_118_ = lp_mathlib_Nat_binaryRec___redArg(v___f_116_, v___f_117_, v_k_113_);
v___x_119_ = lean_apply_2(v___x_30__overap_118_, v_a_114_, v_a_115_);
return v___x_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec_go___redArg___boxed(lean_object* v_inst_120_, lean_object* v_k_121_, lean_object* v_a_122_, lean_object* v_a_123_){
_start:
{
lean_object* v_res_124_; 
v_res_124_ = lp_mathlib_nsmulBinRec_go___redArg(v_inst_120_, v_k_121_, v_a_122_, v_a_123_);
lean_dec(v_k_121_);
return v_res_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec_go(lean_object* v_M_125_, lean_object* v_inst_126_, lean_object* v_k_127_, lean_object* v_a_128_, lean_object* v_a_129_){
_start:
{
lean_object* v___x_130_; 
v___x_130_ = lp_mathlib_nsmulBinRec_go___redArg(v_inst_126_, v_k_127_, v_a_128_, v_a_129_);
return v___x_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec_go___boxed(lean_object* v_M_131_, lean_object* v_inst_132_, lean_object* v_k_133_, lean_object* v_a_134_, lean_object* v_a_135_){
_start:
{
lean_object* v_res_136_; 
v_res_136_ = lp_mathlib_nsmulBinRec_go(v_M_131_, v_inst_132_, v_k_133_, v_a_134_, v_a_135_);
lean_dec(v_k_133_);
return v_res_136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec___redArg(lean_object* v_inst_137_, lean_object* v_inst_138_, lean_object* v_k_139_, lean_object* v_a_140_){
_start:
{
lean_object* v___x_141_; 
v___x_141_ = lp_mathlib_npowBinRec_go___redArg(v_inst_138_, v_k_139_, v_inst_137_, v_a_140_);
return v___x_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec___redArg___boxed(lean_object* v_inst_142_, lean_object* v_inst_143_, lean_object* v_k_144_, lean_object* v_a_145_){
_start:
{
lean_object* v_res_146_; 
v_res_146_ = lp_mathlib_npowBinRec___redArg(v_inst_142_, v_inst_143_, v_k_144_, v_a_145_);
lean_dec(v_k_144_);
return v_res_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec(lean_object* v_M_147_, lean_object* v_inst_148_, lean_object* v_inst_149_, lean_object* v_k_150_, lean_object* v_a_151_){
_start:
{
lean_object* v___x_152_; 
v___x_152_ = lp_mathlib_npowBinRec_go___redArg(v_inst_149_, v_k_150_, v_inst_148_, v_a_151_);
return v___x_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec___boxed(lean_object* v_M_153_, lean_object* v_inst_154_, lean_object* v_inst_155_, lean_object* v_k_156_, lean_object* v_a_157_){
_start:
{
lean_object* v_res_158_; 
v_res_158_ = lp_mathlib_npowBinRec(v_M_153_, v_inst_154_, v_inst_155_, v_k_156_, v_a_157_);
lean_dec(v_k_156_);
return v_res_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec___redArg(lean_object* v_inst_159_, lean_object* v_inst_160_, lean_object* v_k_161_, lean_object* v_a_162_){
_start:
{
lean_object* v___x_163_; 
v___x_163_ = lp_mathlib_nsmulBinRec_go___redArg(v_inst_160_, v_k_161_, v_inst_159_, v_a_162_);
return v___x_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec___redArg___boxed(lean_object* v_inst_164_, lean_object* v_inst_165_, lean_object* v_k_166_, lean_object* v_a_167_){
_start:
{
lean_object* v_res_168_; 
v_res_168_ = lp_mathlib_nsmulBinRec___redArg(v_inst_164_, v_inst_165_, v_k_166_, v_a_167_);
lean_dec(v_k_166_);
return v_res_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec(lean_object* v_M_169_, lean_object* v_inst_170_, lean_object* v_inst_171_, lean_object* v_k_172_, lean_object* v_a_173_){
_start:
{
lean_object* v___x_174_; 
v___x_174_ = lp_mathlib_nsmulBinRec_go___redArg(v_inst_171_, v_k_172_, v_inst_170_, v_a_173_);
return v___x_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec___boxed(lean_object* v_M_175_, lean_object* v_inst_176_, lean_object* v_inst_177_, lean_object* v_k_178_, lean_object* v_a_179_){
_start:
{
lean_object* v_res_180_; 
v_res_180_ = lp_mathlib_nsmulBinRec(v_M_175_, v_inst_176_, v_inst_177_, v_k_178_, v_a_179_);
lean_dec(v_k_178_);
return v_res_180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowRec_x27___redArg(lean_object* v_inst_181_, lean_object* v_inst_182_, lean_object* v_x_183_, lean_object* v_x_184_){
_start:
{
lean_object* v_zero_185_; uint8_t v_isZero_186_; 
v_zero_185_ = lean_unsigned_to_nat(0u);
v_isZero_186_ = lean_nat_dec_eq(v_x_183_, v_zero_185_);
if (v_isZero_186_ == 1)
{
lean_dec(v_x_184_);
lean_dec(v_inst_182_);
lean_inc(v_inst_181_);
return v_inst_181_;
}
else
{
lean_object* v_one_187_; lean_object* v_n_188_; uint8_t v_isZero_189_; 
v_one_187_ = lean_unsigned_to_nat(1u);
v_n_188_ = lean_nat_sub(v_x_183_, v_one_187_);
v_isZero_189_ = lean_nat_dec_eq(v_n_188_, v_zero_185_);
if (v_isZero_189_ == 1)
{
lean_dec(v_n_188_);
lean_dec(v_inst_182_);
return v_x_184_;
}
else
{
lean_object* v_n_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; 
v_n_190_ = lean_nat_sub(v_n_188_, v_one_187_);
lean_dec(v_n_188_);
v___x_191_ = lean_nat_add(v_n_190_, v_one_187_);
lean_dec(v_n_190_);
lean_inc(v_x_184_);
lean_inc(v_inst_182_);
v___x_192_ = lp_mathlib_npowRec_x27___redArg(v_inst_181_, v_inst_182_, v___x_191_, v_x_184_);
lean_dec(v___x_191_);
v___x_193_ = lean_apply_2(v_inst_182_, v___x_192_, v_x_184_);
return v___x_193_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowRec_x27___redArg___boxed(lean_object* v_inst_194_, lean_object* v_inst_195_, lean_object* v_x_196_, lean_object* v_x_197_){
_start:
{
lean_object* v_res_198_; 
v_res_198_ = lp_mathlib_npowRec_x27___redArg(v_inst_194_, v_inst_195_, v_x_196_, v_x_197_);
lean_dec(v_x_196_);
lean_dec(v_inst_194_);
return v_res_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowRec_x27(lean_object* v_M_199_, lean_object* v_inst_200_, lean_object* v_inst_201_, lean_object* v_x_202_, lean_object* v_x_203_){
_start:
{
lean_object* v___x_204_; 
v___x_204_ = lp_mathlib_npowRec_x27___redArg(v_inst_200_, v_inst_201_, v_x_202_, v_x_203_);
return v___x_204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowRec_x27___boxed(lean_object* v_M_205_, lean_object* v_inst_206_, lean_object* v_inst_207_, lean_object* v_x_208_, lean_object* v_x_209_){
_start:
{
lean_object* v_res_210_; 
v_res_210_ = lp_mathlib_npowRec_x27(v_M_205_, v_inst_206_, v_inst_207_, v_x_208_, v_x_209_);
lean_dec(v_x_208_);
lean_dec(v_inst_206_);
return v_res_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulRec_x27___redArg(lean_object* v_inst_211_, lean_object* v_inst_212_, lean_object* v_x_213_, lean_object* v_x_214_){
_start:
{
lean_object* v_zero_215_; uint8_t v_isZero_216_; 
v_zero_215_ = lean_unsigned_to_nat(0u);
v_isZero_216_ = lean_nat_dec_eq(v_x_213_, v_zero_215_);
if (v_isZero_216_ == 1)
{
lean_dec(v_x_214_);
lean_dec(v_inst_212_);
lean_inc(v_inst_211_);
return v_inst_211_;
}
else
{
lean_object* v_one_217_; lean_object* v_n_218_; uint8_t v_isZero_219_; 
v_one_217_ = lean_unsigned_to_nat(1u);
v_n_218_ = lean_nat_sub(v_x_213_, v_one_217_);
v_isZero_219_ = lean_nat_dec_eq(v_n_218_, v_zero_215_);
if (v_isZero_219_ == 1)
{
lean_dec(v_n_218_);
lean_dec(v_inst_212_);
return v_x_214_;
}
else
{
lean_object* v_n_220_; lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; 
v_n_220_ = lean_nat_sub(v_n_218_, v_one_217_);
lean_dec(v_n_218_);
v___x_221_ = lean_nat_add(v_n_220_, v_one_217_);
lean_dec(v_n_220_);
lean_inc(v_x_214_);
lean_inc(v_inst_212_);
v___x_222_ = lp_mathlib_nsmulRec_x27___redArg(v_inst_211_, v_inst_212_, v___x_221_, v_x_214_);
lean_dec(v___x_221_);
v___x_223_ = lean_apply_2(v_inst_212_, v___x_222_, v_x_214_);
return v___x_223_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulRec_x27___redArg___boxed(lean_object* v_inst_224_, lean_object* v_inst_225_, lean_object* v_x_226_, lean_object* v_x_227_){
_start:
{
lean_object* v_res_228_; 
v_res_228_ = lp_mathlib_nsmulRec_x27___redArg(v_inst_224_, v_inst_225_, v_x_226_, v_x_227_);
lean_dec(v_x_226_);
lean_dec(v_inst_224_);
return v_res_228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulRec_x27(lean_object* v_M_229_, lean_object* v_inst_230_, lean_object* v_inst_231_, lean_object* v_x_232_, lean_object* v_x_233_){
_start:
{
lean_object* v___x_234_; 
v___x_234_ = lp_mathlib_nsmulRec_x27___redArg(v_inst_230_, v_inst_231_, v_x_232_, v_x_233_);
return v___x_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulRec_x27___boxed(lean_object* v_M_235_, lean_object* v_inst_236_, lean_object* v_inst_237_, lean_object* v_x_238_, lean_object* v_x_239_){
_start:
{
lean_object* v_res_240_; 
v_res_240_ = lp_mathlib_nsmulRec_x27(v_M_235_, v_inst_236_, v_inst_237_, v_x_238_, v_x_239_);
lean_dec(v_x_238_);
lean_dec(v_inst_236_);
return v_res_240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_Monoid_0__npowRec_x27_match__1_splitter___redArg(lean_object* v_x_241_, lean_object* v_x_242_, lean_object* v_h__1_243_, lean_object* v_h__2_244_, lean_object* v_h__3_245_){
_start:
{
lean_object* v_zero_246_; uint8_t v_isZero_247_; 
v_zero_246_ = lean_unsigned_to_nat(0u);
v_isZero_247_ = lean_nat_dec_eq(v_x_241_, v_zero_246_);
if (v_isZero_247_ == 1)
{
lean_object* v___x_248_; 
lean_dec(v_h__3_245_);
lean_dec(v_h__2_244_);
v___x_248_ = lean_apply_1(v_h__1_243_, v_x_242_);
return v___x_248_;
}
else
{
lean_object* v_one_249_; lean_object* v_n_250_; uint8_t v_isZero_251_; 
lean_dec(v_h__1_243_);
v_one_249_ = lean_unsigned_to_nat(1u);
v_n_250_ = lean_nat_sub(v_x_241_, v_one_249_);
v_isZero_251_ = lean_nat_dec_eq(v_n_250_, v_zero_246_);
if (v_isZero_251_ == 1)
{
lean_object* v___x_252_; 
lean_dec(v_n_250_);
lean_dec(v_h__3_245_);
v___x_252_ = lean_apply_1(v_h__2_244_, v_x_242_);
return v___x_252_;
}
else
{
lean_object* v_n_253_; lean_object* v___x_254_; 
lean_dec(v_h__2_244_);
v_n_253_ = lean_nat_sub(v_n_250_, v_one_249_);
lean_dec(v_n_250_);
v___x_254_ = lean_apply_2(v_h__3_245_, v_n_253_, v_x_242_);
return v___x_254_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_Monoid_0__npowRec_x27_match__1_splitter___redArg___boxed(lean_object* v_x_255_, lean_object* v_x_256_, lean_object* v_h__1_257_, lean_object* v_h__2_258_, lean_object* v_h__3_259_){
_start:
{
lean_object* v_res_260_; 
v_res_260_ = lp_mathlib___private_Mathlib_Algebra_Group_Monoid_0__npowRec_x27_match__1_splitter___redArg(v_x_255_, v_x_256_, v_h__1_257_, v_h__2_258_, v_h__3_259_);
lean_dec(v_x_255_);
return v_res_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_Monoid_0__npowRec_x27_match__1_splitter(lean_object* v_M_261_, lean_object* v_motive_262_, lean_object* v_x_263_, lean_object* v_x_264_, lean_object* v_h__1_265_, lean_object* v_h__2_266_, lean_object* v_h__3_267_){
_start:
{
lean_object* v_zero_268_; uint8_t v_isZero_269_; 
v_zero_268_ = lean_unsigned_to_nat(0u);
v_isZero_269_ = lean_nat_dec_eq(v_x_263_, v_zero_268_);
if (v_isZero_269_ == 1)
{
lean_object* v___x_270_; 
lean_dec(v_h__3_267_);
lean_dec(v_h__2_266_);
v___x_270_ = lean_apply_1(v_h__1_265_, v_x_264_);
return v___x_270_;
}
else
{
lean_object* v_one_271_; lean_object* v_n_272_; uint8_t v_isZero_273_; 
lean_dec(v_h__1_265_);
v_one_271_ = lean_unsigned_to_nat(1u);
v_n_272_ = lean_nat_sub(v_x_263_, v_one_271_);
v_isZero_273_ = lean_nat_dec_eq(v_n_272_, v_zero_268_);
if (v_isZero_273_ == 1)
{
lean_object* v___x_274_; 
lean_dec(v_n_272_);
lean_dec(v_h__3_267_);
v___x_274_ = lean_apply_1(v_h__2_266_, v_x_264_);
return v___x_274_;
}
else
{
lean_object* v_n_275_; lean_object* v___x_276_; 
lean_dec(v_h__2_266_);
v_n_275_ = lean_nat_sub(v_n_272_, v_one_271_);
lean_dec(v_n_272_);
v___x_276_ = lean_apply_2(v_h__3_267_, v_n_275_, v_x_264_);
return v___x_276_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_Monoid_0__npowRec_x27_match__1_splitter___boxed(lean_object* v_M_277_, lean_object* v_motive_278_, lean_object* v_x_279_, lean_object* v_x_280_, lean_object* v_h__1_281_, lean_object* v_h__2_282_, lean_object* v_h__3_283_){
_start:
{
lean_object* v_res_284_; 
v_res_284_ = lp_mathlib___private_Mathlib_Algebra_Group_Monoid_0__npowRec_x27_match__1_splitter(v_M_277_, v_motive_278_, v_x_279_, v_x_280_, v_h__1_281_, v_h__2_282_, v_h__3_283_);
lean_dec(v_x_279_);
return v_res_284_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowRecAuto___redArg(lean_object* v_inst_285_, lean_object* v_inst_286_, lean_object* v_k_287_, lean_object* v_m_288_){
_start:
{
lean_object* v___x_289_; 
v___x_289_ = l_npowRec___redArg(v_inst_286_, v_inst_285_, v_k_287_, v_m_288_);
return v___x_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowRecAuto___redArg___boxed(lean_object* v_inst_290_, lean_object* v_inst_291_, lean_object* v_k_292_, lean_object* v_m_293_){
_start:
{
lean_object* v_res_294_; 
v_res_294_ = lp_mathlib_npowRecAuto___redArg(v_inst_290_, v_inst_291_, v_k_292_, v_m_293_);
lean_dec(v_k_292_);
lean_dec(v_inst_291_);
return v_res_294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowRecAuto(lean_object* v_M_295_, lean_object* v_inst_296_, lean_object* v_inst_297_, lean_object* v_k_298_, lean_object* v_m_299_){
_start:
{
lean_object* v___x_300_; 
v___x_300_ = l_npowRec___redArg(v_inst_297_, v_inst_296_, v_k_298_, v_m_299_);
return v___x_300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowRecAuto___boxed(lean_object* v_M_301_, lean_object* v_inst_302_, lean_object* v_inst_303_, lean_object* v_k_304_, lean_object* v_m_305_){
_start:
{
lean_object* v_res_306_; 
v_res_306_ = lp_mathlib_npowRecAuto(v_M_301_, v_inst_302_, v_inst_303_, v_k_304_, v_m_305_);
lean_dec(v_k_304_);
lean_dec(v_inst_303_);
return v_res_306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulRecAuto___redArg(lean_object* v_inst_307_, lean_object* v_inst_308_, lean_object* v_k_309_, lean_object* v_m_310_){
_start:
{
lean_object* v___x_311_; 
v___x_311_ = l_nsmulRec___redArg(v_inst_308_, v_inst_307_, v_k_309_, v_m_310_);
return v___x_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulRecAuto___redArg___boxed(lean_object* v_inst_312_, lean_object* v_inst_313_, lean_object* v_k_314_, lean_object* v_m_315_){
_start:
{
lean_object* v_res_316_; 
v_res_316_ = lp_mathlib_nsmulRecAuto___redArg(v_inst_312_, v_inst_313_, v_k_314_, v_m_315_);
lean_dec(v_k_314_);
lean_dec(v_inst_313_);
return v_res_316_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulRecAuto(lean_object* v_M_317_, lean_object* v_inst_318_, lean_object* v_inst_319_, lean_object* v_k_320_, lean_object* v_m_321_){
_start:
{
lean_object* v___x_322_; 
v___x_322_ = l_nsmulRec___redArg(v_inst_319_, v_inst_318_, v_k_320_, v_m_321_);
return v___x_322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulRecAuto___boxed(lean_object* v_M_323_, lean_object* v_inst_324_, lean_object* v_inst_325_, lean_object* v_k_326_, lean_object* v_m_327_){
_start:
{
lean_object* v_res_328_; 
v_res_328_ = lp_mathlib_nsmulRecAuto(v_M_323_, v_inst_324_, v_inst_325_, v_k_326_, v_m_327_);
lean_dec(v_k_326_);
lean_dec(v_inst_325_);
return v_res_328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRecAuto___redArg(lean_object* v_inst_329_, lean_object* v_inst_330_, lean_object* v_k_331_, lean_object* v_m_332_){
_start:
{
lean_object* v___x_333_; 
v___x_333_ = lp_mathlib_npowBinRec_go___redArg(v_inst_329_, v_k_331_, v_inst_330_, v_m_332_);
return v___x_333_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRecAuto___redArg___boxed(lean_object* v_inst_334_, lean_object* v_inst_335_, lean_object* v_k_336_, lean_object* v_m_337_){
_start:
{
lean_object* v_res_338_; 
v_res_338_ = lp_mathlib_npowBinRecAuto___redArg(v_inst_334_, v_inst_335_, v_k_336_, v_m_337_);
lean_dec(v_k_336_);
return v_res_338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRecAuto(lean_object* v_M_339_, lean_object* v_inst_340_, lean_object* v_inst_341_, lean_object* v_k_342_, lean_object* v_m_343_){
_start:
{
lean_object* v___x_344_; 
v___x_344_ = lp_mathlib_npowBinRec_go___redArg(v_inst_340_, v_k_342_, v_inst_341_, v_m_343_);
return v___x_344_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRecAuto___boxed(lean_object* v_M_345_, lean_object* v_inst_346_, lean_object* v_inst_347_, lean_object* v_k_348_, lean_object* v_m_349_){
_start:
{
lean_object* v_res_350_; 
v_res_350_ = lp_mathlib_npowBinRecAuto(v_M_345_, v_inst_346_, v_inst_347_, v_k_348_, v_m_349_);
lean_dec(v_k_348_);
return v_res_350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRecAuto___redArg(lean_object* v_inst_351_, lean_object* v_inst_352_, lean_object* v_k_353_, lean_object* v_m_354_){
_start:
{
lean_object* v___x_355_; 
v___x_355_ = lp_mathlib_nsmulBinRec_go___redArg(v_inst_351_, v_k_353_, v_inst_352_, v_m_354_);
return v___x_355_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRecAuto___redArg___boxed(lean_object* v_inst_356_, lean_object* v_inst_357_, lean_object* v_k_358_, lean_object* v_m_359_){
_start:
{
lean_object* v_res_360_; 
v_res_360_ = lp_mathlib_nsmulBinRecAuto___redArg(v_inst_356_, v_inst_357_, v_k_358_, v_m_359_);
lean_dec(v_k_358_);
return v_res_360_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRecAuto(lean_object* v_M_361_, lean_object* v_inst_362_, lean_object* v_inst_363_, lean_object* v_k_364_, lean_object* v_m_365_){
_start:
{
lean_object* v___x_366_; 
v___x_366_ = lp_mathlib_nsmulBinRec_go___redArg(v_inst_362_, v_k_364_, v_inst_363_, v_m_365_);
return v___x_366_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRecAuto___boxed(lean_object* v_M_367_, lean_object* v_inst_368_, lean_object* v_inst_369_, lean_object* v_k_370_, lean_object* v_m_371_){
_start:
{
lean_object* v_res_372_; 
v_res_372_ = lp_mathlib_nsmulBinRecAuto(v_M_367_, v_inst_368_, v_inst_369_, v_k_370_, v_m_371_);
lean_dec(v_k_370_);
return v_res_372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NPow_toPow___redArg___lam__0(lean_object* v_inst_373_, lean_object* v_x_374_, lean_object* v_n_375_){
_start:
{
lean_object* v___x_376_; 
v___x_376_ = lean_apply_2(v_inst_373_, v_n_375_, v_x_374_);
return v___x_376_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NPow_toPow___redArg(lean_object* v_inst_377_){
_start:
{
lean_object* v___f_378_; 
v___f_378_ = lean_alloc_closure((void*)(lp_mathlib_NPow_toPow___redArg___lam__0), 3, 1);
lean_closure_set(v___f_378_, 0, v_inst_377_);
return v___f_378_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NPow_toPow(lean_object* v_M_379_, lean_object* v_inst_380_){
_start:
{
lean_object* v___f_381_; 
v___f_381_ = lean_alloc_closure((void*)(lp_mathlib_NPow_toPow___redArg___lam__0), 3, 1);
lean_closure_set(v___f_381_, 0, v_inst_380_);
return v___f_381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NSMul_toSMul___redArg___lam__0(lean_object* v_inst_382_, lean_object* v_n_383_, lean_object* v_x_384_){
_start:
{
lean_object* v___x_385_; 
v___x_385_ = lean_apply_2(v_inst_382_, v_n_383_, v_x_384_);
return v___x_385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NSMul_toSMul___redArg(lean_object* v_inst_386_){
_start:
{
lean_object* v___f_387_; 
v___f_387_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_387_, 0, v_inst_386_);
return v___f_387_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NSMul_toSMul(lean_object* v_M_388_, lean_object* v_inst_389_){
_start:
{
lean_object* v___f_390_; 
v___f_390_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_390_, 0, v_inst_389_);
return v___f_390_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NPow_ofPow___redArg___lam__0(lean_object* v_inst_391_, lean_object* v_n_392_, lean_object* v_x_393_){
_start:
{
lean_object* v___x_394_; 
v___x_394_ = lean_apply_2(v_inst_391_, v_x_393_, v_n_392_);
return v___x_394_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NPow_ofPow___redArg(lean_object* v_inst_395_){
_start:
{
lean_object* v___f_396_; 
v___f_396_ = lean_alloc_closure((void*)(lp_mathlib_NPow_ofPow___redArg___lam__0), 3, 1);
lean_closure_set(v___f_396_, 0, v_inst_395_);
return v___f_396_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NPow_ofPow(lean_object* v_M_397_, lean_object* v_inst_398_){
_start:
{
lean_object* v___f_399_; 
v___f_399_ = lean_alloc_closure((void*)(lp_mathlib_NPow_ofPow___redArg___lam__0), 3, 1);
lean_closure_set(v___f_399_, 0, v_inst_398_);
return v___f_399_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NSMul_ofSMul___redArg(lean_object* v_inst_400_){
_start:
{
lean_object* v___f_401_; 
v___f_401_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_401_, 0, v_inst_400_);
return v___f_401_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NSMul_ofSMul(lean_object* v_M_402_, lean_object* v_inst_403_){
_start:
{
lean_object* v___f_404_; 
v___f_404_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_404_, 0, v_inst_403_);
return v___f_404_;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__12(void){
_start:
{
lean_object* v___x_431_; lean_object* v___x_432_; 
v___x_431_ = ((lean_object*)(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__10));
v___x_432_ = l_Lean_mkAtom(v___x_431_);
return v___x_432_;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__13(void){
_start:
{
lean_object* v___x_433_; lean_object* v___x_434_; lean_object* v___x_435_; 
v___x_433_ = lean_obj_once(&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__12, &lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__12_once, _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__12);
v___x_434_ = ((lean_object*)(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__5));
v___x_435_ = lean_array_push(v___x_434_, v___x_433_);
return v___x_435_;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__15(void){
_start:
{
lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; 
v___x_440_ = ((lean_object*)(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__14));
v___x_441_ = lean_obj_once(&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__13, &lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__13_once, _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__13);
v___x_442_ = lean_array_push(v___x_441_, v___x_440_);
return v___x_442_;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__16(void){
_start:
{
lean_object* v___x_443_; lean_object* v___x_444_; lean_object* v___x_445_; lean_object* v___x_446_; 
v___x_443_ = lean_obj_once(&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__15, &lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__15_once, _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__15);
v___x_444_ = ((lean_object*)(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__11));
v___x_445_ = lean_box(2);
v___x_446_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_446_, 0, v___x_445_);
lean_ctor_set(v___x_446_, 1, v___x_444_);
lean_ctor_set(v___x_446_, 2, v___x_443_);
return v___x_446_;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__17(void){
_start:
{
lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; 
v___x_447_ = lean_obj_once(&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__16, &lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__16_once, _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__16);
v___x_448_ = ((lean_object*)(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__5));
v___x_449_ = lean_array_push(v___x_448_, v___x_447_);
return v___x_449_;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__19(void){
_start:
{
lean_object* v___x_451_; lean_object* v___x_452_; 
v___x_451_ = ((lean_object*)(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__18));
v___x_452_ = l_Lean_mkAtom(v___x_451_);
return v___x_452_;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__20(void){
_start:
{
lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; 
v___x_453_ = lean_obj_once(&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__19, &lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__19_once, _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__19);
v___x_454_ = lean_obj_once(&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__17, &lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__17_once, _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__17);
v___x_455_ = lean_array_push(v___x_454_, v___x_453_);
return v___x_455_;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__24(void){
_start:
{
lean_object* v___x_463_; lean_object* v___x_464_; 
v___x_463_ = ((lean_object*)(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__23));
v___x_464_ = l_Lean_mkAtom(v___x_463_);
return v___x_464_;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__25(void){
_start:
{
lean_object* v___x_465_; lean_object* v___x_466_; lean_object* v___x_467_; 
v___x_465_ = lean_obj_once(&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__24, &lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__24_once, _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__24);
v___x_466_ = ((lean_object*)(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__5));
v___x_467_ = lean_array_push(v___x_466_, v___x_465_);
return v___x_467_;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__26(void){
_start:
{
lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; 
v___x_468_ = lean_obj_once(&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__25, &lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__25_once, _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__25);
v___x_469_ = ((lean_object*)(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__22));
v___x_470_ = lean_box(2);
v___x_471_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_471_, 0, v___x_470_);
lean_ctor_set(v___x_471_, 1, v___x_469_);
lean_ctor_set(v___x_471_, 2, v___x_468_);
return v___x_471_;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__27(void){
_start:
{
lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; 
v___x_472_ = lean_obj_once(&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__26, &lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__26_once, _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__26);
v___x_473_ = lean_obj_once(&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__20, &lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__20_once, _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__20);
v___x_474_ = lean_array_push(v___x_473_, v___x_472_);
return v___x_474_;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__28(void){
_start:
{
lean_object* v___x_475_; lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; 
v___x_475_ = lean_obj_once(&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__27, &lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__27_once, _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__27);
v___x_476_ = ((lean_object*)(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__9));
v___x_477_ = lean_box(2);
v___x_478_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_478_, 0, v___x_477_);
lean_ctor_set(v___x_478_, 1, v___x_476_);
lean_ctor_set(v___x_478_, 2, v___x_475_);
return v___x_478_;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__29(void){
_start:
{
lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; 
v___x_479_ = lean_obj_once(&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__28, &lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__28_once, _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__28);
v___x_480_ = ((lean_object*)(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__5));
v___x_481_ = lean_array_push(v___x_480_, v___x_479_);
return v___x_481_;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__30(void){
_start:
{
lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; 
v___x_482_ = lean_obj_once(&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__29, &lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__29_once, _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__29);
v___x_483_ = ((lean_object*)(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__7));
v___x_484_ = lean_box(2);
v___x_485_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_485_, 0, v___x_484_);
lean_ctor_set(v___x_485_, 1, v___x_483_);
lean_ctor_set(v___x_485_, 2, v___x_482_);
return v___x_485_;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__31(void){
_start:
{
lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; 
v___x_486_ = lean_obj_once(&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__30, &lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__30_once, _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__30);
v___x_487_ = ((lean_object*)(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__5));
v___x_488_ = lean_array_push(v___x_487_, v___x_486_);
return v___x_488_;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__32(void){
_start:
{
lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; 
v___x_489_ = lean_obj_once(&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__31, &lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__31_once, _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__31);
v___x_490_ = ((lean_object*)(lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__4));
v___x_491_ = lean_box(2);
v___x_492_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_492_, 0, v___x_491_);
lean_ctor_set(v___x_492_, 1, v___x_490_);
lean_ctor_set(v___x_492_, 2, v___x_489_);
return v___x_492_;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam(void){
_start:
{
lean_object* v___x_493_; 
v___x_493_ = lean_obj_once(&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__32, &lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__32_once, _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__32);
return v___x_493_;
}
}
static lean_object* _init_lp_mathlib_AddMonoid_nsmul__succ___autoParam(void){
_start:
{
lean_object* v___x_494_; 
v___x_494_ = lean_obj_once(&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__32, &lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__32_once, _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__32);
return v___x_494_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toAddSemigroup___redArg(lean_object* v_self_495_){
_start:
{
lean_object* v_toAdd_496_; 
v_toAdd_496_ = lean_ctor_get(v_self_495_, 1);
lean_inc(v_toAdd_496_);
return v_toAdd_496_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toAddSemigroup___redArg___boxed(lean_object* v_self_497_){
_start:
{
lean_object* v_res_498_; 
v_res_498_ = lp_mathlib_AddMonoid_toAddSemigroup___redArg(v_self_497_);
lean_dec_ref(v_self_497_);
return v_res_498_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toAddSemigroup(lean_object* v_M_499_, lean_object* v_self_500_){
_start:
{
lean_object* v_toAdd_501_; 
v_toAdd_501_ = lean_ctor_get(v_self_500_, 1);
lean_inc(v_toAdd_501_);
return v_toAdd_501_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toAddSemigroup___boxed(lean_object* v_M_502_, lean_object* v_self_503_){
_start:
{
lean_object* v_res_504_; 
v_res_504_ = lp_mathlib_AddMonoid_toAddSemigroup(v_M_502_, v_self_503_);
lean_dec_ref(v_self_503_);
return v_res_504_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object* v_self_505_){
_start:
{
lean_object* v_toZero_506_; lean_object* v_toAdd_507_; lean_object* v___x_508_; 
v_toZero_506_ = lean_ctor_get(v_self_505_, 0);
v_toAdd_507_ = lean_ctor_get(v_self_505_, 1);
lean_inc(v_toAdd_507_);
lean_inc(v_toZero_506_);
v___x_508_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_508_, 0, v_toZero_506_);
lean_ctor_set(v___x_508_, 1, v_toAdd_507_);
return v___x_508_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg___boxed(lean_object* v_self_509_){
_start:
{
lean_object* v_res_510_; 
v_res_510_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_self_509_);
lean_dec_ref(v_self_509_);
return v_res_510_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toAddZeroClass(lean_object* v_M_511_, lean_object* v_self_512_){
_start:
{
lean_object* v___x_513_; 
v___x_513_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_self_512_);
return v___x_513_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toAddZeroClass___boxed(lean_object* v_M_514_, lean_object* v_self_515_){
_start:
{
lean_object* v_res_516_; 
v_res_516_ = lp_mathlib_AddMonoid_toAddZeroClass(v_M_514_, v_self_515_);
lean_dec_ref(v_self_515_);
return v_res_516_;
}
}
static lean_object* _init_lp_mathlib_Monoid_npow__zero___autoParam(void){
_start:
{
lean_object* v___x_517_; 
v___x_517_ = lean_obj_once(&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__32, &lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__32_once, _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__32);
return v___x_517_;
}
}
static lean_object* _init_lp_mathlib_Monoid_npow__succ___autoParam(void){
_start:
{
lean_object* v___x_518_; 
v___x_518_ = lean_obj_once(&lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__32, &lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__32_once, _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam___closed__32);
return v___x_518_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toSemigroup___redArg(lean_object* v_self_519_){
_start:
{
lean_object* v_toMul_520_; 
v_toMul_520_ = lean_ctor_get(v_self_519_, 1);
lean_inc(v_toMul_520_);
return v_toMul_520_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toSemigroup___redArg___boxed(lean_object* v_self_521_){
_start:
{
lean_object* v_res_522_; 
v_res_522_ = lp_mathlib_Monoid_toSemigroup___redArg(v_self_521_);
lean_dec_ref(v_self_521_);
return v_res_522_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toSemigroup(lean_object* v_M_523_, lean_object* v_self_524_){
_start:
{
lean_object* v_toMul_525_; 
v_toMul_525_ = lean_ctor_get(v_self_524_, 1);
lean_inc(v_toMul_525_);
return v_toMul_525_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toSemigroup___boxed(lean_object* v_M_526_, lean_object* v_self_527_){
_start:
{
lean_object* v_res_528_; 
v_res_528_ = lp_mathlib_Monoid_toSemigroup(v_M_526_, v_self_527_);
lean_dec_ref(v_self_527_);
return v_res_528_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object* v_self_529_){
_start:
{
lean_object* v_toOne_530_; lean_object* v_toMul_531_; lean_object* v___x_532_; 
v_toOne_530_ = lean_ctor_get(v_self_529_, 0);
v_toMul_531_ = lean_ctor_get(v_self_529_, 1);
lean_inc(v_toMul_531_);
lean_inc(v_toOne_530_);
v___x_532_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_532_, 0, v_toOne_530_);
lean_ctor_set(v___x_532_, 1, v_toMul_531_);
return v___x_532_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toMulOneClass___redArg___boxed(lean_object* v_self_533_){
_start:
{
lean_object* v_res_534_; 
v_res_534_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_self_533_);
lean_dec_ref(v_self_533_);
return v_res_534_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toMulOneClass(lean_object* v_M_535_, lean_object* v_self_536_){
_start:
{
lean_object* v___x_537_; 
v___x_537_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_self_536_);
return v___x_537_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toMulOneClass___boxed(lean_object* v_M_538_, lean_object* v_self_539_){
_start:
{
lean_object* v_res_540_; 
v_res_540_ = lp_mathlib_Monoid_toMulOneClass(v_M_538_, v_self_539_);
lean_dec_ref(v_self_539_);
return v_res_540_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_toAddCommSemigroup___redArg(lean_object* v_self_541_){
_start:
{
lean_object* v_toAdd_542_; 
v_toAdd_542_ = lean_ctor_get(v_self_541_, 1);
lean_inc(v_toAdd_542_);
return v_toAdd_542_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_toAddCommSemigroup___redArg___boxed(lean_object* v_self_543_){
_start:
{
lean_object* v_res_544_; 
v_res_544_ = lp_mathlib_AddCommMonoid_toAddCommSemigroup___redArg(v_self_543_);
lean_dec_ref(v_self_543_);
return v_res_544_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_toAddCommSemigroup(lean_object* v_M_545_, lean_object* v_self_546_){
_start:
{
lean_object* v_toAdd_547_; 
v_toAdd_547_ = lean_ctor_get(v_self_546_, 1);
lean_inc(v_toAdd_547_);
return v_toAdd_547_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_toAddCommSemigroup___boxed(lean_object* v_M_548_, lean_object* v_self_549_){
_start:
{
lean_object* v_res_550_; 
v_res_550_ = lp_mathlib_AddCommMonoid_toAddCommSemigroup(v_M_548_, v_self_549_);
lean_dec_ref(v_self_549_);
return v_res_550_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommMonoid_toCommSemigroup___redArg(lean_object* v_self_551_){
_start:
{
lean_object* v_toMul_552_; 
v_toMul_552_ = lean_ctor_get(v_self_551_, 1);
lean_inc(v_toMul_552_);
return v_toMul_552_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommMonoid_toCommSemigroup___redArg___boxed(lean_object* v_self_553_){
_start:
{
lean_object* v_res_554_; 
v_res_554_ = lp_mathlib_CommMonoid_toCommSemigroup___redArg(v_self_553_);
lean_dec_ref(v_self_553_);
return v_res_554_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommMonoid_toCommSemigroup(lean_object* v_M_555_, lean_object* v_self_556_){
_start:
{
lean_object* v_toMul_557_; 
v_toMul_557_ = lean_ctor_get(v_self_556_, 1);
lean_inc(v_toMul_557_);
return v_toMul_557_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommMonoid_toCommSemigroup___boxed(lean_object* v_M_558_, lean_object* v_self_559_){
_start:
{
lean_object* v_res_560_; 
v_res_560_ = lp_mathlib_CommMonoid_toCommSemigroup(v_M_558_, v_self_559_);
lean_dec_ref(v_self_559_);
return v_res_560_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLeftCancelMonoid_toAddLeftCancelSemigroup___redArg(lean_object* v_self_561_){
_start:
{
lean_object* v_toAdd_562_; 
v_toAdd_562_ = lean_ctor_get(v_self_561_, 1);
lean_inc(v_toAdd_562_);
return v_toAdd_562_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLeftCancelMonoid_toAddLeftCancelSemigroup___redArg___boxed(lean_object* v_self_563_){
_start:
{
lean_object* v_res_564_; 
v_res_564_ = lp_mathlib_AddLeftCancelMonoid_toAddLeftCancelSemigroup___redArg(v_self_563_);
lean_dec_ref(v_self_563_);
return v_res_564_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLeftCancelMonoid_toAddLeftCancelSemigroup(lean_object* v_M_565_, lean_object* v_self_566_){
_start:
{
lean_object* v_toAdd_567_; 
v_toAdd_567_ = lean_ctor_get(v_self_566_, 1);
lean_inc(v_toAdd_567_);
return v_toAdd_567_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLeftCancelMonoid_toAddLeftCancelSemigroup___boxed(lean_object* v_M_568_, lean_object* v_self_569_){
_start:
{
lean_object* v_res_570_; 
v_res_570_ = lp_mathlib_AddLeftCancelMonoid_toAddLeftCancelSemigroup(v_M_568_, v_self_569_);
lean_dec_ref(v_self_569_);
return v_res_570_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LeftCancelMonoid_toLeftCancelSemigroup___redArg(lean_object* v_self_571_){
_start:
{
lean_object* v_toMul_572_; 
v_toMul_572_ = lean_ctor_get(v_self_571_, 1);
lean_inc(v_toMul_572_);
return v_toMul_572_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LeftCancelMonoid_toLeftCancelSemigroup___redArg___boxed(lean_object* v_self_573_){
_start:
{
lean_object* v_res_574_; 
v_res_574_ = lp_mathlib_LeftCancelMonoid_toLeftCancelSemigroup___redArg(v_self_573_);
lean_dec_ref(v_self_573_);
return v_res_574_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LeftCancelMonoid_toLeftCancelSemigroup(lean_object* v_M_575_, lean_object* v_self_576_){
_start:
{
lean_object* v_toMul_577_; 
v_toMul_577_ = lean_ctor_get(v_self_576_, 1);
lean_inc(v_toMul_577_);
return v_toMul_577_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LeftCancelMonoid_toLeftCancelSemigroup___boxed(lean_object* v_M_578_, lean_object* v_self_579_){
_start:
{
lean_object* v_res_580_; 
v_res_580_ = lp_mathlib_LeftCancelMonoid_toLeftCancelSemigroup(v_M_578_, v_self_579_);
lean_dec_ref(v_self_579_);
return v_res_580_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddRightCancelMonoid_toAddRightCancelSemigroup___redArg(lean_object* v_self_581_){
_start:
{
lean_object* v_toAdd_582_; 
v_toAdd_582_ = lean_ctor_get(v_self_581_, 1);
lean_inc(v_toAdd_582_);
return v_toAdd_582_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddRightCancelMonoid_toAddRightCancelSemigroup___redArg___boxed(lean_object* v_self_583_){
_start:
{
lean_object* v_res_584_; 
v_res_584_ = lp_mathlib_AddRightCancelMonoid_toAddRightCancelSemigroup___redArg(v_self_583_);
lean_dec_ref(v_self_583_);
return v_res_584_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddRightCancelMonoid_toAddRightCancelSemigroup(lean_object* v_M_585_, lean_object* v_self_586_){
_start:
{
lean_object* v_toAdd_587_; 
v_toAdd_587_ = lean_ctor_get(v_self_586_, 1);
lean_inc(v_toAdd_587_);
return v_toAdd_587_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddRightCancelMonoid_toAddRightCancelSemigroup___boxed(lean_object* v_M_588_, lean_object* v_self_589_){
_start:
{
lean_object* v_res_590_; 
v_res_590_ = lp_mathlib_AddRightCancelMonoid_toAddRightCancelSemigroup(v_M_588_, v_self_589_);
lean_dec_ref(v_self_589_);
return v_res_590_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightCancelMonoid_toRightCancelSemigroup___redArg(lean_object* v_self_591_){
_start:
{
lean_object* v_toMul_592_; 
v_toMul_592_ = lean_ctor_get(v_self_591_, 1);
lean_inc(v_toMul_592_);
return v_toMul_592_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightCancelMonoid_toRightCancelSemigroup___redArg___boxed(lean_object* v_self_593_){
_start:
{
lean_object* v_res_594_; 
v_res_594_ = lp_mathlib_RightCancelMonoid_toRightCancelSemigroup___redArg(v_self_593_);
lean_dec_ref(v_self_593_);
return v_res_594_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightCancelMonoid_toRightCancelSemigroup(lean_object* v_M_595_, lean_object* v_self_596_){
_start:
{
lean_object* v_toMul_597_; 
v_toMul_597_ = lean_ctor_get(v_self_596_, 1);
lean_inc(v_toMul_597_);
return v_toMul_597_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightCancelMonoid_toRightCancelSemigroup___boxed(lean_object* v_M_598_, lean_object* v_self_599_){
_start:
{
lean_object* v_res_600_; 
v_res_600_ = lp_mathlib_RightCancelMonoid_toRightCancelSemigroup(v_M_598_, v_self_599_);
lean_dec_ref(v_self_599_);
return v_res_600_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCancelMonoid_toAddRightCancelMonoid___redArg(lean_object* v_self_601_){
_start:
{
lean_inc_ref(v_self_601_);
return v_self_601_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCancelMonoid_toAddRightCancelMonoid___redArg___boxed(lean_object* v_self_602_){
_start:
{
lean_object* v_res_603_; 
v_res_603_ = lp_mathlib_AddCancelMonoid_toAddRightCancelMonoid___redArg(v_self_602_);
lean_dec_ref(v_self_602_);
return v_res_603_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCancelMonoid_toAddRightCancelMonoid(lean_object* v_M_604_, lean_object* v_self_605_){
_start:
{
lean_inc_ref(v_self_605_);
return v_self_605_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCancelMonoid_toAddRightCancelMonoid___boxed(lean_object* v_M_606_, lean_object* v_self_607_){
_start:
{
lean_object* v_res_608_; 
v_res_608_ = lp_mathlib_AddCancelMonoid_toAddRightCancelMonoid(v_M_606_, v_self_607_);
lean_dec_ref(v_self_607_);
return v_res_608_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelMonoid_toRightCancelMonoid___redArg(lean_object* v_self_609_){
_start:
{
lean_inc_ref(v_self_609_);
return v_self_609_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelMonoid_toRightCancelMonoid___redArg___boxed(lean_object* v_self_610_){
_start:
{
lean_object* v_res_611_; 
v_res_611_ = lp_mathlib_CancelMonoid_toRightCancelMonoid___redArg(v_self_610_);
lean_dec_ref(v_self_610_);
return v_res_611_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelMonoid_toRightCancelMonoid(lean_object* v_M_612_, lean_object* v_self_613_){
_start:
{
lean_inc_ref(v_self_613_);
return v_self_613_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelMonoid_toRightCancelMonoid___boxed(lean_object* v_M_614_, lean_object* v_self_615_){
_start:
{
lean_object* v_res_616_; 
v_res_616_ = lp_mathlib_CancelMonoid_toRightCancelMonoid(v_M_614_, v_self_615_);
lean_dec_ref(v_self_615_);
return v_res_616_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCancelCommMonoid_toAddLeftCancelMonoid___redArg(lean_object* v_self_617_){
_start:
{
lean_inc_ref(v_self_617_);
return v_self_617_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCancelCommMonoid_toAddLeftCancelMonoid___redArg___boxed(lean_object* v_self_618_){
_start:
{
lean_object* v_res_619_; 
v_res_619_ = lp_mathlib_AddCancelCommMonoid_toAddLeftCancelMonoid___redArg(v_self_618_);
lean_dec_ref(v_self_618_);
return v_res_619_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCancelCommMonoid_toAddLeftCancelMonoid(lean_object* v_M_620_, lean_object* v_self_621_){
_start:
{
lean_inc_ref(v_self_621_);
return v_self_621_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCancelCommMonoid_toAddLeftCancelMonoid___boxed(lean_object* v_M_622_, lean_object* v_self_623_){
_start:
{
lean_object* v_res_624_; 
v_res_624_ = lp_mathlib_AddCancelCommMonoid_toAddLeftCancelMonoid(v_M_622_, v_self_623_);
lean_dec_ref(v_self_623_);
return v_res_624_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelCommMonoid_toLeftCancelMonoid___redArg(lean_object* v_self_625_){
_start:
{
lean_inc_ref(v_self_625_);
return v_self_625_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelCommMonoid_toLeftCancelMonoid___redArg___boxed(lean_object* v_self_626_){
_start:
{
lean_object* v_res_627_; 
v_res_627_ = lp_mathlib_CancelCommMonoid_toLeftCancelMonoid___redArg(v_self_626_);
lean_dec_ref(v_self_626_);
return v_res_627_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelCommMonoid_toLeftCancelMonoid(lean_object* v_M_628_, lean_object* v_self_629_){
_start:
{
lean_inc_ref(v_self_629_);
return v_self_629_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelCommMonoid_toLeftCancelMonoid___boxed(lean_object* v_M_630_, lean_object* v_self_631_){
_start:
{
lean_object* v_res_632_; 
v_res_632_ = lp_mathlib_CancelCommMonoid_toLeftCancelMonoid(v_M_630_, v_self_631_);
lean_dec_ref(v_self_631_);
return v_res_632_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelCommMonoid_toCancelMonoid___redArg(lean_object* v_inst_633_){
_start:
{
lean_inc_ref(v_inst_633_);
return v_inst_633_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelCommMonoid_toCancelMonoid___redArg___boxed(lean_object* v_inst_634_){
_start:
{
lean_object* v_res_635_; 
v_res_635_ = lp_mathlib_CancelCommMonoid_toCancelMonoid___redArg(v_inst_634_);
lean_dec_ref(v_inst_634_);
return v_res_635_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelCommMonoid_toCancelMonoid(lean_object* v_M_636_, lean_object* v_inst_637_){
_start:
{
lean_inc_ref(v_inst_637_);
return v_inst_637_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelCommMonoid_toCancelMonoid___boxed(lean_object* v_M_638_, lean_object* v_inst_639_){
_start:
{
lean_object* v_res_640_; 
v_res_640_ = lp_mathlib_CancelCommMonoid_toCancelMonoid(v_M_638_, v_inst_639_);
lean_dec_ref(v_inst_639_);
return v_res_640_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCancelCommMonoid_toAddCancelMonoid___redArg(lean_object* v_inst_641_){
_start:
{
lean_inc_ref(v_inst_641_);
return v_inst_641_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCancelCommMonoid_toAddCancelMonoid___redArg___boxed(lean_object* v_inst_642_){
_start:
{
lean_object* v_res_643_; 
v_res_643_ = lp_mathlib_AddCancelCommMonoid_toAddCancelMonoid___redArg(v_inst_642_);
lean_dec_ref(v_inst_642_);
return v_res_643_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCancelCommMonoid_toAddCancelMonoid(lean_object* v_M_644_, lean_object* v_inst_645_){
_start:
{
lean_inc_ref(v_inst_645_);
return v_inst_645_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCancelCommMonoid_toAddCancelMonoid___boxed(lean_object* v_M_646_, lean_object* v_inst_647_){
_start:
{
lean_object* v_res_648_; 
v_res_648_ = lp_mathlib_AddCancelCommMonoid_toAddCancelMonoid(v_M_646_, v_inst_647_);
lean_dec_ref(v_inst_647_);
return v_res_648_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Logic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Semigroup(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_BinaryRec(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Notation(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Push_Attr(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Monoid(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Logic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Semigroup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_BinaryRec(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Push_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Monoid(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_LibraryNote_forgetful__inheritance = _init_lp_mathlib_LibraryNote_forgetful__inheritance();
lean_mark_persistent(lp_mathlib_LibraryNote_forgetful__inheritance);
lp_mathlib_AddMonoid_nsmul__zero___autoParam = _init_lp_mathlib_AddMonoid_nsmul__zero___autoParam();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__zero___autoParam);
lp_mathlib_AddMonoid_nsmul__succ___autoParam = _init_lp_mathlib_AddMonoid_nsmul__succ___autoParam();
lean_mark_persistent(lp_mathlib_AddMonoid_nsmul__succ___autoParam);
lp_mathlib_Monoid_npow__zero___autoParam = _init_lp_mathlib_Monoid_npow__zero___autoParam();
lean_mark_persistent(lp_mathlib_Monoid_npow__zero___autoParam);
lp_mathlib_Monoid_npow__succ___autoParam = _init_lp_mathlib_Monoid_npow__succ___autoParam();
lean_mark_persistent(lp_mathlib_Monoid_npow__succ___autoParam);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Logic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Semigroup(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_BinaryRec(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Notation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Push_Attr(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Monoid(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Logic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Semigroup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_BinaryRec(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Push_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Monoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Monoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Monoid(builtin);
}
#ifdef __cplusplus
}
#endif
