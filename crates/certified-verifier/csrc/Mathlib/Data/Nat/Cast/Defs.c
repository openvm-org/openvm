// Lean compiler output
// Module: Mathlib.Data.Nat.Cast.Defs
// Imports: public import Init public meta import Init public import Mathlib.Data.Nat.Init public import Mathlib.Tactic.SplitIfs public import Mathlib.Algebra.Group.Monoid public import Mathlib.Tactic.OfNat
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
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_unaryCast___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_unaryCast___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_unaryCast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_unaryCast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOfNatAtLeastTwo___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOfNatAtLeastTwo(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LibraryNote_no__index__around__OfNat_x2eofNat;
static const lean_string_object lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__0 = (const lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__0_value;
static const lean_string_object lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__1 = (const lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__1_value;
static const lean_string_object lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__2 = (const lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__2_value;
static const lean_string_object lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__3 = (const lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__3_value;
static const lean_ctor_object lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__4_value_aux_0),((lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__4_value_aux_1),((lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__4_value_aux_2),((lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__4 = (const lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__4_value;
static const lean_array_object lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__5 = (const lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__5_value;
static const lean_string_object lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__6 = (const lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__6_value;
static const lean_ctor_object lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__7_value_aux_0),((lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__7_value_aux_1),((lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__7_value_aux_2),((lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__6_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__7 = (const lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__7_value;
static const lean_string_object lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__8 = (const lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__8_value;
static const lean_ctor_object lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__9 = (const lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__9_value;
static const lean_string_object lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "intros"};
static const lean_object* lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__10 = (const lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__10_value;
static const lean_ctor_object lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__11_value_aux_0),((lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__11_value_aux_1),((lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__11_value_aux_2),((lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__10_value),LEAN_SCALAR_PTR_LITERAL(26, 175, 18, 116, 252, 50, 128, 45)}};
static const lean_object* lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__11 = (const lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__11_value;
static lean_once_cell_t lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__12;
static lean_once_cell_t lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__13;
static const lean_ctor_object lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__9_value),((lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__5_value)}};
static const lean_object* lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__14 = (const lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__14_value;
static lean_once_cell_t lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__15;
static lean_once_cell_t lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__16;
static lean_once_cell_t lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__17;
static const lean_string_object lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ";"};
static const lean_object* lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__18 = (const lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__18_value;
static lean_once_cell_t lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__19;
static lean_once_cell_t lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__20;
static const lean_string_object lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticRfl"};
static const lean_object* lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__21 = (const lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__21_value;
static const lean_ctor_object lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__22_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__22_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__22_value_aux_0),((lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__22_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__22_value_aux_1),((lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__22_value_aux_2),((lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__21_value),LEAN_SCALAR_PTR_LITERAL(201, 188, 173, 198, 169, 252, 183, 45)}};
static const lean_object* lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__22 = (const lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__22_value;
static const lean_string_object lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "rfl"};
static const lean_object* lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__23 = (const lean_object*)&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__23_value;
static lean_once_cell_t lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__24;
static lean_once_cell_t lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__25;
static lean_once_cell_t lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__26;
static lean_once_cell_t lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__27;
static lean_once_cell_t lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__28;
static lean_once_cell_t lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__29;
static lean_once_cell_t lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__30;
static lean_once_cell_t lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__31;
static lean_once_cell_t lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__32_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__32;
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidWithOne_natCast__succ___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoidWithOne_toAddCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoidWithOne_toAddCommMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoidWithOne_toAddCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoidWithOne_toAddCommMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_binCast___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_binCast___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_binCast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_binCast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Cast_Defs_0__Nat_unaryCast_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Cast_Defs_0__Nat_unaryCast_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Cast_Defs_0__Nat_unaryCast_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Cast_Defs_0__Nat_unaryCast_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidWithOne_unary___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidWithOne_unary(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidWithOne_binary___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidWithOne_binary(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_unaryCast___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_, lean_object* v_inst_3_, lean_object* v_x_4_){
_start:
{
lean_object* v_zero_5_; uint8_t v_isZero_6_; 
v_zero_5_ = lean_unsigned_to_nat(0u);
v_isZero_6_ = lean_nat_dec_eq(v_x_4_, v_zero_5_);
if (v_isZero_6_ == 1)
{
lean_dec(v_inst_3_);
lean_dec(v_inst_1_);
lean_inc(v_inst_2_);
return v_inst_2_;
}
else
{
lean_object* v_one_7_; lean_object* v_n_8_; lean_object* v___x_9_; lean_object* v___x_10_; 
v_one_7_ = lean_unsigned_to_nat(1u);
v_n_8_ = lean_nat_sub(v_x_4_, v_one_7_);
lean_inc(v_inst_3_);
lean_inc(v_inst_1_);
v___x_9_ = lp_mathlib_Nat_unaryCast___redArg(v_inst_1_, v_inst_2_, v_inst_3_, v_n_8_);
lean_dec(v_n_8_);
v___x_10_ = lean_apply_2(v_inst_3_, v___x_9_, v_inst_1_);
return v___x_10_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_unaryCast___redArg___boxed(lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_inst_13_, lean_object* v_x_14_){
_start:
{
lean_object* v_res_15_; 
v_res_15_ = lp_mathlib_Nat_unaryCast___redArg(v_inst_11_, v_inst_12_, v_inst_13_, v_x_14_);
lean_dec(v_x_14_);
lean_dec(v_inst_12_);
return v_res_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_unaryCast(lean_object* v_R_16_, lean_object* v_inst_17_, lean_object* v_inst_18_, lean_object* v_inst_19_, lean_object* v_x_20_){
_start:
{
lean_object* v___x_21_; 
v___x_21_ = lp_mathlib_Nat_unaryCast___redArg(v_inst_17_, v_inst_18_, v_inst_19_, v_x_20_);
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_unaryCast___boxed(lean_object* v_R_22_, lean_object* v_inst_23_, lean_object* v_inst_24_, lean_object* v_inst_25_, lean_object* v_x_26_){
_start:
{
lean_object* v_res_27_; 
v_res_27_ = lp_mathlib_Nat_unaryCast(v_R_22_, v_inst_23_, v_inst_24_, v_inst_25_, v_x_26_);
lean_dec(v_x_26_);
lean_dec(v_inst_24_);
return v_res_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOfNatAtLeastTwo___redArg(lean_object* v_n_28_, lean_object* v_inst_29_){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = lean_apply_1(v_inst_29_, v_n_28_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOfNatAtLeastTwo(lean_object* v_R_31_, lean_object* v_n_32_, lean_object* v_inst_33_, lean_object* v_inst_34_){
_start:
{
lean_object* v___x_35_; 
v___x_35_ = lean_apply_1(v_inst_33_, v_n_32_);
return v___x_35_;
}
}
static lean_object* _init_lp_mathlib_LibraryNote_no__index__around__OfNat_x2eofNat(void){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lean_box(0);
return v___x_36_;
}
}
static lean_object* _init_lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__12(void){
_start:
{
lean_object* v___x_63_; lean_object* v___x_64_; 
v___x_63_ = ((lean_object*)(lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__10));
v___x_64_ = l_Lean_mkAtom(v___x_63_);
return v___x_64_;
}
}
static lean_object* _init_lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__13(void){
_start:
{
lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; 
v___x_65_ = lean_obj_once(&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__12, &lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__12_once, _init_lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__12);
v___x_66_ = ((lean_object*)(lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__5));
v___x_67_ = lean_array_push(v___x_66_, v___x_65_);
return v___x_67_;
}
}
static lean_object* _init_lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__15(void){
_start:
{
lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; 
v___x_72_ = ((lean_object*)(lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__14));
v___x_73_ = lean_obj_once(&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__13, &lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__13_once, _init_lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__13);
v___x_74_ = lean_array_push(v___x_73_, v___x_72_);
return v___x_74_;
}
}
static lean_object* _init_lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__16(void){
_start:
{
lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; 
v___x_75_ = lean_obj_once(&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__15, &lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__15_once, _init_lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__15);
v___x_76_ = ((lean_object*)(lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__11));
v___x_77_ = lean_box(2);
v___x_78_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_78_, 0, v___x_77_);
lean_ctor_set(v___x_78_, 1, v___x_76_);
lean_ctor_set(v___x_78_, 2, v___x_75_);
return v___x_78_;
}
}
static lean_object* _init_lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__17(void){
_start:
{
lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; 
v___x_79_ = lean_obj_once(&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__16, &lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__16_once, _init_lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__16);
v___x_80_ = ((lean_object*)(lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__5));
v___x_81_ = lean_array_push(v___x_80_, v___x_79_);
return v___x_81_;
}
}
static lean_object* _init_lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__19(void){
_start:
{
lean_object* v___x_83_; lean_object* v___x_84_; 
v___x_83_ = ((lean_object*)(lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__18));
v___x_84_ = l_Lean_mkAtom(v___x_83_);
return v___x_84_;
}
}
static lean_object* _init_lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__20(void){
_start:
{
lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; 
v___x_85_ = lean_obj_once(&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__19, &lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__19_once, _init_lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__19);
v___x_86_ = lean_obj_once(&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__17, &lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__17_once, _init_lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__17);
v___x_87_ = lean_array_push(v___x_86_, v___x_85_);
return v___x_87_;
}
}
static lean_object* _init_lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__24(void){
_start:
{
lean_object* v___x_95_; lean_object* v___x_96_; 
v___x_95_ = ((lean_object*)(lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__23));
v___x_96_ = l_Lean_mkAtom(v___x_95_);
return v___x_96_;
}
}
static lean_object* _init_lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__25(void){
_start:
{
lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; 
v___x_97_ = lean_obj_once(&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__24, &lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__24_once, _init_lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__24);
v___x_98_ = ((lean_object*)(lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__5));
v___x_99_ = lean_array_push(v___x_98_, v___x_97_);
return v___x_99_;
}
}
static lean_object* _init_lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__26(void){
_start:
{
lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; 
v___x_100_ = lean_obj_once(&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__25, &lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__25_once, _init_lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__25);
v___x_101_ = ((lean_object*)(lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__22));
v___x_102_ = lean_box(2);
v___x_103_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_103_, 0, v___x_102_);
lean_ctor_set(v___x_103_, 1, v___x_101_);
lean_ctor_set(v___x_103_, 2, v___x_100_);
return v___x_103_;
}
}
static lean_object* _init_lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__27(void){
_start:
{
lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; 
v___x_104_ = lean_obj_once(&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__26, &lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__26_once, _init_lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__26);
v___x_105_ = lean_obj_once(&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__20, &lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__20_once, _init_lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__20);
v___x_106_ = lean_array_push(v___x_105_, v___x_104_);
return v___x_106_;
}
}
static lean_object* _init_lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__28(void){
_start:
{
lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; 
v___x_107_ = lean_obj_once(&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__27, &lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__27_once, _init_lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__27);
v___x_108_ = ((lean_object*)(lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__9));
v___x_109_ = lean_box(2);
v___x_110_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_110_, 0, v___x_109_);
lean_ctor_set(v___x_110_, 1, v___x_108_);
lean_ctor_set(v___x_110_, 2, v___x_107_);
return v___x_110_;
}
}
static lean_object* _init_lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__29(void){
_start:
{
lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; 
v___x_111_ = lean_obj_once(&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__28, &lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__28_once, _init_lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__28);
v___x_112_ = ((lean_object*)(lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__5));
v___x_113_ = lean_array_push(v___x_112_, v___x_111_);
return v___x_113_;
}
}
static lean_object* _init_lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__30(void){
_start:
{
lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; 
v___x_114_ = lean_obj_once(&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__29, &lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__29_once, _init_lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__29);
v___x_115_ = ((lean_object*)(lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__7));
v___x_116_ = lean_box(2);
v___x_117_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_117_, 0, v___x_116_);
lean_ctor_set(v___x_117_, 1, v___x_115_);
lean_ctor_set(v___x_117_, 2, v___x_114_);
return v___x_117_;
}
}
static lean_object* _init_lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__31(void){
_start:
{
lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; 
v___x_118_ = lean_obj_once(&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__30, &lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__30_once, _init_lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__30);
v___x_119_ = ((lean_object*)(lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__5));
v___x_120_ = lean_array_push(v___x_119_, v___x_118_);
return v___x_120_;
}
}
static lean_object* _init_lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__32(void){
_start:
{
lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; 
v___x_121_ = lean_obj_once(&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__31, &lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__31_once, _init_lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__31);
v___x_122_ = ((lean_object*)(lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__4));
v___x_123_ = lean_box(2);
v___x_124_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_124_, 0, v___x_123_);
lean_ctor_set(v___x_124_, 1, v___x_122_);
lean_ctor_set(v___x_124_, 2, v___x_121_);
return v___x_124_;
}
}
static lean_object* _init_lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam(void){
_start:
{
lean_object* v___x_125_; 
v___x_125_ = lean_obj_once(&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__32, &lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__32_once, _init_lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__32);
return v___x_125_;
}
}
static lean_object* _init_lp_mathlib_AddMonoidWithOne_natCast__succ___autoParam(void){
_start:
{
lean_object* v___x_126_; 
v___x_126_ = lean_obj_once(&lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__32, &lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__32_once, _init_lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam___closed__32);
return v___x_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoidWithOne_toAddCommMonoid___redArg(lean_object* v_self_127_){
_start:
{
lean_object* v_toAddMonoid_128_; 
v_toAddMonoid_128_ = lean_ctor_get(v_self_127_, 1);
lean_inc_ref(v_toAddMonoid_128_);
return v_toAddMonoid_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoidWithOne_toAddCommMonoid___redArg___boxed(lean_object* v_self_129_){
_start:
{
lean_object* v_res_130_; 
v_res_130_ = lp_mathlib_AddCommMonoidWithOne_toAddCommMonoid___redArg(v_self_129_);
lean_dec_ref(v_self_129_);
return v_res_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoidWithOne_toAddCommMonoid(lean_object* v_R_131_, lean_object* v_self_132_){
_start:
{
lean_object* v_toAddMonoid_133_; 
v_toAddMonoid_133_ = lean_ctor_get(v_self_132_, 1);
lean_inc_ref(v_toAddMonoid_133_);
return v_toAddMonoid_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoidWithOne_toAddCommMonoid___boxed(lean_object* v_R_134_, lean_object* v_self_135_){
_start:
{
lean_object* v_res_136_; 
v_res_136_ = lp_mathlib_AddCommMonoidWithOne_toAddCommMonoid(v_R_134_, v_self_135_);
lean_dec_ref(v_self_135_);
return v_res_136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_binCast___redArg(lean_object* v_inst_137_, lean_object* v_inst_138_, lean_object* v_inst_139_, lean_object* v_x_140_){
_start:
{
lean_object* v_zero_141_; uint8_t v_isZero_142_; 
v_zero_141_ = lean_unsigned_to_nat(0u);
v_isZero_142_ = lean_nat_dec_eq(v_x_140_, v_zero_141_);
if (v_isZero_142_ == 1)
{
lean_dec(v_inst_139_);
lean_dec(v_inst_138_);
lean_inc(v_inst_137_);
return v_inst_137_;
}
else
{
lean_object* v_one_143_; lean_object* v_n_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; uint8_t v___x_148_; 
v_one_143_ = lean_unsigned_to_nat(1u);
v_n_144_ = lean_nat_sub(v_x_140_, v_one_143_);
v___x_145_ = lean_nat_add(v_n_144_, v_one_143_);
lean_dec(v_n_144_);
v___x_146_ = lean_unsigned_to_nat(2u);
v___x_147_ = lean_nat_mod(v___x_145_, v___x_146_);
v___x_148_ = lean_nat_dec_eq(v___x_147_, v_zero_141_);
lean_dec(v___x_147_);
if (v___x_148_ == 0)
{
lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; 
v___x_149_ = lean_nat_shiftr(v___x_145_, v_one_143_);
lean_dec(v___x_145_);
lean_inc_n(v_inst_139_, 2);
lean_inc(v_inst_138_);
v___x_150_ = lp_mathlib_Nat_binCast___redArg(v_inst_137_, v_inst_138_, v_inst_139_, v___x_149_);
lean_dec(v___x_149_);
lean_inc(v___x_150_);
v___x_151_ = lean_apply_2(v_inst_139_, v___x_150_, v___x_150_);
v___x_152_ = lean_apply_2(v_inst_139_, v___x_151_, v_inst_138_);
return v___x_152_;
}
else
{
lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; 
v___x_153_ = lean_nat_shiftr(v___x_145_, v_one_143_);
lean_dec(v___x_145_);
lean_inc(v_inst_139_);
v___x_154_ = lp_mathlib_Nat_binCast___redArg(v_inst_137_, v_inst_138_, v_inst_139_, v___x_153_);
lean_dec(v___x_153_);
lean_inc(v___x_154_);
v___x_155_ = lean_apply_2(v_inst_139_, v___x_154_, v___x_154_);
return v___x_155_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_binCast___redArg___boxed(lean_object* v_inst_156_, lean_object* v_inst_157_, lean_object* v_inst_158_, lean_object* v_x_159_){
_start:
{
lean_object* v_res_160_; 
v_res_160_ = lp_mathlib_Nat_binCast___redArg(v_inst_156_, v_inst_157_, v_inst_158_, v_x_159_);
lean_dec(v_x_159_);
lean_dec(v_inst_156_);
return v_res_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_binCast(lean_object* v_R_161_, lean_object* v_inst_162_, lean_object* v_inst_163_, lean_object* v_inst_164_, lean_object* v_x_165_){
_start:
{
lean_object* v___x_166_; 
v___x_166_ = lp_mathlib_Nat_binCast___redArg(v_inst_162_, v_inst_163_, v_inst_164_, v_x_165_);
return v___x_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_binCast___boxed(lean_object* v_R_167_, lean_object* v_inst_168_, lean_object* v_inst_169_, lean_object* v_inst_170_, lean_object* v_x_171_){
_start:
{
lean_object* v_res_172_; 
v_res_172_ = lp_mathlib_Nat_binCast(v_R_167_, v_inst_168_, v_inst_169_, v_inst_170_, v_x_171_);
lean_dec(v_x_171_);
lean_dec(v_inst_168_);
return v_res_172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Cast_Defs_0__Nat_unaryCast_match__1_splitter___redArg(lean_object* v_x_173_, lean_object* v_h__1_174_, lean_object* v_h__2_175_){
_start:
{
lean_object* v_zero_176_; uint8_t v_isZero_177_; 
v_zero_176_ = lean_unsigned_to_nat(0u);
v_isZero_177_ = lean_nat_dec_eq(v_x_173_, v_zero_176_);
if (v_isZero_177_ == 1)
{
lean_object* v___x_178_; lean_object* v___x_179_; 
lean_dec(v_h__2_175_);
v___x_178_ = lean_box(0);
v___x_179_ = lean_apply_1(v_h__1_174_, v___x_178_);
return v___x_179_;
}
else
{
lean_object* v_one_180_; lean_object* v_n_181_; lean_object* v___x_182_; 
lean_dec(v_h__1_174_);
v_one_180_ = lean_unsigned_to_nat(1u);
v_n_181_ = lean_nat_sub(v_x_173_, v_one_180_);
v___x_182_ = lean_apply_1(v_h__2_175_, v_n_181_);
return v___x_182_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Cast_Defs_0__Nat_unaryCast_match__1_splitter___redArg___boxed(lean_object* v_x_183_, lean_object* v_h__1_184_, lean_object* v_h__2_185_){
_start:
{
lean_object* v_res_186_; 
v_res_186_ = lp_mathlib___private_Mathlib_Data_Nat_Cast_Defs_0__Nat_unaryCast_match__1_splitter___redArg(v_x_183_, v_h__1_184_, v_h__2_185_);
lean_dec(v_x_183_);
return v_res_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Cast_Defs_0__Nat_unaryCast_match__1_splitter(lean_object* v_motive_187_, lean_object* v_x_188_, lean_object* v_h__1_189_, lean_object* v_h__2_190_){
_start:
{
lean_object* v_zero_191_; uint8_t v_isZero_192_; 
v_zero_191_ = lean_unsigned_to_nat(0u);
v_isZero_192_ = lean_nat_dec_eq(v_x_188_, v_zero_191_);
if (v_isZero_192_ == 1)
{
lean_object* v___x_193_; lean_object* v___x_194_; 
lean_dec(v_h__2_190_);
v___x_193_ = lean_box(0);
v___x_194_ = lean_apply_1(v_h__1_189_, v___x_193_);
return v___x_194_;
}
else
{
lean_object* v_one_195_; lean_object* v_n_196_; lean_object* v___x_197_; 
lean_dec(v_h__1_189_);
v_one_195_ = lean_unsigned_to_nat(1u);
v_n_196_ = lean_nat_sub(v_x_188_, v_one_195_);
v___x_197_ = lean_apply_1(v_h__2_190_, v_n_196_);
return v___x_197_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Cast_Defs_0__Nat_unaryCast_match__1_splitter___boxed(lean_object* v_motive_198_, lean_object* v_x_199_, lean_object* v_h__1_200_, lean_object* v_h__2_201_){
_start:
{
lean_object* v_res_202_; 
v_res_202_ = lp_mathlib___private_Mathlib_Data_Nat_Cast_Defs_0__Nat_unaryCast_match__1_splitter(v_motive_198_, v_x_199_, v_h__1_200_, v_h__2_201_);
lean_dec(v_x_199_);
return v_res_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidWithOne_unary___redArg(lean_object* v_inst_203_, lean_object* v_inst_204_){
_start:
{
lean_object* v_toZero_205_; lean_object* v_toAdd_206_; lean_object* v___x_207_; lean_object* v___x_208_; 
v_toZero_205_ = lean_ctor_get(v_inst_203_, 0);
v_toAdd_206_ = lean_ctor_get(v_inst_203_, 1);
lean_inc(v_toAdd_206_);
lean_inc(v_toZero_205_);
lean_inc(v_inst_204_);
v___x_207_ = lean_alloc_closure((void*)(lp_mathlib_Nat_unaryCast___boxed), 5, 4);
lean_closure_set(v___x_207_, 0, lean_box(0));
lean_closure_set(v___x_207_, 1, v_inst_204_);
lean_closure_set(v___x_207_, 2, v_toZero_205_);
lean_closure_set(v___x_207_, 3, v_toAdd_206_);
v___x_208_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_208_, 0, v___x_207_);
lean_ctor_set(v___x_208_, 1, v_inst_203_);
lean_ctor_set(v___x_208_, 2, v_inst_204_);
return v___x_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidWithOne_unary(lean_object* v_R_209_, lean_object* v_inst_210_, lean_object* v_inst_211_){
_start:
{
lean_object* v_toZero_212_; lean_object* v_toAdd_213_; lean_object* v___x_214_; lean_object* v___x_215_; 
v_toZero_212_ = lean_ctor_get(v_inst_210_, 0);
v_toAdd_213_ = lean_ctor_get(v_inst_210_, 1);
lean_inc(v_toAdd_213_);
lean_inc(v_toZero_212_);
lean_inc(v_inst_211_);
v___x_214_ = lean_alloc_closure((void*)(lp_mathlib_Nat_unaryCast___boxed), 5, 4);
lean_closure_set(v___x_214_, 0, lean_box(0));
lean_closure_set(v___x_214_, 1, v_inst_211_);
lean_closure_set(v___x_214_, 2, v_toZero_212_);
lean_closure_set(v___x_214_, 3, v_toAdd_213_);
v___x_215_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_215_, 0, v___x_214_);
lean_ctor_set(v___x_215_, 1, v_inst_210_);
lean_ctor_set(v___x_215_, 2, v_inst_211_);
return v___x_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidWithOne_binary___redArg(lean_object* v_inst_216_, lean_object* v_inst_217_){
_start:
{
lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v_toZero_220_; lean_object* v_toAdd_221_; lean_object* v___x_222_; lean_object* v___x_223_; 
v___x_218_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_216_);
v___x_219_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_218_);
v_toZero_220_ = lean_ctor_get(v___x_219_, 0);
lean_inc(v_toZero_220_);
lean_dec_ref(v___x_219_);
v_toAdd_221_ = lean_ctor_get(v_inst_216_, 1);
lean_inc(v_toAdd_221_);
lean_inc(v_inst_217_);
v___x_222_ = lean_alloc_closure((void*)(lp_mathlib_Nat_binCast___boxed), 5, 4);
lean_closure_set(v___x_222_, 0, lean_box(0));
lean_closure_set(v___x_222_, 1, v_toZero_220_);
lean_closure_set(v___x_222_, 2, v_inst_217_);
lean_closure_set(v___x_222_, 3, v_toAdd_221_);
v___x_223_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_223_, 0, v___x_222_);
lean_ctor_set(v___x_223_, 1, v_inst_216_);
lean_ctor_set(v___x_223_, 2, v_inst_217_);
return v___x_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidWithOne_binary(lean_object* v_R_224_, lean_object* v_inst_225_, lean_object* v_inst_226_){
_start:
{
lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v_toZero_229_; lean_object* v_toAdd_230_; lean_object* v___x_231_; lean_object* v___x_232_; 
v___x_227_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_225_);
v___x_228_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_227_);
v_toZero_229_ = lean_ctor_get(v___x_228_, 0);
lean_inc(v_toZero_229_);
lean_dec_ref(v___x_228_);
v_toAdd_230_ = lean_ctor_get(v_inst_225_, 1);
lean_inc(v_toAdd_230_);
lean_inc(v_inst_226_);
v___x_231_ = lean_alloc_closure((void*)(lp_mathlib_Nat_binCast___boxed), 5, 4);
lean_closure_set(v___x_231_, 0, lean_box(0));
lean_closure_set(v___x_231_, 1, v_toZero_229_);
lean_closure_set(v___x_231_, 2, v_inst_226_);
lean_closure_set(v___x_231_, 3, v_toAdd_230_);
v___x_232_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_232_, 0, v___x_231_);
lean_ctor_set(v___x_232_, 1, v_inst_225_);
lean_ctor_set(v___x_232_, 2, v_inst_226_);
return v___x_232_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_SplitIfs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Monoid(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_OfNat(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_SplitIfs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Monoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_OfNat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Nat_Cast_Defs(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_LibraryNote_no__index__around__OfNat_x2eofNat = _init_lp_mathlib_LibraryNote_no__index__around__OfNat_x2eofNat();
lean_mark_persistent(lp_mathlib_LibraryNote_no__index__around__OfNat_x2eofNat);
lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam = _init_lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam();
lean_mark_persistent(lp_mathlib_AddMonoidWithOne_natCast__zero___autoParam);
lp_mathlib_AddMonoidWithOne_natCast__succ___autoParam = _init_lp_mathlib_AddMonoidWithOne_natCast__succ___autoParam();
lean_mark_persistent(lp_mathlib_AddMonoidWithOne_natCast__succ___autoParam);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_SplitIfs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Monoid(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_OfNat(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Nat_Cast_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_SplitIfs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Monoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_OfNat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Nat_Cast_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Nat_Cast_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
