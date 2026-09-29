// Lean compiler output
// Module: Mathlib.Data.Int.Cast.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Defs public import Mathlib.Data.Nat.Cast.Defs
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
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
uint8_t lean_int_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_abs(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Int_castDef___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Int_castDef___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Int_castDef___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_castDef___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_castDef(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_castDef___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__0 = (const lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__0_value;
static const lean_string_object lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__1 = (const lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__1_value;
static const lean_string_object lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__2 = (const lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__2_value;
static const lean_string_object lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__3 = (const lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__3_value;
static const lean_ctor_object lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__4_value_aux_0),((lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__4_value_aux_1),((lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__4_value_aux_2),((lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__4 = (const lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__4_value;
static const lean_array_object lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__5 = (const lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__5_value;
static const lean_string_object lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__6 = (const lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__6_value;
static const lean_ctor_object lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__7_value_aux_0),((lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__7_value_aux_1),((lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__7_value_aux_2),((lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__6_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__7 = (const lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__7_value;
static const lean_string_object lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__8 = (const lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__8_value;
static const lean_ctor_object lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__9 = (const lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__9_value;
static const lean_string_object lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "intros"};
static const lean_object* lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__10 = (const lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__10_value;
static const lean_ctor_object lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__11_value_aux_0),((lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__11_value_aux_1),((lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__11_value_aux_2),((lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__10_value),LEAN_SCALAR_PTR_LITERAL(26, 175, 18, 116, 252, 50, 128, 45)}};
static const lean_object* lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__11 = (const lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__11_value;
static lean_once_cell_t lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__12;
static lean_once_cell_t lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__13;
static const lean_ctor_object lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__9_value),((lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__5_value)}};
static const lean_object* lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__14 = (const lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__14_value;
static lean_once_cell_t lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__15;
static lean_once_cell_t lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__16;
static lean_once_cell_t lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__17;
static const lean_string_object lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ";"};
static const lean_object* lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__18 = (const lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__18_value;
static lean_once_cell_t lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__19;
static lean_once_cell_t lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__20;
static const lean_string_object lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticRfl"};
static const lean_object* lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__21 = (const lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__21_value;
static const lean_ctor_object lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__22_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__22_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__22_value_aux_0),((lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__22_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__22_value_aux_1),((lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__22_value_aux_2),((lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__21_value),LEAN_SCALAR_PTR_LITERAL(201, 188, 173, 198, 169, 252, 183, 45)}};
static const lean_object* lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__22 = (const lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__22_value;
static const lean_string_object lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "rfl"};
static const lean_object* lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__23 = (const lean_object*)&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__23_value;
static lean_once_cell_t lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__24;
static lean_once_cell_t lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__25;
static lean_once_cell_t lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__26;
static lean_once_cell_t lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__27;
static lean_once_cell_t lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__28;
static lean_once_cell_t lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__29;
static lean_once_cell_t lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__30;
static lean_once_cell_t lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__31;
static lean_once_cell_t lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__32_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__32;
LEAN_EXPORT lean_object* lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_AddGroupWithOne_intCast__negSucc___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_AddGroupWithOne_toAddGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddGroupWithOne_toAddGroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddGroupWithOne_toAddGroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddGroupWithOne_toAddGroup___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroupWithOne_toAddCommMonoidWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroupWithOne_toAddCommMonoidWithOne___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroupWithOne_toAddCommMonoidWithOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroupWithOne_toAddCommMonoidWithOne___boxed(lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Int_castDef___redArg___closed__0(void){
_start:
{
lean_object* v_natZero_1_; lean_object* v_intZero_2_; 
v_natZero_1_ = lean_unsigned_to_nat(0u);
v_intZero_2_ = lean_nat_to_int(v_natZero_1_);
return v_intZero_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_castDef___redArg(lean_object* v_inst_3_, lean_object* v_inst_4_, lean_object* v_x_5_){
_start:
{
lean_object* v_intZero_6_; uint8_t v_isNeg_7_; 
v_intZero_6_ = lean_obj_once(&lp_mathlib_Int_castDef___redArg___closed__0, &lp_mathlib_Int_castDef___redArg___closed__0_once, _init_lp_mathlib_Int_castDef___redArg___closed__0);
v_isNeg_7_ = lean_int_dec_lt(v_x_5_, v_intZero_6_);
if (v_isNeg_7_ == 0)
{
lean_object* v_a_8_; lean_object* v___x_9_; 
lean_dec(v_inst_4_);
v_a_8_ = lean_nat_abs(v_x_5_);
v___x_9_ = lean_apply_1(v_inst_3_, v_a_8_);
return v___x_9_;
}
else
{
lean_object* v_abs_10_; lean_object* v_one_11_; lean_object* v_a_12_; lean_object* v___x_13_; lean_object* v___x_14_; lean_object* v___x_15_; 
v_abs_10_ = lean_nat_abs(v_x_5_);
v_one_11_ = lean_unsigned_to_nat(1u);
v_a_12_ = lean_nat_sub(v_abs_10_, v_one_11_);
lean_dec(v_abs_10_);
v___x_13_ = lean_nat_add(v_a_12_, v_one_11_);
lean_dec(v_a_12_);
v___x_14_ = lean_apply_1(v_inst_3_, v___x_13_);
v___x_15_ = lean_apply_1(v_inst_4_, v___x_14_);
return v___x_15_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_castDef___redArg___boxed(lean_object* v_inst_16_, lean_object* v_inst_17_, lean_object* v_x_18_){
_start:
{
lean_object* v_res_19_; 
v_res_19_ = lp_mathlib_Int_castDef___redArg(v_inst_16_, v_inst_17_, v_x_18_);
lean_dec(v_x_18_);
return v_res_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_castDef(lean_object* v_R_20_, lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_x_23_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = lp_mathlib_Int_castDef___redArg(v_inst_21_, v_inst_22_, v_x_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_castDef___boxed(lean_object* v_R_25_, lean_object* v_inst_26_, lean_object* v_inst_27_, lean_object* v_x_28_){
_start:
{
lean_object* v_res_29_; 
v_res_29_ = lp_mathlib_Int_castDef(v_R_25_, v_inst_26_, v_inst_27_, v_x_28_);
lean_dec(v_x_28_);
return v_res_29_;
}
}
static lean_object* _init_lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__12(void){
_start:
{
lean_object* v___x_56_; lean_object* v___x_57_; 
v___x_56_ = ((lean_object*)(lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__10));
v___x_57_ = l_Lean_mkAtom(v___x_56_);
return v___x_57_;
}
}
static lean_object* _init_lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__13(void){
_start:
{
lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; 
v___x_58_ = lean_obj_once(&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__12, &lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__12_once, _init_lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__12);
v___x_59_ = ((lean_object*)(lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__5));
v___x_60_ = lean_array_push(v___x_59_, v___x_58_);
return v___x_60_;
}
}
static lean_object* _init_lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__15(void){
_start:
{
lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; 
v___x_65_ = ((lean_object*)(lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__14));
v___x_66_ = lean_obj_once(&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__13, &lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__13_once, _init_lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__13);
v___x_67_ = lean_array_push(v___x_66_, v___x_65_);
return v___x_67_;
}
}
static lean_object* _init_lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__16(void){
_start:
{
lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; 
v___x_68_ = lean_obj_once(&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__15, &lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__15_once, _init_lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__15);
v___x_69_ = ((lean_object*)(lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__11));
v___x_70_ = lean_box(2);
v___x_71_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_71_, 0, v___x_70_);
lean_ctor_set(v___x_71_, 1, v___x_69_);
lean_ctor_set(v___x_71_, 2, v___x_68_);
return v___x_71_;
}
}
static lean_object* _init_lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__17(void){
_start:
{
lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; 
v___x_72_ = lean_obj_once(&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__16, &lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__16_once, _init_lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__16);
v___x_73_ = ((lean_object*)(lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__5));
v___x_74_ = lean_array_push(v___x_73_, v___x_72_);
return v___x_74_;
}
}
static lean_object* _init_lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__19(void){
_start:
{
lean_object* v___x_76_; lean_object* v___x_77_; 
v___x_76_ = ((lean_object*)(lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__18));
v___x_77_ = l_Lean_mkAtom(v___x_76_);
return v___x_77_;
}
}
static lean_object* _init_lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__20(void){
_start:
{
lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; 
v___x_78_ = lean_obj_once(&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__19, &lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__19_once, _init_lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__19);
v___x_79_ = lean_obj_once(&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__17, &lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__17_once, _init_lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__17);
v___x_80_ = lean_array_push(v___x_79_, v___x_78_);
return v___x_80_;
}
}
static lean_object* _init_lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__24(void){
_start:
{
lean_object* v___x_88_; lean_object* v___x_89_; 
v___x_88_ = ((lean_object*)(lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__23));
v___x_89_ = l_Lean_mkAtom(v___x_88_);
return v___x_89_;
}
}
static lean_object* _init_lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__25(void){
_start:
{
lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; 
v___x_90_ = lean_obj_once(&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__24, &lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__24_once, _init_lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__24);
v___x_91_ = ((lean_object*)(lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__5));
v___x_92_ = lean_array_push(v___x_91_, v___x_90_);
return v___x_92_;
}
}
static lean_object* _init_lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__26(void){
_start:
{
lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; 
v___x_93_ = lean_obj_once(&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__25, &lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__25_once, _init_lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__25);
v___x_94_ = ((lean_object*)(lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__22));
v___x_95_ = lean_box(2);
v___x_96_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_96_, 0, v___x_95_);
lean_ctor_set(v___x_96_, 1, v___x_94_);
lean_ctor_set(v___x_96_, 2, v___x_93_);
return v___x_96_;
}
}
static lean_object* _init_lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__27(void){
_start:
{
lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; 
v___x_97_ = lean_obj_once(&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__26, &lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__26_once, _init_lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__26);
v___x_98_ = lean_obj_once(&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__20, &lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__20_once, _init_lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__20);
v___x_99_ = lean_array_push(v___x_98_, v___x_97_);
return v___x_99_;
}
}
static lean_object* _init_lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__28(void){
_start:
{
lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; 
v___x_100_ = lean_obj_once(&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__27, &lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__27_once, _init_lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__27);
v___x_101_ = ((lean_object*)(lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__9));
v___x_102_ = lean_box(2);
v___x_103_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_103_, 0, v___x_102_);
lean_ctor_set(v___x_103_, 1, v___x_101_);
lean_ctor_set(v___x_103_, 2, v___x_100_);
return v___x_103_;
}
}
static lean_object* _init_lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__29(void){
_start:
{
lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; 
v___x_104_ = lean_obj_once(&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__28, &lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__28_once, _init_lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__28);
v___x_105_ = ((lean_object*)(lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__5));
v___x_106_ = lean_array_push(v___x_105_, v___x_104_);
return v___x_106_;
}
}
static lean_object* _init_lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__30(void){
_start:
{
lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; 
v___x_107_ = lean_obj_once(&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__29, &lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__29_once, _init_lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__29);
v___x_108_ = ((lean_object*)(lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__7));
v___x_109_ = lean_box(2);
v___x_110_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_110_, 0, v___x_109_);
lean_ctor_set(v___x_110_, 1, v___x_108_);
lean_ctor_set(v___x_110_, 2, v___x_107_);
return v___x_110_;
}
}
static lean_object* _init_lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__31(void){
_start:
{
lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; 
v___x_111_ = lean_obj_once(&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__30, &lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__30_once, _init_lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__30);
v___x_112_ = ((lean_object*)(lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__5));
v___x_113_ = lean_array_push(v___x_112_, v___x_111_);
return v___x_113_;
}
}
static lean_object* _init_lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__32(void){
_start:
{
lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; 
v___x_114_ = lean_obj_once(&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__31, &lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__31_once, _init_lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__31);
v___x_115_ = ((lean_object*)(lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__4));
v___x_116_ = lean_box(2);
v___x_117_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_117_, 0, v___x_116_);
lean_ctor_set(v___x_117_, 1, v___x_115_);
lean_ctor_set(v___x_117_, 2, v___x_114_);
return v___x_117_;
}
}
static lean_object* _init_lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam(void){
_start:
{
lean_object* v___x_118_; 
v___x_118_ = lean_obj_once(&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__32, &lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__32_once, _init_lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__32);
return v___x_118_;
}
}
static lean_object* _init_lp_mathlib_AddGroupWithOne_intCast__negSucc___autoParam(void){
_start:
{
lean_object* v___x_119_; 
v___x_119_ = lean_obj_once(&lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__32, &lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__32_once, _init_lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam___closed__32);
return v___x_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddGroupWithOne_toAddGroup___redArg(lean_object* v_self_120_){
_start:
{
lean_object* v_toAddMonoidWithOne_121_; lean_object* v_toNeg_122_; lean_object* v_toSub_123_; lean_object* v_toZSMul_124_; lean_object* v_toAddMonoid_125_; lean_object* v___x_126_; 
v_toAddMonoidWithOne_121_ = lean_ctor_get(v_self_120_, 1);
v_toNeg_122_ = lean_ctor_get(v_self_120_, 2);
v_toSub_123_ = lean_ctor_get(v_self_120_, 3);
v_toZSMul_124_ = lean_ctor_get(v_self_120_, 4);
v_toAddMonoid_125_ = lean_ctor_get(v_toAddMonoidWithOne_121_, 1);
lean_inc(v_toZSMul_124_);
lean_inc(v_toSub_123_);
lean_inc(v_toNeg_122_);
lean_inc_ref(v_toAddMonoid_125_);
v___x_126_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_126_, 0, v_toAddMonoid_125_);
lean_ctor_set(v___x_126_, 1, v_toNeg_122_);
lean_ctor_set(v___x_126_, 2, v_toSub_123_);
lean_ctor_set(v___x_126_, 3, v_toZSMul_124_);
return v___x_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddGroupWithOne_toAddGroup___redArg___boxed(lean_object* v_self_127_){
_start:
{
lean_object* v_res_128_; 
v_res_128_ = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(v_self_127_);
lean_dec_ref(v_self_127_);
return v_res_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddGroupWithOne_toAddGroup(lean_object* v_R_129_, lean_object* v_self_130_){
_start:
{
lean_object* v___x_131_; 
v___x_131_ = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(v_self_130_);
return v___x_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddGroupWithOne_toAddGroup___boxed(lean_object* v_R_132_, lean_object* v_self_133_){
_start:
{
lean_object* v_res_134_; 
v_res_134_ = lp_mathlib_AddGroupWithOne_toAddGroup(v_R_132_, v_self_133_);
lean_dec_ref(v_self_133_);
return v_res_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(lean_object* v_self_135_){
_start:
{
lean_object* v_toAddCommGroup_136_; lean_object* v_toIntCast_137_; lean_object* v_toNatCast_138_; lean_object* v_toOne_139_; lean_object* v_toAddMonoid_140_; lean_object* v_toNeg_141_; lean_object* v_toSub_142_; lean_object* v_toZSMul_143_; lean_object* v___x_144_; lean_object* v___x_145_; 
v_toAddCommGroup_136_ = lean_ctor_get(v_self_135_, 0);
v_toIntCast_137_ = lean_ctor_get(v_self_135_, 1);
v_toNatCast_138_ = lean_ctor_get(v_self_135_, 2);
v_toOne_139_ = lean_ctor_get(v_self_135_, 3);
v_toAddMonoid_140_ = lean_ctor_get(v_toAddCommGroup_136_, 0);
v_toNeg_141_ = lean_ctor_get(v_toAddCommGroup_136_, 1);
v_toSub_142_ = lean_ctor_get(v_toAddCommGroup_136_, 2);
v_toZSMul_143_ = lean_ctor_get(v_toAddCommGroup_136_, 3);
lean_inc(v_toOne_139_);
lean_inc_ref(v_toAddMonoid_140_);
lean_inc(v_toNatCast_138_);
v___x_144_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_144_, 0, v_toNatCast_138_);
lean_ctor_set(v___x_144_, 1, v_toAddMonoid_140_);
lean_ctor_set(v___x_144_, 2, v_toOne_139_);
lean_inc(v_toZSMul_143_);
lean_inc(v_toSub_142_);
lean_inc(v_toNeg_141_);
lean_inc(v_toIntCast_137_);
v___x_145_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_145_, 0, v_toIntCast_137_);
lean_ctor_set(v___x_145_, 1, v___x_144_);
lean_ctor_set(v___x_145_, 2, v_toNeg_141_);
lean_ctor_set(v___x_145_, 3, v_toSub_142_);
lean_ctor_set(v___x_145_, 4, v_toZSMul_143_);
return v___x_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg___boxed(lean_object* v_self_146_){
_start:
{
lean_object* v_res_147_; 
v_res_147_ = lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(v_self_146_);
lean_dec_ref(v_self_146_);
return v_res_147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne(lean_object* v_R_148_, lean_object* v_self_149_){
_start:
{
lean_object* v___x_150_; 
v___x_150_ = lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(v_self_149_);
return v___x_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___boxed(lean_object* v_R_151_, lean_object* v_self_152_){
_start:
{
lean_object* v_res_153_; 
v_res_153_ = lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne(v_R_151_, v_self_152_);
lean_dec_ref(v_self_152_);
return v_res_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroupWithOne_toAddCommMonoidWithOne___redArg(lean_object* v_self_154_){
_start:
{
lean_object* v_toAddCommGroup_155_; lean_object* v_toNatCast_156_; lean_object* v_toOne_157_; lean_object* v_toAddMonoid_158_; lean_object* v___x_159_; 
v_toAddCommGroup_155_ = lean_ctor_get(v_self_154_, 0);
v_toNatCast_156_ = lean_ctor_get(v_self_154_, 2);
v_toOne_157_ = lean_ctor_get(v_self_154_, 3);
v_toAddMonoid_158_ = lean_ctor_get(v_toAddCommGroup_155_, 0);
lean_inc(v_toOne_157_);
lean_inc_ref(v_toAddMonoid_158_);
lean_inc(v_toNatCast_156_);
v___x_159_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_159_, 0, v_toNatCast_156_);
lean_ctor_set(v___x_159_, 1, v_toAddMonoid_158_);
lean_ctor_set(v___x_159_, 2, v_toOne_157_);
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroupWithOne_toAddCommMonoidWithOne___redArg___boxed(lean_object* v_self_160_){
_start:
{
lean_object* v_res_161_; 
v_res_161_ = lp_mathlib_AddCommGroupWithOne_toAddCommMonoidWithOne___redArg(v_self_160_);
lean_dec_ref(v_self_160_);
return v_res_161_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroupWithOne_toAddCommMonoidWithOne(lean_object* v_R_162_, lean_object* v_self_163_){
_start:
{
lean_object* v___x_164_; 
v___x_164_ = lp_mathlib_AddCommGroupWithOne_toAddCommMonoidWithOne___redArg(v_self_163_);
return v___x_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroupWithOne_toAddCommMonoidWithOne___boxed(lean_object* v_R_165_, lean_object* v_self_166_){
_start:
{
lean_object* v_res_167_; 
v_res_167_ = lp_mathlib_AddCommGroupWithOne_toAddCommMonoidWithOne(v_R_165_, v_self_166_);
lean_dec_ref(v_self_166_);
return v_res_167_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Int_Cast_Defs(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam = _init_lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam();
lean_mark_persistent(lp_mathlib_AddGroupWithOne_intCast__ofNat___autoParam);
lp_mathlib_AddGroupWithOne_intCast__negSucc___autoParam = _init_lp_mathlib_AddGroupWithOne_intCast__negSucc___autoParam();
lean_mark_persistent(lp_mathlib_AddGroupWithOne_intCast__negSucc___autoParam);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Cast_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Int_Cast_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Cast_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Int_Cast_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Int_Cast_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
