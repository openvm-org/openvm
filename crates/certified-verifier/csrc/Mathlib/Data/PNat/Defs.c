// Lean compiler output
// Module: Mathlib.Data.PNat.Defs
// Imports: public import Init public meta import Init public import Mathlib.Data.Int.Order.Basic public import Mathlib.Data.Nat.Basic public import Mathlib.Data.PNat.Notation public import Mathlib.Order.Basic public import Mathlib.Tactic.Coe public import Mathlib.Tactic.Lift
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
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
lean_object* lean_nat_pred(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* lp_mathlib_instDecidableEqPNat___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderPNat___aux__9(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderPNat___aux__9___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderPNat___aux__11(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderPNat___aux__11___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderPNat___aux__13(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderPNat___aux__13___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderPNat___aux__16(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderPNat___aux__16___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderPNat___aux__18(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderPNat___aux__18___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderPNat___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderPNat___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderPNat___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderPNat___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderPNat___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderPNat___lam__2___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_instLinearOrderPNat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instLinearOrderPNat___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instLinearOrderPNat___closed__0 = (const lean_object*)&lp_mathlib_instLinearOrderPNat___closed__0_value;
static const lean_closure_object lp_mathlib_instLinearOrderPNat___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instLinearOrderPNat___lam__1___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instLinearOrderPNat___closed__1 = (const lean_object*)&lp_mathlib_instLinearOrderPNat___closed__1_value;
static const lean_closure_object lp_mathlib_instLinearOrderPNat___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instLinearOrderPNat___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instLinearOrderPNat___closed__2 = (const lean_object*)&lp_mathlib_instLinearOrderPNat___closed__2_value;
static const lean_ctor_object lp_mathlib_instLinearOrderPNat___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_instLinearOrderPNat___closed__3 = (const lean_object*)&lp_mathlib_instLinearOrderPNat___closed__3_value;
static lean_once_cell_t lp_mathlib_instLinearOrderPNat___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instLinearOrderPNat___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderPNat;
LEAN_EXPORT lean_object* lp_mathlib_instOnePNat;
LEAN_EXPORT lean_object* lp_mathlib_instOfNatPNatOfNeZeroNat___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOfNatPNatOfNeZeroNat___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOfNatPNatOfNeZeroNat(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOfNatPNatOfNeZeroNat___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_natPred(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_natPred___boxed(lean_object*);
static const lean_string_object lp_mathlib_Nat_toPNat___auto__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Nat_toPNat___auto__1___closed__0 = (const lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__0_value;
static const lean_string_object lp_mathlib_Nat_toPNat___auto__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Nat_toPNat___auto__1___closed__1 = (const lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__1_value;
static const lean_string_object lp_mathlib_Nat_toPNat___auto__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Nat_toPNat___auto__1___closed__2 = (const lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__2_value;
static const lean_string_object lp_mathlib_Nat_toPNat___auto__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Nat_toPNat___auto__1___closed__3 = (const lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Nat_toPNat___auto__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Nat_toPNat___auto__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Nat_toPNat___auto__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Nat_toPNat___auto__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_Nat_toPNat___auto__1___closed__4 = (const lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__4_value;
static const lean_array_object lp_mathlib_Nat_toPNat___auto__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Nat_toPNat___auto__1___closed__5 = (const lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__5_value;
static const lean_string_object lp_mathlib_Nat_toPNat___auto__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_Nat_toPNat___auto__1___closed__6 = (const lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__6_value;
static const lean_ctor_object lp_mathlib_Nat_toPNat___auto__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Nat_toPNat___auto__1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Nat_toPNat___auto__1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__7_value_aux_1),((lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Nat_toPNat___auto__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__7_value_aux_2),((lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_Nat_toPNat___auto__1___closed__7 = (const lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__7_value;
static const lean_string_object lp_mathlib_Nat_toPNat___auto__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Nat_toPNat___auto__1___closed__8 = (const lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__8_value;
static const lean_ctor_object lp_mathlib_Nat_toPNat___auto__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Nat_toPNat___auto__1___closed__9 = (const lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__9_value;
static const lean_string_object lp_mathlib_Nat_toPNat___auto__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "decide"};
static const lean_object* lp_mathlib_Nat_toPNat___auto__1___closed__10 = (const lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__10_value;
static const lean_ctor_object lp_mathlib_Nat_toPNat___auto__1___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Nat_toPNat___auto__1___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Nat_toPNat___auto__1___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__11_value_aux_1),((lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Nat_toPNat___auto__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__11_value_aux_2),((lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(53, 158, 1, 232, 101, 200, 191, 197)}};
static const lean_object* lp_mathlib_Nat_toPNat___auto__1___closed__11 = (const lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__11_value;
static lean_once_cell_t lp_mathlib_Nat_toPNat___auto__1___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Nat_toPNat___auto__1___closed__12;
static lean_once_cell_t lp_mathlib_Nat_toPNat___auto__1___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Nat_toPNat___auto__1___closed__13;
static const lean_string_object lp_mathlib_Nat_toPNat___auto__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "optConfig"};
static const lean_object* lp_mathlib_Nat_toPNat___auto__1___closed__14 = (const lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__14_value;
static const lean_ctor_object lp_mathlib_Nat_toPNat___auto__1___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Nat_toPNat___auto__1___closed__15_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__15_value_aux_0),((lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Nat_toPNat___auto__1___closed__15_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__15_value_aux_1),((lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Nat_toPNat___auto__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__15_value_aux_2),((lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(137, 208, 10, 74, 108, 50, 106, 48)}};
static const lean_object* lp_mathlib_Nat_toPNat___auto__1___closed__15 = (const lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__15_value;
static const lean_ctor_object lp_mathlib_Nat_toPNat___auto__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__9_value),((lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__5_value)}};
static const lean_object* lp_mathlib_Nat_toPNat___auto__1___closed__16 = (const lean_object*)&lp_mathlib_Nat_toPNat___auto__1___closed__16_value;
static lean_once_cell_t lp_mathlib_Nat_toPNat___auto__1___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Nat_toPNat___auto__1___closed__17;
static lean_once_cell_t lp_mathlib_Nat_toPNat___auto__1___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Nat_toPNat___auto__1___closed__18;
static lean_once_cell_t lp_mathlib_Nat_toPNat___auto__1___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Nat_toPNat___auto__1___closed__19;
static lean_once_cell_t lp_mathlib_Nat_toPNat___auto__1___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Nat_toPNat___auto__1___closed__20;
static lean_once_cell_t lp_mathlib_Nat_toPNat___auto__1___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Nat_toPNat___auto__1___closed__21;
static lean_once_cell_t lp_mathlib_Nat_toPNat___auto__1___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Nat_toPNat___auto__1___closed__22;
static lean_once_cell_t lp_mathlib_Nat_toPNat___auto__1___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Nat_toPNat___auto__1___closed__23;
static lean_once_cell_t lp_mathlib_Nat_toPNat___auto__1___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Nat_toPNat___auto__1___closed__24;
static lean_once_cell_t lp_mathlib_Nat_toPNat___auto__1___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Nat_toPNat___auto__1___closed__25;
static lean_once_cell_t lp_mathlib_Nat_toPNat___auto__1___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Nat_toPNat___auto__1___closed__26;
LEAN_EXPORT lean_object* lp_mathlib_Nat_toPNat___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Nat_toPNat___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_toPNat___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_toPNat(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_toPNat___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_succPNat(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_succPNat___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_toPNat_x27(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_toPNat_x27___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_instInhabited;
LEAN_EXPORT lean_object* lp_mathlib_PNat_instWellFoundedRelation;
LEAN_EXPORT lean_object* lp_mathlib_PNat_strongInductionOn___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_strongInductionOn___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_strongInductionOn(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_modDivAux(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_modDivAux___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_modDiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_modDiv___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_mod(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_mod___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_div(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_div___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_divExact(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_divExact___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderPNat___aux__9(lean_object* v_a_1_, lean_object* v_a_2_){
_start:
{
uint8_t v___x_3_; 
v___x_3_ = lean_nat_dec_le(v_a_1_, v_a_2_);
if (v___x_3_ == 0)
{
lean_inc(v_a_2_);
return v_a_2_;
}
else
{
lean_inc(v_a_1_);
return v_a_1_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderPNat___aux__9___boxed(lean_object* v_a_4_, lean_object* v_a_5_){
_start:
{
lean_object* v_res_6_; 
v_res_6_ = lp_mathlib_instLinearOrderPNat___aux__9(v_a_4_, v_a_5_);
lean_dec(v_a_5_);
lean_dec(v_a_4_);
return v_res_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderPNat___aux__11(lean_object* v_a_7_, lean_object* v_a_8_){
_start:
{
uint8_t v___x_9_; 
v___x_9_ = lean_nat_dec_le(v_a_7_, v_a_8_);
if (v___x_9_ == 0)
{
lean_inc(v_a_7_);
return v_a_7_;
}
else
{
lean_inc(v_a_8_);
return v_a_8_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderPNat___aux__11___boxed(lean_object* v_a_10_, lean_object* v_a_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_mathlib_instLinearOrderPNat___aux__11(v_a_10_, v_a_11_);
lean_dec(v_a_11_);
lean_dec(v_a_10_);
return v_res_12_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderPNat___aux__13(lean_object* v_a_13_, lean_object* v_b_14_){
_start:
{
uint8_t v___x_15_; 
v___x_15_ = lean_nat_dec_lt(v_a_13_, v_b_14_);
if (v___x_15_ == 0)
{
uint8_t v___x_16_; 
v___x_16_ = lean_nat_dec_eq(v_a_13_, v_b_14_);
if (v___x_16_ == 0)
{
uint8_t v___x_17_; 
v___x_17_ = 2;
return v___x_17_;
}
else
{
uint8_t v___x_18_; 
v___x_18_ = 1;
return v___x_18_;
}
}
else
{
uint8_t v___x_19_; 
v___x_19_ = 0;
return v___x_19_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderPNat___aux__13___boxed(lean_object* v_a_20_, lean_object* v_b_21_){
_start:
{
uint8_t v_res_22_; lean_object* v_r_23_; 
v_res_22_ = lp_mathlib_instLinearOrderPNat___aux__13(v_a_20_, v_b_21_);
lean_dec(v_b_21_);
lean_dec(v_a_20_);
v_r_23_ = lean_box(v_res_22_);
return v_r_23_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderPNat___aux__16(lean_object* v_a_24_, lean_object* v_b_25_){
_start:
{
uint8_t v___x_26_; 
v___x_26_ = lean_nat_dec_le(v_a_24_, v_b_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderPNat___aux__16___boxed(lean_object* v_a_27_, lean_object* v_b_28_){
_start:
{
uint8_t v_res_29_; lean_object* v_r_30_; 
v_res_29_ = lp_mathlib_instLinearOrderPNat___aux__16(v_a_27_, v_b_28_);
lean_dec(v_b_28_);
lean_dec(v_a_27_);
v_r_30_ = lean_box(v_res_29_);
return v_r_30_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderPNat___aux__18(lean_object* v_a_31_, lean_object* v_b_32_){
_start:
{
uint8_t v___x_33_; 
v___x_33_ = lean_nat_dec_lt(v_a_31_, v_b_32_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderPNat___aux__18___boxed(lean_object* v_a_34_, lean_object* v_b_35_){
_start:
{
uint8_t v_res_36_; lean_object* v_r_37_; 
v_res_36_ = lp_mathlib_instLinearOrderPNat___aux__18(v_a_34_, v_b_35_);
lean_dec(v_b_35_);
lean_dec(v_a_34_);
v_r_37_ = lean_box(v_res_36_);
return v_r_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderPNat___lam__0(lean_object* v___y_38_, lean_object* v___y_39_){
_start:
{
uint8_t v___x_40_; 
v___x_40_ = lean_nat_dec_le(v___y_38_, v___y_39_);
if (v___x_40_ == 0)
{
lean_inc(v___y_39_);
return v___y_39_;
}
else
{
lean_inc(v___y_38_);
return v___y_38_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderPNat___lam__0___boxed(lean_object* v___y_41_, lean_object* v___y_42_){
_start:
{
lean_object* v_res_43_; 
v_res_43_ = lp_mathlib_instLinearOrderPNat___lam__0(v___y_41_, v___y_42_);
lean_dec(v___y_42_);
lean_dec(v___y_41_);
return v_res_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderPNat___lam__1(lean_object* v___y_44_, lean_object* v___y_45_){
_start:
{
uint8_t v___x_46_; 
v___x_46_ = lean_nat_dec_le(v___y_44_, v___y_45_);
if (v___x_46_ == 0)
{
lean_inc(v___y_44_);
return v___y_44_;
}
else
{
lean_inc(v___y_45_);
return v___y_45_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderPNat___lam__1___boxed(lean_object* v___y_47_, lean_object* v___y_48_){
_start:
{
lean_object* v_res_49_; 
v_res_49_ = lp_mathlib_instLinearOrderPNat___lam__1(v___y_47_, v___y_48_);
lean_dec(v___y_48_);
lean_dec(v___y_47_);
return v_res_49_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderPNat___lam__2(lean_object* v___y_50_, lean_object* v___y_51_){
_start:
{
uint8_t v___x_52_; 
v___x_52_ = lean_nat_dec_lt(v___y_50_, v___y_51_);
if (v___x_52_ == 0)
{
uint8_t v___x_53_; 
v___x_53_ = lean_nat_dec_eq(v___y_50_, v___y_51_);
if (v___x_53_ == 0)
{
uint8_t v___x_54_; 
v___x_54_ = 2;
return v___x_54_;
}
else
{
uint8_t v___x_55_; 
v___x_55_ = 1;
return v___x_55_;
}
}
else
{
uint8_t v___x_56_; 
v___x_56_ = 0;
return v___x_56_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderPNat___lam__2___boxed(lean_object* v___y_57_, lean_object* v___y_58_){
_start:
{
uint8_t v_res_59_; lean_object* v_r_60_; 
v_res_59_ = lp_mathlib_instLinearOrderPNat___lam__2(v___y_57_, v___y_58_);
lean_dec(v___y_58_);
lean_dec(v___y_57_);
v_r_60_ = lean_box(v_res_59_);
return v_r_60_;
}
}
static lean_object* _init_lp_mathlib_instLinearOrderPNat___closed__4(void){
_start:
{
lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___f_70_; lean_object* v___f_71_; lean_object* v___f_72_; lean_object* v___x_73_; lean_object* v___x_74_; 
v___x_67_ = lean_alloc_closure((void*)(lp_mathlib_instLinearOrderPNat___aux__18___boxed), 2, 0);
v___x_68_ = lean_alloc_closure((void*)(lp_mathlib_instDecidableEqPNat___boxed), 2, 0);
v___x_69_ = lean_alloc_closure((void*)(lp_mathlib_instLinearOrderPNat___aux__16___boxed), 2, 0);
v___f_70_ = ((lean_object*)(lp_mathlib_instLinearOrderPNat___closed__2));
v___f_71_ = ((lean_object*)(lp_mathlib_instLinearOrderPNat___closed__1));
v___f_72_ = ((lean_object*)(lp_mathlib_instLinearOrderPNat___closed__0));
v___x_73_ = ((lean_object*)(lp_mathlib_instLinearOrderPNat___closed__3));
v___x_74_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_74_, 0, v___x_73_);
lean_ctor_set(v___x_74_, 1, v___f_72_);
lean_ctor_set(v___x_74_, 2, v___f_71_);
lean_ctor_set(v___x_74_, 3, v___f_70_);
lean_ctor_set(v___x_74_, 4, v___x_69_);
lean_ctor_set(v___x_74_, 5, v___x_68_);
lean_ctor_set(v___x_74_, 6, v___x_67_);
return v___x_74_;
}
}
static lean_object* _init_lp_mathlib_instLinearOrderPNat(void){
_start:
{
lean_object* v___x_75_; 
v___x_75_ = lean_obj_once(&lp_mathlib_instLinearOrderPNat___closed__4, &lp_mathlib_instLinearOrderPNat___closed__4_once, _init_lp_mathlib_instLinearOrderPNat___closed__4);
return v___x_75_;
}
}
static lean_object* _init_lp_mathlib_instOnePNat(void){
_start:
{
lean_object* v___x_76_; 
v___x_76_ = lean_unsigned_to_nat(1u);
return v___x_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOfNatPNatOfNeZeroNat___redArg(lean_object* v_n_77_){
_start:
{
lean_inc(v_n_77_);
return v_n_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOfNatPNatOfNeZeroNat___redArg___boxed(lean_object* v_n_78_){
_start:
{
lean_object* v_res_79_; 
v_res_79_ = lp_mathlib_instOfNatPNatOfNeZeroNat___redArg(v_n_78_);
lean_dec(v_n_78_);
return v_res_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOfNatPNatOfNeZeroNat(lean_object* v_n_80_, lean_object* v_inst_81_){
_start:
{
lean_inc(v_n_80_);
return v_n_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOfNatPNatOfNeZeroNat___boxed(lean_object* v_n_82_, lean_object* v_inst_83_){
_start:
{
lean_object* v_res_84_; 
v_res_84_ = lp_mathlib_instOfNatPNatOfNeZeroNat(v_n_82_, v_inst_83_);
lean_dec(v_n_82_);
return v_res_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_natPred(lean_object* v_i_85_){
_start:
{
lean_object* v___x_86_; lean_object* v___x_87_; 
v___x_86_ = lean_unsigned_to_nat(1u);
v___x_87_ = lean_nat_sub(v_i_85_, v___x_86_);
return v___x_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_natPred___boxed(lean_object* v_i_88_){
_start:
{
lean_object* v_res_89_; 
v_res_89_ = lp_mathlib_PNat_natPred(v_i_88_);
lean_dec(v_i_88_);
return v_res_89_;
}
}
static lean_object* _init_lp_mathlib_Nat_toPNat___auto__1___closed__12(void){
_start:
{
lean_object* v___x_116_; lean_object* v___x_117_; 
v___x_116_ = ((lean_object*)(lp_mathlib_Nat_toPNat___auto__1___closed__10));
v___x_117_ = l_Lean_mkAtom(v___x_116_);
return v___x_117_;
}
}
static lean_object* _init_lp_mathlib_Nat_toPNat___auto__1___closed__13(void){
_start:
{
lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; 
v___x_118_ = lean_obj_once(&lp_mathlib_Nat_toPNat___auto__1___closed__12, &lp_mathlib_Nat_toPNat___auto__1___closed__12_once, _init_lp_mathlib_Nat_toPNat___auto__1___closed__12);
v___x_119_ = ((lean_object*)(lp_mathlib_Nat_toPNat___auto__1___closed__5));
v___x_120_ = lean_array_push(v___x_119_, v___x_118_);
return v___x_120_;
}
}
static lean_object* _init_lp_mathlib_Nat_toPNat___auto__1___closed__17(void){
_start:
{
lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; 
v___x_131_ = ((lean_object*)(lp_mathlib_Nat_toPNat___auto__1___closed__16));
v___x_132_ = ((lean_object*)(lp_mathlib_Nat_toPNat___auto__1___closed__5));
v___x_133_ = lean_array_push(v___x_132_, v___x_131_);
return v___x_133_;
}
}
static lean_object* _init_lp_mathlib_Nat_toPNat___auto__1___closed__18(void){
_start:
{
lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; 
v___x_134_ = lean_obj_once(&lp_mathlib_Nat_toPNat___auto__1___closed__17, &lp_mathlib_Nat_toPNat___auto__1___closed__17_once, _init_lp_mathlib_Nat_toPNat___auto__1___closed__17);
v___x_135_ = ((lean_object*)(lp_mathlib_Nat_toPNat___auto__1___closed__15));
v___x_136_ = lean_box(2);
v___x_137_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_137_, 0, v___x_136_);
lean_ctor_set(v___x_137_, 1, v___x_135_);
lean_ctor_set(v___x_137_, 2, v___x_134_);
return v___x_137_;
}
}
static lean_object* _init_lp_mathlib_Nat_toPNat___auto__1___closed__19(void){
_start:
{
lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; 
v___x_138_ = lean_obj_once(&lp_mathlib_Nat_toPNat___auto__1___closed__18, &lp_mathlib_Nat_toPNat___auto__1___closed__18_once, _init_lp_mathlib_Nat_toPNat___auto__1___closed__18);
v___x_139_ = lean_obj_once(&lp_mathlib_Nat_toPNat___auto__1___closed__13, &lp_mathlib_Nat_toPNat___auto__1___closed__13_once, _init_lp_mathlib_Nat_toPNat___auto__1___closed__13);
v___x_140_ = lean_array_push(v___x_139_, v___x_138_);
return v___x_140_;
}
}
static lean_object* _init_lp_mathlib_Nat_toPNat___auto__1___closed__20(void){
_start:
{
lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; 
v___x_141_ = lean_obj_once(&lp_mathlib_Nat_toPNat___auto__1___closed__19, &lp_mathlib_Nat_toPNat___auto__1___closed__19_once, _init_lp_mathlib_Nat_toPNat___auto__1___closed__19);
v___x_142_ = ((lean_object*)(lp_mathlib_Nat_toPNat___auto__1___closed__11));
v___x_143_ = lean_box(2);
v___x_144_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_144_, 0, v___x_143_);
lean_ctor_set(v___x_144_, 1, v___x_142_);
lean_ctor_set(v___x_144_, 2, v___x_141_);
return v___x_144_;
}
}
static lean_object* _init_lp_mathlib_Nat_toPNat___auto__1___closed__21(void){
_start:
{
lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; 
v___x_145_ = lean_obj_once(&lp_mathlib_Nat_toPNat___auto__1___closed__20, &lp_mathlib_Nat_toPNat___auto__1___closed__20_once, _init_lp_mathlib_Nat_toPNat___auto__1___closed__20);
v___x_146_ = ((lean_object*)(lp_mathlib_Nat_toPNat___auto__1___closed__5));
v___x_147_ = lean_array_push(v___x_146_, v___x_145_);
return v___x_147_;
}
}
static lean_object* _init_lp_mathlib_Nat_toPNat___auto__1___closed__22(void){
_start:
{
lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; 
v___x_148_ = lean_obj_once(&lp_mathlib_Nat_toPNat___auto__1___closed__21, &lp_mathlib_Nat_toPNat___auto__1___closed__21_once, _init_lp_mathlib_Nat_toPNat___auto__1___closed__21);
v___x_149_ = ((lean_object*)(lp_mathlib_Nat_toPNat___auto__1___closed__9));
v___x_150_ = lean_box(2);
v___x_151_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_151_, 0, v___x_150_);
lean_ctor_set(v___x_151_, 1, v___x_149_);
lean_ctor_set(v___x_151_, 2, v___x_148_);
return v___x_151_;
}
}
static lean_object* _init_lp_mathlib_Nat_toPNat___auto__1___closed__23(void){
_start:
{
lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; 
v___x_152_ = lean_obj_once(&lp_mathlib_Nat_toPNat___auto__1___closed__22, &lp_mathlib_Nat_toPNat___auto__1___closed__22_once, _init_lp_mathlib_Nat_toPNat___auto__1___closed__22);
v___x_153_ = ((lean_object*)(lp_mathlib_Nat_toPNat___auto__1___closed__5));
v___x_154_ = lean_array_push(v___x_153_, v___x_152_);
return v___x_154_;
}
}
static lean_object* _init_lp_mathlib_Nat_toPNat___auto__1___closed__24(void){
_start:
{
lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; 
v___x_155_ = lean_obj_once(&lp_mathlib_Nat_toPNat___auto__1___closed__23, &lp_mathlib_Nat_toPNat___auto__1___closed__23_once, _init_lp_mathlib_Nat_toPNat___auto__1___closed__23);
v___x_156_ = ((lean_object*)(lp_mathlib_Nat_toPNat___auto__1___closed__7));
v___x_157_ = lean_box(2);
v___x_158_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_158_, 0, v___x_157_);
lean_ctor_set(v___x_158_, 1, v___x_156_);
lean_ctor_set(v___x_158_, 2, v___x_155_);
return v___x_158_;
}
}
static lean_object* _init_lp_mathlib_Nat_toPNat___auto__1___closed__25(void){
_start:
{
lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; 
v___x_159_ = lean_obj_once(&lp_mathlib_Nat_toPNat___auto__1___closed__24, &lp_mathlib_Nat_toPNat___auto__1___closed__24_once, _init_lp_mathlib_Nat_toPNat___auto__1___closed__24);
v___x_160_ = ((lean_object*)(lp_mathlib_Nat_toPNat___auto__1___closed__5));
v___x_161_ = lean_array_push(v___x_160_, v___x_159_);
return v___x_161_;
}
}
static lean_object* _init_lp_mathlib_Nat_toPNat___auto__1___closed__26(void){
_start:
{
lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; 
v___x_162_ = lean_obj_once(&lp_mathlib_Nat_toPNat___auto__1___closed__25, &lp_mathlib_Nat_toPNat___auto__1___closed__25_once, _init_lp_mathlib_Nat_toPNat___auto__1___closed__25);
v___x_163_ = ((lean_object*)(lp_mathlib_Nat_toPNat___auto__1___closed__4));
v___x_164_ = lean_box(2);
v___x_165_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_165_, 0, v___x_164_);
lean_ctor_set(v___x_165_, 1, v___x_163_);
lean_ctor_set(v___x_165_, 2, v___x_162_);
return v___x_165_;
}
}
static lean_object* _init_lp_mathlib_Nat_toPNat___auto__1(void){
_start:
{
lean_object* v___x_166_; 
v___x_166_ = lean_obj_once(&lp_mathlib_Nat_toPNat___auto__1___closed__26, &lp_mathlib_Nat_toPNat___auto__1___closed__26_once, _init_lp_mathlib_Nat_toPNat___auto__1___closed__26);
return v___x_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_toPNat___redArg(lean_object* v_n_167_){
_start:
{
lean_inc(v_n_167_);
return v_n_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_toPNat___redArg___boxed(lean_object* v_n_168_){
_start:
{
lean_object* v_res_169_; 
v_res_169_ = lp_mathlib_Nat_toPNat___redArg(v_n_168_);
lean_dec(v_n_168_);
return v_res_169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_toPNat(lean_object* v_n_170_, lean_object* v_h_171_){
_start:
{
lean_inc(v_n_170_);
return v_n_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_toPNat___boxed(lean_object* v_n_172_, lean_object* v_h_173_){
_start:
{
lean_object* v_res_174_; 
v_res_174_ = lp_mathlib_Nat_toPNat(v_n_172_, v_h_173_);
lean_dec(v_n_172_);
return v_res_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_succPNat(lean_object* v_n_175_){
_start:
{
lean_object* v___x_176_; lean_object* v___x_177_; 
v___x_176_ = lean_unsigned_to_nat(1u);
v___x_177_ = lean_nat_add(v_n_175_, v___x_176_);
return v___x_177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_succPNat___boxed(lean_object* v_n_178_){
_start:
{
lean_object* v_res_179_; 
v_res_179_ = lp_mathlib_Nat_succPNat(v_n_178_);
lean_dec(v_n_178_);
return v_res_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_toPNat_x27(lean_object* v_n_180_){
_start:
{
lean_object* v___x_181_; lean_object* v___x_182_; 
v___x_181_ = lean_nat_pred(v_n_180_);
v___x_182_ = lp_mathlib_Nat_succPNat(v___x_181_);
lean_dec(v___x_181_);
return v___x_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_toPNat_x27___boxed(lean_object* v_n_183_){
_start:
{
lean_object* v_res_184_; 
v_res_184_ = lp_mathlib_Nat_toPNat_x27(v_n_183_);
lean_dec(v_n_183_);
return v_res_184_;
}
}
static lean_object* _init_lp_mathlib_PNat_instInhabited(void){
_start:
{
lean_object* v___x_185_; 
v___x_185_ = lean_unsigned_to_nat(1u);
return v___x_185_;
}
}
static lean_object* _init_lp_mathlib_PNat_instWellFoundedRelation(void){
_start:
{
lean_object* v___x_186_; 
v___x_186_ = lean_box(0);
return v___x_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_strongInductionOn___redArg(lean_object* v_n_187_, lean_object* v_x_188_){
_start:
{
lean_object* v___f_189_; lean_object* v___x_190_; 
lean_inc(v_x_188_);
v___f_189_ = lean_alloc_closure((void*)(lp_mathlib_PNat_strongInductionOn___redArg___lam__0), 3, 1);
lean_closure_set(v___f_189_, 0, v_x_188_);
v___x_190_ = lean_apply_2(v_x_188_, v_n_187_, v___f_189_);
return v___x_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_strongInductionOn___redArg___lam__0(lean_object* v_x_191_, lean_object* v_a_192_, lean_object* v_x_193_){
_start:
{
lean_object* v___x_194_; 
v___x_194_ = lp_mathlib_PNat_strongInductionOn___redArg(v_a_192_, v_x_191_);
return v___x_194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_strongInductionOn(lean_object* v_p_195_, lean_object* v_n_196_, lean_object* v_x_197_){
_start:
{
lean_object* v___x_198_; 
v___x_198_ = lp_mathlib_PNat_strongInductionOn___redArg(v_n_196_, v_x_197_);
return v___x_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_modDivAux(lean_object* v_x_199_, lean_object* v_x_200_, lean_object* v_x_201_){
_start:
{
lean_object* v_zero_202_; uint8_t v_isZero_203_; 
v_zero_202_ = lean_unsigned_to_nat(0u);
v_isZero_203_ = lean_nat_dec_eq(v_x_200_, v_zero_202_);
if (v_isZero_203_ == 1)
{
lean_object* v___x_204_; lean_object* v___x_205_; 
v___x_204_ = lean_nat_pred(v_x_201_);
lean_dec(v_x_201_);
v___x_205_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_205_, 0, v_x_199_);
lean_ctor_set(v___x_205_, 1, v___x_204_);
return v___x_205_;
}
else
{
lean_object* v_one_206_; lean_object* v_n_207_; lean_object* v___x_208_; lean_object* v___x_209_; 
lean_dec(v_x_199_);
v_one_206_ = lean_unsigned_to_nat(1u);
v_n_207_ = lean_nat_sub(v_x_200_, v_one_206_);
v___x_208_ = lean_nat_add(v_n_207_, v_one_206_);
lean_dec(v_n_207_);
v___x_209_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_209_, 0, v___x_208_);
lean_ctor_set(v___x_209_, 1, v_x_201_);
return v___x_209_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_modDivAux___boxed(lean_object* v_x_210_, lean_object* v_x_211_, lean_object* v_x_212_){
_start:
{
lean_object* v_res_213_; 
v_res_213_ = lp_mathlib_PNat_modDivAux(v_x_210_, v_x_211_, v_x_212_);
lean_dec(v_x_211_);
return v_res_213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_modDiv(lean_object* v_m_214_, lean_object* v_k_215_){
_start:
{
lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; 
v___x_216_ = lean_nat_mod(v_m_214_, v_k_215_);
v___x_217_ = lean_nat_div(v_m_214_, v_k_215_);
v___x_218_ = lp_mathlib_PNat_modDivAux(v_k_215_, v___x_216_, v___x_217_);
lean_dec(v___x_216_);
return v___x_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_modDiv___boxed(lean_object* v_m_219_, lean_object* v_k_220_){
_start:
{
lean_object* v_res_221_; 
v_res_221_ = lp_mathlib_PNat_modDiv(v_m_219_, v_k_220_);
lean_dec(v_m_219_);
return v_res_221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_mod(lean_object* v_m_222_, lean_object* v_k_223_){
_start:
{
lean_object* v___x_224_; lean_object* v_fst_225_; 
v___x_224_ = lp_mathlib_PNat_modDiv(v_m_222_, v_k_223_);
v_fst_225_ = lean_ctor_get(v___x_224_, 0);
lean_inc(v_fst_225_);
lean_dec_ref(v___x_224_);
return v_fst_225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_mod___boxed(lean_object* v_m_226_, lean_object* v_k_227_){
_start:
{
lean_object* v_res_228_; 
v_res_228_ = lp_mathlib_PNat_mod(v_m_226_, v_k_227_);
lean_dec(v_m_226_);
return v_res_228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_div(lean_object* v_m_229_, lean_object* v_k_230_){
_start:
{
lean_object* v___x_231_; lean_object* v_snd_232_; 
v___x_231_ = lp_mathlib_PNat_modDiv(v_m_229_, v_k_230_);
v_snd_232_ = lean_ctor_get(v___x_231_, 1);
lean_inc(v_snd_232_);
lean_dec_ref(v___x_231_);
return v_snd_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_div___boxed(lean_object* v_m_233_, lean_object* v_k_234_){
_start:
{
lean_object* v_res_235_; 
v_res_235_ = lp_mathlib_PNat_div(v_m_233_, v_k_234_);
lean_dec(v_m_233_);
return v_res_235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_divExact(lean_object* v_m_236_, lean_object* v_k_237_){
_start:
{
lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; 
v___x_238_ = lp_mathlib_PNat_div(v_m_236_, v_k_237_);
v___x_239_ = lean_unsigned_to_nat(1u);
v___x_240_ = lean_nat_add(v___x_238_, v___x_239_);
lean_dec(v___x_238_);
return v___x_240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_divExact___boxed(lean_object* v_m_241_, lean_object* v_k_242_){
_start:
{
lean_object* v_res_243_; 
v_res_243_ = lp_mathlib_PNat_divExact(v_m_241_, v_k_242_);
lean_dec(v_m_241_);
return v_res_243_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Order_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_PNat_Notation(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Coe(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Lift(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_PNat_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_PNat_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Coe(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Lift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_instLinearOrderPNat = _init_lp_mathlib_instLinearOrderPNat();
lean_mark_persistent(lp_mathlib_instLinearOrderPNat);
lp_mathlib_instOnePNat = _init_lp_mathlib_instOnePNat();
lean_mark_persistent(lp_mathlib_instOnePNat);
lp_mathlib_PNat_instInhabited = _init_lp_mathlib_PNat_instInhabited();
lean_mark_persistent(lp_mathlib_PNat_instInhabited);
lp_mathlib_PNat_instWellFoundedRelation = _init_lp_mathlib_PNat_instWellFoundedRelation();
lean_mark_persistent(lp_mathlib_PNat_instWellFoundedRelation);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_PNat_Defs(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Nat_toPNat___auto__1 = _init_lp_mathlib_Nat_toPNat___auto__1();
lean_mark_persistent(lp_mathlib_Nat_toPNat___auto__1);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Order_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_PNat_Notation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Coe(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Lift(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_PNat_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_PNat_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Coe(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Lift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_PNat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_PNat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_PNat_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
