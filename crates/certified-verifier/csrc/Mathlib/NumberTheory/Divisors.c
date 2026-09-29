// Lean compiler output
// Module: Mathlib.NumberTheory.Divisors
// Imports: public import Init public meta import Init public import Mathlib.Algebra.IsPrimePow public import Mathlib.Algebra.Order.BigOperators.Group.Finset public import Mathlib.Algebra.Order.Interval.Finset.SuccPred public import Mathlib.Algebra.Order.Ring.Int public import Mathlib.Algebra.Ring.CharZero public import Mathlib.Data.Finset.NatAntidiagonal public import Mathlib.Data.Nat.Cast.Order.Ring public import Mathlib.Data.Nat.PrimeFin public import Mathlib.Data.Nat.SuccPred public import Mathlib.Order.Interval.Finset.Nat
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
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_List_range_x27TR_go(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
uint8_t lean_int_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_abs(lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Int_ofNat___boxed(lean_object*);
uint8_t l_Nat_decidable__dvd(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_filter___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_map___redArg(lean_object*, lean_object*);
lean_object* l_Int_neg___boxed(lean_object*);
lean_object* lp_mathlib_Equiv_toEmbedding___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Embedding_trans___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lp_mathlib_Multiset_filterMap___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Embedding_prodMap___redArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesIdent(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ico___at___00Nat_divisors_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ico___at___00Nat_divisors_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Nat_divisors___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_divisors___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_divisors(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_properDivisors(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Icc___at___00Nat_divisorsAntidiagonal_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Icc___at___00Nat_divisorsAntidiagonal_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_divisorsAntidiagonal___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_divisorsAntidiagonal___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_divisorsAntidiagonal(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00Nat_divisorsAntidiagonalList_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00Nat_divisorsAntidiagonalList_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Nat_divisorsAntidiagonalList___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Nat_divisorsAntidiagonalList___closed__0 = (const lean_object*)&lp_mathlib_Nat_divisorsAntidiagonalList___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Nat_divisorsAntidiagonalList(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__2_value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "NumberTheory"};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__4_value),LEAN_SCALAR_PTR_LITERAL(116, 102, 119, 85, 237, 72, 2, 242)}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Divisors"};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__6_value),LEAN_SCALAR_PTR_LITERAL(199, 185, 5, 107, 30, 225, 168, 255)}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(186, 97, 159, 97, 151, 132, 211, 148)}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__8_value;
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Int"};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__8_value),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__9_value),LEAN_SCALAR_PTR_LITERAL(168, 252, 91, 105, 205, 140, 93, 152)}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "termNatCast"};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__11_value),LEAN_SCALAR_PTR_LITERAL(75, 218, 221, 7, 33, 193, 207, 161)}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__12_value;
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "natCast"};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__13_value)}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__12_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__14_value)}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__15_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__15_value;
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "Nat.castEmbedding"};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__6;
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "castEmbedding"};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__9_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(233, 103, 129, 126, 21, 54, 242, 186)}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__11_value;
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__13_value;
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "namedArgument"};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__15_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__15_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__15_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__15_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__15_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(226, 89, 129, 113, 173, 121, 169, 188)}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__15_value;
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__16_value;
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "R"};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__17_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__18;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__17_value),LEAN_SCALAR_PTR_LITERAL(10, 150, 1, 122, 163, 250, 19, 99)}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__19_value;
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ":="};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__20_value;
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 5, .m_data = "termℤ"};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__21_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__21_value),LEAN_SCALAR_PTR_LITERAL(233, 223, 72, 110, 220, 141, 14, 239)}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__22_value;
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "ℤ"};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__23 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__23_value;
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__24 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__24_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______unexpand__Nat__castEmbedding__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______unexpand__Nat__castEmbedding__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______unexpand__Nat__castEmbedding__1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______unexpand__Nat__castEmbedding__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______unexpand__Nat__castEmbedding__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______unexpand__Nat__castEmbedding__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______unexpand__Nat__castEmbedding__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______unexpand__Nat__castEmbedding__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______unexpand__Nat__castEmbedding__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNegNatCast___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "termNegNatCast"};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNegNatCast___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNegNatCast___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNegNatCast___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNegNatCast___closed__0_value),LEAN_SCALAR_PTR_LITERAL(63, 13, 214, 100, 8, 253, 143, 251)}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNegNatCast___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNegNatCast___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNegNatCast___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "negNatCast"};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNegNatCast___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNegNatCast___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNegNatCast___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNegNatCast___closed__2_value)}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNegNatCast___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNegNatCast___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNegNatCast___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNegNatCast___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNegNatCast___closed__3_value)}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNegNatCast___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNegNatCast___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNegNatCast = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNegNatCast___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "Function.Embedding.trans"};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__1;
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Function"};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Embedding"};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trans"};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(225, 8, 186, 189, 152, 89, 197, 12)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__5_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(25, 64, 244, 242, 24, 63, 210, 118)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__5_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(235, 185, 253, 90, 100, 93, 220, 150)}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__5_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__6_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__9_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__9_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__9_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__9_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(124, 9, 161, 194, 227, 100, 20, 110)}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__9_value;
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__11_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__11_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__11_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(41, 104, 206, 51, 21, 254, 100, 101)}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__11_value;
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__13_value;
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__14_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__15;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__9_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__16_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__16_value)}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__17_value;
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Finset"};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__18_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__18_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__19_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__19_value)}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__20_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__20_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__21_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__17_value),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__21_value)}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__22_value;
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "Equiv.toEmbedding"};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__23 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__23_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__24;
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Equiv"};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__25 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__25_value;
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "toEmbedding"};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__26 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__26_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__27_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__25_value),LEAN_SCALAR_PTR_LITERAL(0, 253, 123, 237, 128, 91, 245, 83)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__27_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__26_value),LEAN_SCALAR_PTR_LITERAL(85, 59, 221, 62, 175, 39, 190, 148)}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__27 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__27_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__27_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__28 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__28_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__28_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__29 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__29_value;
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Equiv.neg"};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__30 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__30_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__31;
static const lean_string_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "neg"};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__32 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__32_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__33_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__25_value),LEAN_SCALAR_PTR_LITERAL(0, 253, 123, 237, 128, 91, 245, 83)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__33_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__32_value),LEAN_SCALAR_PTR_LITERAL(119, 255, 106, 1, 182, 248, 177, 41)}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__33 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__33_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__33_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__34 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__34_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__34_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__35 = (const lean_object*)&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__35_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______unexpand__Function__Embedding__trans__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______unexpand__Function__Embedding__trans__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_neg___at___00Int_divisors_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Int_neg___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_neg___at___00Int_divisors_spec__1___closed__0 = (const lean_object*)&lp_mathlib_Equiv_neg___at___00Int_divisors_spec__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Equiv_neg___at___00Int_divisors_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_neg___at___00Int_divisors_spec__1___closed__0_value),((lean_object*)&lp_mathlib_Equiv_neg___at___00Int_divisors_spec__1___closed__0_value)}};
static const lean_object* lp_mathlib_Equiv_neg___at___00Int_divisors_spec__1___closed__1 = (const lean_object*)&lp_mathlib_Equiv_neg___at___00Int_divisors_spec__1___closed__1_value;
LEAN_EXPORT const lean_object* lp_mathlib_Equiv_neg___at___00Int_divisors_spec__1 = (const lean_object*)&lp_mathlib_Equiv_neg___at___00Int_divisors_spec__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Nat_castEmbedding___at___00Int_divisors_spec__0(lean_object*);
static const lean_closure_object lp_mathlib_Int_divisors___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_toEmbedding___redArg___lam__0, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Equiv_neg___at___00Int_divisors_spec__1___closed__1_value)} };
static const lean_object* lp_mathlib_Int_divisors___closed__0 = (const lean_object*)&lp_mathlib_Int_divisors___closed__0_value;
static lean_once_cell_t lp_mathlib_Int_divisors___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Int_divisors___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Int_divisors(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_divisors___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Nat_cast___at___00Nat_castEmbedding___at___00Int_divisors_spec__0_spec__0_spec__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Nat_castEmbedding___at___00Int_divisors_spec__0_spec__0(lean_object*);
static lean_once_cell_t lp_mathlib_Int_divisorsAntidiag___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Int_divisorsAntidiag___closed__0;
static lean_once_cell_t lp_mathlib_Int_divisorsAntidiag___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Int_divisorsAntidiag___closed__1;
static lean_once_cell_t lp_mathlib_Int_divisorsAntidiag___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Int_divisorsAntidiag___closed__2;
static lean_once_cell_t lp_mathlib_Int_divisorsAntidiag___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Int_divisorsAntidiag___closed__3;
static lean_once_cell_t lp_mathlib_Int_divisorsAntidiag___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Int_divisorsAntidiag___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Int_divisorsAntidiag(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_divisorsAntidiag___boxed(lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_divisorsAntidiag_match__1_splitter___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_divisorsAntidiag_match__1_splitter___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_divisorsAntidiag_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_divisorsAntidiag_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_divisorsAntidiag_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_divisorsAntidiag_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ico___at___00Nat_divisors_spec__0(lean_object* v_a_1_, lean_object* v_b_2_){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; lean_object* v___x_5_; lean_object* v___x_6_; lean_object* v___x_7_; 
v___x_3_ = lean_nat_sub(v_b_2_, v_a_1_);
v___x_4_ = lean_unsigned_to_nat(1u);
v___x_5_ = lean_nat_add(v_a_1_, v___x_3_);
v___x_6_ = lean_box(0);
v___x_7_ = l_List_range_x27TR_go(v___x_4_, v___x_3_, v___x_5_, v___x_6_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ico___at___00Nat_divisors_spec__0___boxed(lean_object* v_a_8_, lean_object* v_b_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_mathlib_Finset_Ico___at___00Nat_divisors_spec__0(v_a_8_, v_b_9_);
lean_dec(v_b_9_);
lean_dec(v_a_8_);
return v_res_10_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Nat_divisors___lam__0(lean_object* v_n_11_, lean_object* v_a_12_){
_start:
{
uint8_t v___x_13_; 
v___x_13_ = l_Nat_decidable__dvd(v_a_12_, v_n_11_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_divisors___lam__0___boxed(lean_object* v_n_14_, lean_object* v_a_15_){
_start:
{
uint8_t v_res_16_; lean_object* v_r_17_; 
v_res_16_ = lp_mathlib_Nat_divisors___lam__0(v_n_14_, v_a_15_);
lean_dec(v_a_15_);
lean_dec(v_n_14_);
v_r_17_ = lean_box(v_res_16_);
return v_r_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_divisors(lean_object* v_n_18_){
_start:
{
lean_object* v___f_19_; lean_object* v___x_20_; lean_object* v___x_21_; lean_object* v___x_22_; lean_object* v___x_23_; 
lean_inc(v_n_18_);
v___f_19_ = lean_alloc_closure((void*)(lp_mathlib_Nat_divisors___lam__0___boxed), 2, 1);
lean_closure_set(v___f_19_, 0, v_n_18_);
v___x_20_ = lean_unsigned_to_nat(1u);
v___x_21_ = lean_nat_add(v_n_18_, v___x_20_);
lean_dec(v_n_18_);
v___x_22_ = lp_mathlib_Finset_Ico___at___00Nat_divisors_spec__0(v___x_20_, v___x_21_);
lean_dec(v___x_21_);
v___x_23_ = lp_mathlib_Multiset_filter___redArg(v___f_19_, v___x_22_);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_properDivisors(lean_object* v_n_24_){
_start:
{
lean_object* v___f_25_; lean_object* v___x_26_; lean_object* v___x_27_; lean_object* v___x_28_; 
lean_inc(v_n_24_);
v___f_25_ = lean_alloc_closure((void*)(lp_mathlib_Nat_divisors___lam__0___boxed), 2, 1);
lean_closure_set(v___f_25_, 0, v_n_24_);
v___x_26_ = lean_unsigned_to_nat(1u);
v___x_27_ = lp_mathlib_Finset_Ico___at___00Nat_divisors_spec__0(v___x_26_, v_n_24_);
lean_dec(v_n_24_);
v___x_28_ = lp_mathlib_Multiset_filter___redArg(v___f_25_, v___x_27_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Icc___at___00Nat_divisorsAntidiagonal_spec__0(lean_object* v_a_29_, lean_object* v_b_30_){
_start:
{
lean_object* v___x_31_; lean_object* v___x_32_; lean_object* v___x_33_; lean_object* v___x_34_; lean_object* v___x_35_; lean_object* v___x_36_; 
v___x_31_ = lean_unsigned_to_nat(1u);
v___x_32_ = lean_nat_add(v_b_30_, v___x_31_);
v___x_33_ = lean_nat_sub(v___x_32_, v_a_29_);
lean_dec(v___x_32_);
v___x_34_ = lean_nat_add(v_a_29_, v___x_33_);
v___x_35_ = lean_box(0);
v___x_36_ = l_List_range_x27TR_go(v___x_31_, v___x_33_, v___x_34_, v___x_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Icc___at___00Nat_divisorsAntidiagonal_spec__0___boxed(lean_object* v_a_37_, lean_object* v_b_38_){
_start:
{
lean_object* v_res_39_; 
v_res_39_ = lp_mathlib_Finset_Icc___at___00Nat_divisorsAntidiagonal_spec__0(v_a_37_, v_b_38_);
lean_dec(v_b_38_);
lean_dec(v_a_37_);
return v_res_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_divisorsAntidiagonal___lam__0(lean_object* v_n_40_, lean_object* v_x_41_){
_start:
{
lean_object* v_y_42_; lean_object* v___x_43_; uint8_t v___x_44_; 
v_y_42_ = lean_nat_div(v_n_40_, v_x_41_);
v___x_43_ = lean_nat_mul(v_x_41_, v_y_42_);
v___x_44_ = lean_nat_dec_eq(v___x_43_, v_n_40_);
lean_dec(v___x_43_);
if (v___x_44_ == 0)
{
lean_object* v___x_45_; 
lean_dec(v_y_42_);
lean_dec(v_x_41_);
v___x_45_ = lean_box(0);
return v___x_45_;
}
else
{
lean_object* v___x_46_; lean_object* v___x_47_; 
v___x_46_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_46_, 0, v_x_41_);
lean_ctor_set(v___x_46_, 1, v_y_42_);
v___x_47_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_47_, 0, v___x_46_);
return v___x_47_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_divisorsAntidiagonal___lam__0___boxed(lean_object* v_n_48_, lean_object* v_x_49_){
_start:
{
lean_object* v_res_50_; 
v_res_50_ = lp_mathlib_Nat_divisorsAntidiagonal___lam__0(v_n_48_, v_x_49_);
lean_dec(v_n_48_);
return v_res_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_divisorsAntidiagonal(lean_object* v_n_51_){
_start:
{
lean_object* v___f_52_; lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; 
lean_inc(v_n_51_);
v___f_52_ = lean_alloc_closure((void*)(lp_mathlib_Nat_divisorsAntidiagonal___lam__0___boxed), 2, 1);
lean_closure_set(v___f_52_, 0, v_n_51_);
v___x_53_ = lean_unsigned_to_nat(1u);
v___x_54_ = lp_mathlib_Finset_Icc___at___00Nat_divisorsAntidiagonal_spec__0(v___x_53_, v_n_51_);
lean_dec(v_n_51_);
v___x_55_ = lp_mathlib_Multiset_filterMap___redArg(v___f_52_, v___x_54_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00Nat_divisorsAntidiagonalList_spec__0(lean_object* v_n_56_, lean_object* v_a_57_, lean_object* v_a_58_){
_start:
{
if (lean_obj_tag(v_a_57_) == 0)
{
lean_object* v___x_59_; 
v___x_59_ = lean_array_to_list(v_a_58_);
return v___x_59_;
}
else
{
lean_object* v_head_60_; lean_object* v_tail_61_; lean_object* v___x_63_; uint8_t v_isShared_64_; uint8_t v_isSharedCheck_74_; 
v_head_60_ = lean_ctor_get(v_a_57_, 0);
v_tail_61_ = lean_ctor_get(v_a_57_, 1);
v_isSharedCheck_74_ = !lean_is_exclusive(v_a_57_);
if (v_isSharedCheck_74_ == 0)
{
v___x_63_ = v_a_57_;
v_isShared_64_ = v_isSharedCheck_74_;
goto v_resetjp_62_;
}
else
{
lean_inc(v_tail_61_);
lean_inc(v_head_60_);
lean_dec(v_a_57_);
v___x_63_ = lean_box(0);
v_isShared_64_ = v_isSharedCheck_74_;
goto v_resetjp_62_;
}
v_resetjp_62_:
{
lean_object* v_y_65_; lean_object* v___x_66_; uint8_t v___x_67_; 
v_y_65_ = lean_nat_div(v_n_56_, v_head_60_);
v___x_66_ = lean_nat_mul(v_head_60_, v_y_65_);
v___x_67_ = lean_nat_dec_eq(v___x_66_, v_n_56_);
lean_dec(v___x_66_);
if (v___x_67_ == 0)
{
lean_dec(v_y_65_);
lean_del_object(v___x_63_);
lean_dec(v_head_60_);
v_a_57_ = v_tail_61_;
goto _start;
}
else
{
lean_object* v___x_70_; 
if (v_isShared_64_ == 0)
{
lean_ctor_set_tag(v___x_63_, 0);
lean_ctor_set(v___x_63_, 1, v_y_65_);
v___x_70_ = v___x_63_;
goto v_reusejp_69_;
}
else
{
lean_object* v_reuseFailAlloc_73_; 
v_reuseFailAlloc_73_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_73_, 0, v_head_60_);
lean_ctor_set(v_reuseFailAlloc_73_, 1, v_y_65_);
v___x_70_ = v_reuseFailAlloc_73_;
goto v_reusejp_69_;
}
v_reusejp_69_:
{
lean_object* v___x_71_; 
v___x_71_ = lean_array_push(v_a_58_, v___x_70_);
v_a_57_ = v_tail_61_;
v_a_58_ = v___x_71_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00Nat_divisorsAntidiagonalList_spec__0___boxed(lean_object* v_n_75_, lean_object* v_a_76_, lean_object* v_a_77_){
_start:
{
lean_object* v_res_78_; 
v_res_78_ = lp_mathlib_List_filterMapTR_go___at___00Nat_divisorsAntidiagonalList_spec__0(v_n_75_, v_a_76_, v_a_77_);
lean_dec(v_n_75_);
return v_res_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_divisorsAntidiagonalList(lean_object* v_n_81_){
_start:
{
lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; 
v___x_82_ = lean_unsigned_to_nat(1u);
v___x_83_ = lean_nat_add(v___x_82_, v_n_81_);
v___x_84_ = lean_box(0);
lean_inc(v_n_81_);
v___x_85_ = l_List_range_x27TR_go(v___x_82_, v_n_81_, v___x_83_, v___x_84_);
v___x_86_ = ((lean_object*)(lp_mathlib_Nat_divisorsAntidiagonalList___closed__0));
v___x_87_ = lp_mathlib_List_filterMapTR_go___at___00Nat_divisorsAntidiagonalList_spec__0(v_n_81_, v___x_85_, v___x_86_);
lean_dec(v_n_81_);
return v___x_87_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__6(void){
_start:
{
lean_object* v___x_133_; lean_object* v___x_134_; 
v___x_133_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__5));
v___x_134_ = l_String_toRawSubstring_x27(v___x_133_);
return v___x_134_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__18(void){
_start:
{
lean_object* v___x_157_; lean_object* v___x_158_; 
v___x_157_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__17));
v___x_158_ = l_String_toRawSubstring_x27(v___x_157_);
return v___x_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1(lean_object* v_x_167_, lean_object* v_a_168_, lean_object* v_a_169_){
_start:
{
lean_object* v___x_170_; uint8_t v___x_171_; 
v___x_170_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__12));
v___x_171_ = l_Lean_Syntax_isOfKind(v_x_167_, v___x_170_);
if (v___x_171_ == 0)
{
lean_object* v___x_172_; lean_object* v___x_173_; 
v___x_172_ = lean_box(1);
v___x_173_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_173_, 0, v___x_172_);
lean_ctor_set(v___x_173_, 1, v_a_169_);
return v___x_173_;
}
else
{
lean_object* v_quotContext_174_; lean_object* v_currMacroScope_175_; lean_object* v_ref_176_; uint8_t v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; 
v_quotContext_174_ = lean_ctor_get(v_a_168_, 1);
v_currMacroScope_175_ = lean_ctor_get(v_a_168_, 2);
v_ref_176_ = lean_ctor_get(v_a_168_, 5);
v___x_177_ = 0;
v___x_178_ = l_Lean_SourceInfo_fromRef(v_ref_176_, v___x_177_);
v___x_179_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__4));
v___x_180_ = lean_obj_once(&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__6, &lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__6_once, _init_lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__6);
v___x_181_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__9));
lean_inc_n(v_currMacroScope_175_, 2);
lean_inc_n(v_quotContext_174_, 2);
v___x_182_ = l_Lean_addMacroScope(v_quotContext_174_, v___x_181_, v_currMacroScope_175_);
v___x_183_ = lean_box(0);
v___x_184_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__11));
lean_inc_n(v___x_178_, 9);
v___x_185_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_185_, 0, v___x_178_);
lean_ctor_set(v___x_185_, 1, v___x_180_);
lean_ctor_set(v___x_185_, 2, v___x_182_);
lean_ctor_set(v___x_185_, 3, v___x_184_);
v___x_186_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__13));
v___x_187_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__15));
v___x_188_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__16));
v___x_189_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_189_, 0, v___x_178_);
lean_ctor_set(v___x_189_, 1, v___x_188_);
v___x_190_ = lean_obj_once(&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__18, &lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__18_once, _init_lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__18);
v___x_191_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__19));
v___x_192_ = l_Lean_addMacroScope(v_quotContext_174_, v___x_191_, v_currMacroScope_175_);
v___x_193_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_193_, 0, v___x_178_);
lean_ctor_set(v___x_193_, 1, v___x_190_);
lean_ctor_set(v___x_193_, 2, v___x_192_);
lean_ctor_set(v___x_193_, 3, v___x_183_);
v___x_194_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__20));
v___x_195_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_195_, 0, v___x_178_);
lean_ctor_set(v___x_195_, 1, v___x_194_);
v___x_196_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__22));
v___x_197_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__23));
v___x_198_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_198_, 0, v___x_178_);
lean_ctor_set(v___x_198_, 1, v___x_197_);
v___x_199_ = l_Lean_Syntax_node1(v___x_178_, v___x_196_, v___x_198_);
v___x_200_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__24));
v___x_201_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_201_, 0, v___x_178_);
lean_ctor_set(v___x_201_, 1, v___x_200_);
v___x_202_ = l_Lean_Syntax_node5(v___x_178_, v___x_187_, v___x_189_, v___x_193_, v___x_195_, v___x_199_, v___x_201_);
v___x_203_ = l_Lean_Syntax_node1(v___x_178_, v___x_186_, v___x_202_);
v___x_204_ = l_Lean_Syntax_node2(v___x_178_, v___x_179_, v___x_185_, v___x_203_);
v___x_205_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_205_, 0, v___x_204_);
lean_ctor_set(v___x_205_, 1, v_a_169_);
return v___x_205_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___boxed(lean_object* v_x_206_, lean_object* v_a_207_, lean_object* v_a_208_){
_start:
{
lean_object* v_res_209_; 
v_res_209_ = lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1(v_x_206_, v_a_207_, v_a_208_);
lean_dec_ref(v_a_207_);
return v_res_209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______unexpand__Nat__castEmbedding__1(lean_object* v_x_213_, lean_object* v_a_214_, lean_object* v_a_215_){
_start:
{
lean_object* v___x_216_; uint8_t v___x_217_; 
v___x_216_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__4));
lean_inc(v_x_213_);
v___x_217_ = l_Lean_Syntax_isOfKind(v_x_213_, v___x_216_);
if (v___x_217_ == 0)
{
lean_object* v___x_218_; lean_object* v___x_219_; 
lean_dec(v_x_213_);
v___x_218_ = lean_box(0);
v___x_219_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_219_, 0, v___x_218_);
lean_ctor_set(v___x_219_, 1, v_a_215_);
return v___x_219_;
}
else
{
lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; uint8_t v___x_223_; 
v___x_220_ = lean_unsigned_to_nat(0u);
v___x_221_ = l_Lean_Syntax_getArg(v_x_213_, v___x_220_);
v___x_222_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______unexpand__Nat__castEmbedding__1___closed__1));
lean_inc(v___x_221_);
v___x_223_ = l_Lean_Syntax_isOfKind(v___x_221_, v___x_222_);
if (v___x_223_ == 0)
{
lean_object* v___x_224_; lean_object* v___x_225_; 
lean_dec(v___x_221_);
lean_dec(v_x_213_);
v___x_224_ = lean_box(0);
v___x_225_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_225_, 0, v___x_224_);
lean_ctor_set(v___x_225_, 1, v_a_215_);
return v___x_225_;
}
else
{
lean_object* v___x_226_; lean_object* v___x_227_; uint8_t v___x_228_; 
v___x_226_ = lean_unsigned_to_nat(1u);
v___x_227_ = l_Lean_Syntax_getArg(v_x_213_, v___x_226_);
lean_dec(v_x_213_);
lean_inc(v___x_227_);
v___x_228_ = l_Lean_Syntax_matchesNull(v___x_227_, v___x_226_);
if (v___x_228_ == 0)
{
lean_object* v___x_229_; lean_object* v___x_230_; 
lean_dec(v___x_227_);
lean_dec(v___x_221_);
v___x_229_ = lean_box(0);
v___x_230_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_230_, 0, v___x_229_);
lean_ctor_set(v___x_230_, 1, v_a_215_);
return v___x_230_;
}
else
{
lean_object* v___x_231_; lean_object* v___x_232_; uint8_t v___x_233_; 
v___x_231_ = l_Lean_Syntax_getArg(v___x_227_, v___x_220_);
lean_dec(v___x_227_);
v___x_232_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__15));
lean_inc(v___x_231_);
v___x_233_ = l_Lean_Syntax_isOfKind(v___x_231_, v___x_232_);
if (v___x_233_ == 0)
{
lean_object* v___x_234_; lean_object* v___x_235_; 
lean_dec(v___x_231_);
lean_dec(v___x_221_);
v___x_234_ = lean_box(0);
v___x_235_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_235_, 0, v___x_234_);
lean_ctor_set(v___x_235_, 1, v_a_215_);
return v___x_235_;
}
else
{
lean_object* v___x_236_; lean_object* v___x_237_; uint8_t v___x_238_; 
v___x_236_ = l_Lean_Syntax_getArg(v___x_231_, v___x_226_);
v___x_237_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__19));
v___x_238_ = l_Lean_Syntax_matchesIdent(v___x_236_, v___x_237_);
lean_dec(v___x_236_);
if (v___x_238_ == 0)
{
lean_object* v___x_239_; lean_object* v___x_240_; 
lean_dec(v___x_231_);
lean_dec(v___x_221_);
v___x_239_ = lean_box(0);
v___x_240_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_240_, 0, v___x_239_);
lean_ctor_set(v___x_240_, 1, v_a_215_);
return v___x_240_;
}
else
{
lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; uint8_t v___x_244_; 
v___x_241_ = lean_unsigned_to_nat(3u);
v___x_242_ = l_Lean_Syntax_getArg(v___x_231_, v___x_241_);
lean_dec(v___x_231_);
v___x_243_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__22));
v___x_244_ = l_Lean_Syntax_isOfKind(v___x_242_, v___x_243_);
if (v___x_244_ == 0)
{
lean_object* v___x_245_; lean_object* v___x_246_; 
lean_dec(v___x_221_);
v___x_245_ = lean_box(0);
v___x_246_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_246_, 0, v___x_245_);
lean_ctor_set(v___x_246_, 1, v_a_215_);
return v___x_246_;
}
else
{
lean_object* v_ref_247_; uint8_t v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; 
v_ref_247_ = l_Lean_replaceRef(v___x_221_, v_a_214_);
lean_dec(v___x_221_);
v___x_248_ = 0;
v___x_249_ = l_Lean_SourceInfo_fromRef(v_ref_247_, v___x_248_);
lean_dec(v_ref_247_);
v___x_250_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__12));
v___x_251_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNatCast___closed__13));
lean_inc(v___x_249_);
v___x_252_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_252_, 0, v___x_249_);
lean_ctor_set(v___x_252_, 1, v___x_251_);
v___x_253_ = l_Lean_Syntax_node1(v___x_249_, v___x_250_, v___x_252_);
v___x_254_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_254_, 0, v___x_253_);
lean_ctor_set(v___x_254_, 1, v_a_215_);
return v___x_254_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______unexpand__Nat__castEmbedding__1___boxed(lean_object* v_x_255_, lean_object* v_a_256_, lean_object* v_a_257_){
_start:
{
lean_object* v_res_258_; 
v_res_258_ = lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______unexpand__Nat__castEmbedding__1(v_x_255_, v_a_256_, v_a_257_);
lean_dec(v_a_256_);
return v_res_258_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__1(void){
_start:
{
lean_object* v___x_272_; lean_object* v___x_273_; 
v___x_272_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__0));
v___x_273_ = l_String_toRawSubstring_x27(v___x_272_);
return v___x_273_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__15(void){
_start:
{
lean_object* v___x_303_; lean_object* v___x_304_; 
v___x_303_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__14));
v___x_304_ = l_String_toRawSubstring_x27(v___x_303_);
return v___x_304_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__24(void){
_start:
{
lean_object* v___x_321_; lean_object* v___x_322_; 
v___x_321_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__23));
v___x_322_ = l_String_toRawSubstring_x27(v___x_321_);
return v___x_322_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__31(void){
_start:
{
lean_object* v___x_335_; lean_object* v___x_336_; 
v___x_335_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__30));
v___x_336_ = l_String_toRawSubstring_x27(v___x_335_);
return v___x_336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1(lean_object* v_x_347_, lean_object* v_a_348_, lean_object* v_a_349_){
_start:
{
lean_object* v___x_350_; lean_object* v___x_351_; uint8_t v___x_352_; 
v___x_350_ = lean_box(0);
v___x_351_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNegNatCast___closed__1));
v___x_352_ = l_Lean_Syntax_isOfKind(v_x_347_, v___x_351_);
if (v___x_352_ == 0)
{
lean_object* v___x_353_; lean_object* v___x_354_; 
v___x_353_ = lean_box(1);
v___x_354_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_354_, 0, v___x_353_);
lean_ctor_set(v___x_354_, 1, v_a_349_);
return v___x_354_;
}
else
{
lean_object* v_quotContext_355_; lean_object* v_currMacroScope_356_; lean_object* v_ref_357_; uint8_t v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; lean_object* v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; lean_object* v___x_364_; lean_object* v___x_365_; lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v___x_386_; lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_399_; lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; lean_object* v___x_407_; 
v_quotContext_355_ = lean_ctor_get(v_a_348_, 1);
v_currMacroScope_356_ = lean_ctor_get(v_a_348_, 2);
v_ref_357_ = lean_ctor_get(v_a_348_, 5);
v___x_358_ = 0;
v___x_359_ = l_Lean_SourceInfo_fromRef(v_ref_357_, v___x_358_);
v___x_360_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__4));
v___x_361_ = lean_obj_once(&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__1, &lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__1_once, _init_lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__1);
v___x_362_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__5));
lean_inc_n(v_currMacroScope_356_, 5);
lean_inc_n(v_quotContext_355_, 5);
v___x_363_ = l_Lean_addMacroScope(v_quotContext_355_, v___x_362_, v_currMacroScope_356_);
v___x_364_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__7));
lean_inc_n(v___x_359_, 18);
v___x_365_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_365_, 0, v___x_359_);
lean_ctor_set(v___x_365_, 1, v___x_361_);
lean_ctor_set(v___x_365_, 2, v___x_363_);
lean_ctor_set(v___x_365_, 3, v___x_364_);
v___x_366_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__13));
v___x_367_ = lean_obj_once(&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__6, &lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__6_once, _init_lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__6);
v___x_368_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__9));
v___x_369_ = l_Lean_addMacroScope(v_quotContext_355_, v___x_368_, v_currMacroScope_356_);
v___x_370_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__11));
v___x_371_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_371_, 0, v___x_359_);
lean_ctor_set(v___x_371_, 1, v___x_367_);
lean_ctor_set(v___x_371_, 2, v___x_369_);
lean_ctor_set(v___x_371_, 3, v___x_370_);
v___x_372_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__9));
v___x_373_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__11));
v___x_374_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__16));
v___x_375_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_375_, 0, v___x_359_);
lean_ctor_set(v___x_375_, 1, v___x_374_);
v___x_376_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__13));
v___x_377_ = lean_obj_once(&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__15, &lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__15_once, _init_lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__15);
v___x_378_ = l_Lean_addMacroScope(v_quotContext_355_, v___x_350_, v_currMacroScope_356_);
v___x_379_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__22));
v___x_380_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_380_, 0, v___x_359_);
lean_ctor_set(v___x_380_, 1, v___x_377_);
lean_ctor_set(v___x_380_, 2, v___x_378_);
lean_ctor_set(v___x_380_, 3, v___x_379_);
v___x_381_ = l_Lean_Syntax_node1(v___x_359_, v___x_376_, v___x_380_);
v___x_382_ = l_Lean_Syntax_node2(v___x_359_, v___x_373_, v___x_375_, v___x_381_);
v___x_383_ = lean_obj_once(&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__24, &lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__24_once, _init_lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__24);
v___x_384_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__27));
v___x_385_ = l_Lean_addMacroScope(v_quotContext_355_, v___x_384_, v_currMacroScope_356_);
v___x_386_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__29));
v___x_387_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_387_, 0, v___x_359_);
lean_ctor_set(v___x_387_, 1, v___x_383_);
lean_ctor_set(v___x_387_, 2, v___x_385_);
lean_ctor_set(v___x_387_, 3, v___x_386_);
v___x_388_ = lean_obj_once(&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__31, &lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__31_once, _init_lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__31);
v___x_389_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__33));
v___x_390_ = l_Lean_addMacroScope(v_quotContext_355_, v___x_389_, v_currMacroScope_356_);
v___x_391_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__35));
v___x_392_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_392_, 0, v___x_359_);
lean_ctor_set(v___x_392_, 1, v___x_388_);
lean_ctor_set(v___x_392_, 2, v___x_390_);
lean_ctor_set(v___x_392_, 3, v___x_391_);
v___x_393_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__22));
v___x_394_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__23));
v___x_395_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_395_, 0, v___x_359_);
lean_ctor_set(v___x_395_, 1, v___x_394_);
v___x_396_ = l_Lean_Syntax_node1(v___x_359_, v___x_393_, v___x_395_);
v___x_397_ = l_Lean_Syntax_node1(v___x_359_, v___x_366_, v___x_396_);
v___x_398_ = l_Lean_Syntax_node2(v___x_359_, v___x_360_, v___x_392_, v___x_397_);
v___x_399_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__24));
v___x_400_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_400_, 0, v___x_359_);
lean_ctor_set(v___x_400_, 1, v___x_399_);
lean_inc_ref(v___x_400_);
lean_inc(v___x_382_);
v___x_401_ = l_Lean_Syntax_node3(v___x_359_, v___x_372_, v___x_382_, v___x_398_, v___x_400_);
v___x_402_ = l_Lean_Syntax_node1(v___x_359_, v___x_366_, v___x_401_);
v___x_403_ = l_Lean_Syntax_node2(v___x_359_, v___x_360_, v___x_387_, v___x_402_);
v___x_404_ = l_Lean_Syntax_node3(v___x_359_, v___x_372_, v___x_382_, v___x_403_, v___x_400_);
v___x_405_ = l_Lean_Syntax_node2(v___x_359_, v___x_366_, v___x_371_, v___x_404_);
v___x_406_ = l_Lean_Syntax_node2(v___x_359_, v___x_360_, v___x_365_, v___x_405_);
v___x_407_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_407_, 0, v___x_406_);
lean_ctor_set(v___x_407_, 1, v_a_349_);
return v___x_407_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___boxed(lean_object* v_x_408_, lean_object* v_a_409_, lean_object* v_a_410_){
_start:
{
lean_object* v_res_411_; 
v_res_411_ = lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1(v_x_408_, v_a_409_, v_a_410_);
lean_dec_ref(v_a_409_);
return v_res_411_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______unexpand__Function__Embedding__trans__1(lean_object* v_x_412_, lean_object* v_a_413_, lean_object* v_a_414_){
_start:
{
lean_object* v___x_415_; uint8_t v___x_416_; 
v___x_415_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__4));
lean_inc(v_x_412_);
v___x_416_ = l_Lean_Syntax_isOfKind(v_x_412_, v___x_415_);
if (v___x_416_ == 0)
{
lean_object* v___x_417_; lean_object* v___x_418_; 
lean_dec(v_x_412_);
v___x_417_ = lean_box(0);
v___x_418_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_418_, 0, v___x_417_);
lean_ctor_set(v___x_418_, 1, v_a_414_);
return v___x_418_;
}
else
{
lean_object* v___x_419_; lean_object* v___x_420_; lean_object* v___x_421_; uint8_t v___x_422_; 
v___x_419_ = lean_unsigned_to_nat(0u);
v___x_420_ = l_Lean_Syntax_getArg(v_x_412_, v___x_419_);
v___x_421_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______unexpand__Nat__castEmbedding__1___closed__1));
lean_inc(v___x_420_);
v___x_422_ = l_Lean_Syntax_isOfKind(v___x_420_, v___x_421_);
if (v___x_422_ == 0)
{
lean_object* v___x_423_; lean_object* v___x_424_; 
lean_dec(v___x_420_);
lean_dec(v_x_412_);
v___x_423_ = lean_box(0);
v___x_424_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_424_, 0, v___x_423_);
lean_ctor_set(v___x_424_, 1, v_a_414_);
return v___x_424_;
}
else
{
lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v___x_427_; uint8_t v___x_428_; 
v___x_425_ = lean_unsigned_to_nat(1u);
v___x_426_ = l_Lean_Syntax_getArg(v_x_412_, v___x_425_);
lean_dec(v_x_412_);
v___x_427_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_426_);
v___x_428_ = l_Lean_Syntax_matchesNull(v___x_426_, v___x_427_);
if (v___x_428_ == 0)
{
lean_object* v___x_429_; lean_object* v___x_430_; 
lean_dec(v___x_426_);
lean_dec(v___x_420_);
v___x_429_ = lean_box(0);
v___x_430_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_430_, 0, v___x_429_);
lean_ctor_set(v___x_430_, 1, v_a_414_);
return v___x_430_;
}
else
{
lean_object* v___x_431_; lean_object* v___x_432_; uint8_t v___x_433_; 
v___x_431_ = l_Lean_Syntax_getArg(v___x_426_, v___x_419_);
v___x_432_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__9));
v___x_433_ = l_Lean_Syntax_matchesIdent(v___x_431_, v___x_432_);
lean_dec(v___x_431_);
if (v___x_433_ == 0)
{
lean_object* v___x_434_; lean_object* v___x_435_; 
lean_dec(v___x_426_);
lean_dec(v___x_420_);
v___x_434_ = lean_box(0);
v___x_435_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_435_, 0, v___x_434_);
lean_ctor_set(v___x_435_, 1, v_a_414_);
return v___x_435_;
}
else
{
lean_object* v___x_436_; uint8_t v___x_437_; 
v___x_436_ = l_Lean_Syntax_getArg(v___x_426_, v___x_425_);
lean_dec(v___x_426_);
lean_inc(v___x_436_);
v___x_437_ = l_Lean_Syntax_isOfKind(v___x_436_, v___x_415_);
if (v___x_437_ == 0)
{
lean_object* v___x_438_; lean_object* v___x_439_; 
lean_dec(v___x_436_);
lean_dec(v___x_420_);
v___x_438_ = lean_box(0);
v___x_439_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_439_, 0, v___x_438_);
lean_ctor_set(v___x_439_, 1, v_a_414_);
return v___x_439_;
}
else
{
lean_object* v___x_440_; lean_object* v___x_441_; uint8_t v___x_442_; 
v___x_440_ = l_Lean_Syntax_getArg(v___x_436_, v___x_419_);
v___x_441_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__27));
v___x_442_ = l_Lean_Syntax_matchesIdent(v___x_440_, v___x_441_);
lean_dec(v___x_440_);
if (v___x_442_ == 0)
{
lean_object* v___x_443_; lean_object* v___x_444_; 
lean_dec(v___x_436_);
lean_dec(v___x_420_);
v___x_443_ = lean_box(0);
v___x_444_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_444_, 0, v___x_443_);
lean_ctor_set(v___x_444_, 1, v_a_414_);
return v___x_444_;
}
else
{
lean_object* v___x_445_; uint8_t v___x_446_; 
v___x_445_ = l_Lean_Syntax_getArg(v___x_436_, v___x_425_);
lean_dec(v___x_436_);
lean_inc(v___x_445_);
v___x_446_ = l_Lean_Syntax_matchesNull(v___x_445_, v___x_425_);
if (v___x_446_ == 0)
{
lean_object* v___x_447_; lean_object* v___x_448_; 
lean_dec(v___x_445_);
lean_dec(v___x_420_);
v___x_447_ = lean_box(0);
v___x_448_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_448_, 0, v___x_447_);
lean_ctor_set(v___x_448_, 1, v_a_414_);
return v___x_448_;
}
else
{
lean_object* v___x_449_; uint8_t v___x_450_; 
v___x_449_ = l_Lean_Syntax_getArg(v___x_445_, v___x_419_);
lean_dec(v___x_445_);
lean_inc(v___x_449_);
v___x_450_ = l_Lean_Syntax_isOfKind(v___x_449_, v___x_415_);
if (v___x_450_ == 0)
{
lean_object* v___x_451_; lean_object* v___x_452_; 
lean_dec(v___x_449_);
lean_dec(v___x_420_);
v___x_451_ = lean_box(0);
v___x_452_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_452_, 0, v___x_451_);
lean_ctor_set(v___x_452_, 1, v_a_414_);
return v___x_452_;
}
else
{
lean_object* v___x_453_; lean_object* v___x_454_; uint8_t v___x_455_; 
v___x_453_ = l_Lean_Syntax_getArg(v___x_449_, v___x_419_);
v___x_454_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNegNatCast__1___closed__33));
v___x_455_ = l_Lean_Syntax_matchesIdent(v___x_453_, v___x_454_);
lean_dec(v___x_453_);
if (v___x_455_ == 0)
{
lean_object* v___x_456_; lean_object* v___x_457_; 
lean_dec(v___x_449_);
lean_dec(v___x_420_);
v___x_456_ = lean_box(0);
v___x_457_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_457_, 0, v___x_456_);
lean_ctor_set(v___x_457_, 1, v_a_414_);
return v___x_457_;
}
else
{
lean_object* v___x_458_; uint8_t v___x_459_; 
v___x_458_ = l_Lean_Syntax_getArg(v___x_449_, v___x_425_);
lean_dec(v___x_449_);
lean_inc(v___x_458_);
v___x_459_ = l_Lean_Syntax_matchesNull(v___x_458_, v___x_425_);
if (v___x_459_ == 0)
{
lean_object* v___x_460_; lean_object* v___x_461_; 
lean_dec(v___x_458_);
lean_dec(v___x_420_);
v___x_460_ = lean_box(0);
v___x_461_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_461_, 0, v___x_460_);
lean_ctor_set(v___x_461_, 1, v_a_414_);
return v___x_461_;
}
else
{
lean_object* v___x_462_; lean_object* v___x_463_; uint8_t v___x_464_; 
v___x_462_ = l_Lean_Syntax_getArg(v___x_458_, v___x_419_);
lean_dec(v___x_458_);
v___x_463_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______macroRules____private__Mathlib__NumberTheory__Divisors__0__Int__termNatCast__1___closed__22));
v___x_464_ = l_Lean_Syntax_isOfKind(v___x_462_, v___x_463_);
if (v___x_464_ == 0)
{
lean_object* v___x_465_; lean_object* v___x_466_; 
lean_dec(v___x_420_);
v___x_465_ = lean_box(0);
v___x_466_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_466_, 0, v___x_465_);
lean_ctor_set(v___x_466_, 1, v_a_414_);
return v___x_466_;
}
else
{
lean_object* v_ref_467_; uint8_t v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; 
v_ref_467_ = l_Lean_replaceRef(v___x_420_, v_a_413_);
lean_dec(v___x_420_);
v___x_468_ = 0;
v___x_469_ = l_Lean_SourceInfo_fromRef(v_ref_467_, v___x_468_);
lean_dec(v_ref_467_);
v___x_470_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNegNatCast___closed__1));
v___x_471_ = ((lean_object*)(lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_termNegNatCast___closed__2));
lean_inc(v___x_469_);
v___x_472_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_472_, 0, v___x_469_);
lean_ctor_set(v___x_472_, 1, v___x_471_);
v___x_473_ = l_Lean_Syntax_node1(v___x_469_, v___x_470_, v___x_472_);
v___x_474_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_474_, 0, v___x_473_);
lean_ctor_set(v___x_474_, 1, v_a_414_);
return v___x_474_;
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
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______unexpand__Function__Embedding__trans__1___boxed(lean_object* v_x_475_, lean_object* v_a_476_, lean_object* v_a_477_){
_start:
{
lean_object* v_res_478_; 
v_res_478_ = lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int___aux__Mathlib__NumberTheory__Divisors______unexpand__Function__Embedding__trans__1(v_x_475_, v_a_476_, v_a_477_);
lean_dec(v_a_476_);
return v_res_478_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_castEmbedding___at___00Int_divisors_spec__0(lean_object* v_inst_483_){
_start:
{
lean_object* v___f_484_; 
v___f_484_ = lean_alloc_closure((void*)(l_Int_ofNat___boxed), 1, 0);
return v___f_484_;
}
}
static lean_object* _init_lp_mathlib_Int_divisors___closed__1(void){
_start:
{
lean_object* v___f_487_; lean_object* v___f_488_; lean_object* v___f_489_; 
v___f_487_ = ((lean_object*)(lp_mathlib_Int_divisors___closed__0));
v___f_488_ = lean_alloc_closure((void*)(l_Int_ofNat___boxed), 1, 0);
v___f_489_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_trans___redArg___lam__0), 3, 2);
lean_closure_set(v___f_489_, 0, v___f_488_);
lean_closure_set(v___f_489_, 1, v___f_487_);
return v___f_489_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_divisors(lean_object* v_z_490_){
_start:
{
lean_object* v___f_491_; lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v___x_494_; lean_object* v___f_495_; lean_object* v___x_496_; lean_object* v___x_497_; 
v___f_491_ = lean_alloc_closure((void*)(l_Int_ofNat___boxed), 1, 0);
v___x_492_ = lean_nat_abs(v_z_490_);
v___x_493_ = lp_mathlib_Nat_divisors(v___x_492_);
lean_inc(v___x_493_);
v___x_494_ = lp_mathlib_Finset_map___redArg(v___f_491_, v___x_493_);
v___f_495_ = lean_obj_once(&lp_mathlib_Int_divisors___closed__1, &lp_mathlib_Int_divisors___closed__1_once, _init_lp_mathlib_Int_divisors___closed__1);
v___x_496_ = lp_mathlib_Finset_map___redArg(v___f_495_, v___x_493_);
v___x_497_ = l_List_appendTR___redArg(v___x_494_, v___x_496_);
return v___x_497_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_divisors___boxed(lean_object* v_z_498_){
_start:
{
lean_object* v_res_499_; 
v_res_499_ = lp_mathlib_Int_divisors(v_z_498_);
lean_dec(v_z_498_);
return v_res_499_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Nat_cast___at___00Nat_castEmbedding___at___00Int_divisors_spec__0_spec__0_spec__2(lean_object* v_a_500_){
_start:
{
lean_object* v___x_501_; 
v___x_501_ = lean_nat_to_int(v_a_500_);
return v___x_501_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Nat_castEmbedding___at___00Int_divisors_spec__0_spec__0(lean_object* v_a_502_){
_start:
{
lean_object* v___x_503_; 
v___x_503_ = lean_nat_to_int(v_a_502_);
return v___x_503_;
}
}
static lean_object* _init_lp_mathlib_Int_divisorsAntidiag___closed__0(void){
_start:
{
lean_object* v_natZero_504_; lean_object* v_intZero_505_; 
v_natZero_504_ = lean_unsigned_to_nat(0u);
v_intZero_505_ = lean_nat_to_int(v_natZero_504_);
return v_intZero_505_;
}
}
static lean_object* _init_lp_mathlib_Int_divisorsAntidiag___closed__1(void){
_start:
{
lean_object* v___f_506_; lean_object* v___x_507_; 
v___f_506_ = lean_alloc_closure((void*)(l_Int_ofNat___boxed), 1, 0);
lean_inc_ref(v___f_506_);
v___x_507_ = lp_mathlib_Function_Embedding_prodMap___redArg(v___f_506_, v___f_506_);
return v___x_507_;
}
}
static lean_object* _init_lp_mathlib_Int_divisorsAntidiag___closed__2(void){
_start:
{
lean_object* v___f_508_; lean_object* v___x_509_; 
v___f_508_ = lean_obj_once(&lp_mathlib_Int_divisors___closed__1, &lp_mathlib_Int_divisors___closed__1_once, _init_lp_mathlib_Int_divisors___closed__1);
v___x_509_ = lp_mathlib_Function_Embedding_prodMap___redArg(v___f_508_, v___f_508_);
return v___x_509_;
}
}
static lean_object* _init_lp_mathlib_Int_divisorsAntidiag___closed__3(void){
_start:
{
lean_object* v___f_510_; lean_object* v___f_511_; lean_object* v___x_512_; 
v___f_510_ = lean_obj_once(&lp_mathlib_Int_divisors___closed__1, &lp_mathlib_Int_divisors___closed__1_once, _init_lp_mathlib_Int_divisors___closed__1);
v___f_511_ = lean_alloc_closure((void*)(l_Int_ofNat___boxed), 1, 0);
v___x_512_ = lp_mathlib_Function_Embedding_prodMap___redArg(v___f_511_, v___f_510_);
return v___x_512_;
}
}
static lean_object* _init_lp_mathlib_Int_divisorsAntidiag___closed__4(void){
_start:
{
lean_object* v___f_513_; lean_object* v___f_514_; lean_object* v___x_515_; 
v___f_513_ = lean_alloc_closure((void*)(l_Int_ofNat___boxed), 1, 0);
v___f_514_ = lean_obj_once(&lp_mathlib_Int_divisors___closed__1, &lp_mathlib_Int_divisors___closed__1_once, _init_lp_mathlib_Int_divisors___closed__1);
v___x_515_ = lp_mathlib_Function_Embedding_prodMap___redArg(v___f_514_, v___f_513_);
return v___x_515_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_divisorsAntidiag(lean_object* v_x_516_){
_start:
{
lean_object* v_intZero_517_; uint8_t v_isNeg_518_; 
v_intZero_517_ = lean_obj_once(&lp_mathlib_Int_divisorsAntidiag___closed__0, &lp_mathlib_Int_divisorsAntidiag___closed__0_once, _init_lp_mathlib_Int_divisorsAntidiag___closed__0);
v_isNeg_518_ = lean_int_dec_lt(v_x_516_, v_intZero_517_);
if (v_isNeg_518_ == 0)
{
lean_object* v_a_519_; lean_object* v_s_520_; lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; 
v_a_519_ = lean_nat_abs(v_x_516_);
v_s_520_ = lp_mathlib_Nat_divisorsAntidiagonal(v_a_519_);
v___x_521_ = lean_obj_once(&lp_mathlib_Int_divisorsAntidiag___closed__1, &lp_mathlib_Int_divisorsAntidiag___closed__1_once, _init_lp_mathlib_Int_divisorsAntidiag___closed__1);
lean_inc(v_s_520_);
v___x_522_ = lp_mathlib_Finset_map___redArg(v___x_521_, v_s_520_);
v___x_523_ = lean_obj_once(&lp_mathlib_Int_divisorsAntidiag___closed__2, &lp_mathlib_Int_divisorsAntidiag___closed__2_once, _init_lp_mathlib_Int_divisorsAntidiag___closed__2);
v___x_524_ = lp_mathlib_Finset_map___redArg(v___x_523_, v_s_520_);
v___x_525_ = l_List_appendTR___redArg(v___x_522_, v___x_524_);
return v___x_525_;
}
else
{
lean_object* v_abs_526_; lean_object* v_one_527_; lean_object* v_a_528_; lean_object* v___x_529_; lean_object* v_s_530_; lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___x_535_; 
v_abs_526_ = lean_nat_abs(v_x_516_);
v_one_527_ = lean_unsigned_to_nat(1u);
v_a_528_ = lean_nat_sub(v_abs_526_, v_one_527_);
lean_dec(v_abs_526_);
v___x_529_ = lean_nat_add(v_a_528_, v_one_527_);
lean_dec(v_a_528_);
v_s_530_ = lp_mathlib_Nat_divisorsAntidiagonal(v___x_529_);
v___x_531_ = lean_obj_once(&lp_mathlib_Int_divisorsAntidiag___closed__3, &lp_mathlib_Int_divisorsAntidiag___closed__3_once, _init_lp_mathlib_Int_divisorsAntidiag___closed__3);
lean_inc(v_s_530_);
v___x_532_ = lp_mathlib_Finset_map___redArg(v___x_531_, v_s_530_);
v___x_533_ = lean_obj_once(&lp_mathlib_Int_divisorsAntidiag___closed__4, &lp_mathlib_Int_divisorsAntidiag___closed__4_once, _init_lp_mathlib_Int_divisorsAntidiag___closed__4);
v___x_534_ = lp_mathlib_Finset_map___redArg(v___x_533_, v_s_530_);
v___x_535_ = l_List_appendTR___redArg(v___x_532_, v___x_534_);
return v___x_535_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_divisorsAntidiag___boxed(lean_object* v_x_536_){
_start:
{
lean_object* v_res_537_; 
v_res_537_ = lp_mathlib_Int_divisorsAntidiag(v_x_536_);
lean_dec(v_x_536_);
return v_res_537_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_divisorsAntidiag_match__1_splitter___redArg___closed__0(void){
_start:
{
lean_object* v_natZero_538_; lean_object* v_intZero_539_; 
v_natZero_538_ = lean_unsigned_to_nat(0u);
v_intZero_539_ = lean_nat_to_int(v_natZero_538_);
return v_intZero_539_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_divisorsAntidiag_match__1_splitter___redArg(lean_object* v_x_540_, lean_object* v_h__1_541_, lean_object* v_h__2_542_){
_start:
{
lean_object* v_intZero_543_; uint8_t v_isNeg_544_; 
v_intZero_543_ = lean_obj_once(&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_divisorsAntidiag_match__1_splitter___redArg___closed__0, &lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_divisorsAntidiag_match__1_splitter___redArg___closed__0_once, _init_lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_divisorsAntidiag_match__1_splitter___redArg___closed__0);
v_isNeg_544_ = lean_int_dec_lt(v_x_540_, v_intZero_543_);
if (v_isNeg_544_ == 0)
{
lean_object* v_a_545_; lean_object* v___x_546_; 
lean_dec(v_h__2_542_);
v_a_545_ = lean_nat_abs(v_x_540_);
v___x_546_ = lean_apply_1(v_h__1_541_, v_a_545_);
return v___x_546_;
}
else
{
lean_object* v_abs_547_; lean_object* v_one_548_; lean_object* v_a_549_; lean_object* v___x_550_; 
lean_dec(v_h__1_541_);
v_abs_547_ = lean_nat_abs(v_x_540_);
v_one_548_ = lean_unsigned_to_nat(1u);
v_a_549_ = lean_nat_sub(v_abs_547_, v_one_548_);
lean_dec(v_abs_547_);
v___x_550_ = lean_apply_1(v_h__2_542_, v_a_549_);
return v___x_550_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_divisorsAntidiag_match__1_splitter___redArg___boxed(lean_object* v_x_551_, lean_object* v_h__1_552_, lean_object* v_h__2_553_){
_start:
{
lean_object* v_res_554_; 
v_res_554_ = lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_divisorsAntidiag_match__1_splitter___redArg(v_x_551_, v_h__1_552_, v_h__2_553_);
lean_dec(v_x_551_);
return v_res_554_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_divisorsAntidiag_match__1_splitter(lean_object* v_motive_555_, lean_object* v_x_556_, lean_object* v_h__1_557_, lean_object* v_h__2_558_){
_start:
{
lean_object* v_intZero_559_; uint8_t v_isNeg_560_; 
v_intZero_559_ = lean_obj_once(&lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_divisorsAntidiag_match__1_splitter___redArg___closed__0, &lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_divisorsAntidiag_match__1_splitter___redArg___closed__0_once, _init_lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_divisorsAntidiag_match__1_splitter___redArg___closed__0);
v_isNeg_560_ = lean_int_dec_lt(v_x_556_, v_intZero_559_);
if (v_isNeg_560_ == 0)
{
lean_object* v_a_561_; lean_object* v___x_562_; 
lean_dec(v_h__2_558_);
v_a_561_ = lean_nat_abs(v_x_556_);
v___x_562_ = lean_apply_1(v_h__1_557_, v_a_561_);
return v___x_562_;
}
else
{
lean_object* v_abs_563_; lean_object* v_one_564_; lean_object* v_a_565_; lean_object* v___x_566_; 
lean_dec(v_h__1_557_);
v_abs_563_ = lean_nat_abs(v_x_556_);
v_one_564_ = lean_unsigned_to_nat(1u);
v_a_565_ = lean_nat_sub(v_abs_563_, v_one_564_);
lean_dec(v_abs_563_);
v___x_566_ = lean_apply_1(v_h__2_558_, v_a_565_);
return v___x_566_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_divisorsAntidiag_match__1_splitter___boxed(lean_object* v_motive_567_, lean_object* v_x_568_, lean_object* v_h__1_569_, lean_object* v_h__2_570_){
_start:
{
lean_object* v_res_571_; 
v_res_571_ = lp_mathlib___private_Mathlib_NumberTheory_Divisors_0__Int_divisorsAntidiag_match__1_splitter(v_motive_567_, v_x_568_, v_h__1_569_, v_h__2_570_);
lean_dec(v_x_568_);
return v_res_571_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_IsPrimePow(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_BigOperators_Group_Finset(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Interval_Finset_SuccPred(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Int(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_CharZero(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_NatAntidiagonal(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Ring(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_PrimeFin(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_SuccPred(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Interval_Finset_Nat(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_NumberTheory_Divisors(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_IsPrimePow(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_BigOperators_Group_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Interval_Finset_SuccPred(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Int(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_CharZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_NatAntidiagonal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_PrimeFin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_SuccPred(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Interval_Finset_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_NumberTheory_Divisors(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_IsPrimePow(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_BigOperators_Group_Finset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Interval_Finset_SuccPred(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Int(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_CharZero(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_NatAntidiagonal(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Ring(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_PrimeFin(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_SuccPred(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Finset_Nat(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_NumberTheory_Divisors(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_IsPrimePow(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_BigOperators_Group_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Interval_Finset_SuccPred(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Ring_Int(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_CharZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_NatAntidiagonal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_PrimeFin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_SuccPred(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Finset_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_NumberTheory_Divisors(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_NumberTheory_Divisors(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_NumberTheory_Divisors(builtin);
}
#ifdef __cplusplus
}
#endif
