// Lean compiler output
// Module: Mathlib.Data.Nat.Factorization.Defs
// Imports: public import Init public meta import Init public import Batteries.Data.List.Count public import Mathlib.Data.Finsupp.Multiset public import Mathlib.Data.Finsupp.Order public import Mathlib.Data.Nat.PrimeFin public import Mathlib.NumberTheory.Padics.PadicVal.Defs
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
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lean_array_mk(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
size_t lean_usize_sub(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t l_List_decidableBAll___redArg(lean_object*, lean_object*);
lean_object* l_Nat_pow___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_map___redArg(lean_object*, lean_object*);
lean_object* l_Nat_mul___boxed(lean_object*, lean_object*);
lean_object* l_List_foldrTR___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finsupp_toMultiset(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_primeFactorsList(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_nat_pow(lean_object*, lean_object*);
lean_object* l_Nat_lcm(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_countP_go___at___00Multiset_countP___at___00Multiset_count___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_countP_go___at___00Multiset_countP___at___00Multiset_count___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__0_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_countP___at___00Multiset_count___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_countP___at___00Multiset_count___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_toFinsupp___at___00Nat_factorization_spec__0___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_toFinsupp___at___00Nat_factorization_spec__0___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__1_spec__3_spec__5_spec__6_spec__7_spec__8___lam__0(lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__1_spec__3_spec__5_spec__6_spec__7_spec__8___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__1_spec__3_spec__5_spec__6_spec__7_spec__8(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__1_spec__3_spec__5_spec__6_spec__7_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldrTR___at___00List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__1_spec__3_spec__5_spec__6_spec__7(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__1_spec__3_spec__5_spec__6___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_dedup___at___00Multiset_toFinset___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__1_spec__3(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_toFinset___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_toFinsupp___at___00Nat_factorization_spec__0___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_toFinsupp___at___00Nat_factorization_spec__0___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Multiset_toFinsupp___at___00Nat_factorization_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Multiset_toFinsupp___at___00Nat_factorization_spec__0___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Multiset_toFinsupp___at___00Nat_factorization_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Multiset_toFinsupp___at___00Nat_factorization_spec__0___closed__0_value;
static const lean_closure_object lp_mathlib_Multiset_toFinsupp___at___00Nat_factorization_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Multiset_toFinsupp___at___00Nat_factorization_spec__0___lam__2, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Multiset_toFinsupp___at___00Nat_factorization_spec__0___closed__1 = (const lean_object*)&lp_mathlib_Multiset_toFinsupp___at___00Nat_factorization_spec__0___closed__1_value;
static const lean_ctor_object lp_mathlib_Multiset_toFinsupp___at___00Nat_factorization_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Multiset_toFinsupp___at___00Nat_factorization_spec__0___closed__1_value),((lean_object*)&lp_mathlib_Multiset_toFinsupp___at___00Nat_factorization_spec__0___closed__0_value)}};
static const lean_object* lp_mathlib_Multiset_toFinsupp___at___00Nat_factorization_spec__0___closed__2 = (const lean_object*)&lp_mathlib_Multiset_toFinsupp___at___00Nat_factorization_spec__0___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_Multiset_toFinsupp___at___00Nat_factorization_spec__0 = (const lean_object*)&lp_mathlib_Multiset_toFinsupp___at___00Nat_factorization_spec__0___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Nat_factorization(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_factorization___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_count___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_count___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_countP___at___00Multiset_count___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_countP___at___00Multiset_count___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__1_spec__3_spec__5(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__1_spec__3_spec__5_spec__6(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_prod___at___00Nat_factorizationEquiv_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Multiset_prod___at___00Finset_prod___at___00Finsupp_prod___at___00Nat_factorizationEquiv_spec__0_spec__0_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Nat_mul___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Multiset_prod___at___00Finset_prod___at___00Finsupp_prod___at___00Nat_factorizationEquiv_spec__0_spec__0_spec__1___closed__0 = (const lean_object*)&lp_mathlib_Multiset_prod___at___00Finset_prod___at___00Finsupp_prod___at___00Nat_factorizationEquiv_spec__0_spec__0_spec__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Multiset_prod___at___00Finset_prod___at___00Finsupp_prod___at___00Nat_factorizationEquiv_spec__0_spec__0_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_prod___at___00Finsupp_prod___at___00Nat_factorizationEquiv_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_prod___at___00Nat_factorizationEquiv_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_factorizationEquiv___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Nat_factorizationEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Nat_pow___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Nat_factorizationEquiv___closed__0 = (const lean_object*)&lp_mathlib_Nat_factorizationEquiv___closed__0_value;
static const lean_closure_object lp_mathlib_Nat_factorizationEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Nat_factorization___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Nat_factorizationEquiv___closed__1 = (const lean_object*)&lp_mathlib_Nat_factorizationEquiv___closed__1_value;
static const lean_closure_object lp_mathlib_Nat_factorizationEquiv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Nat_factorizationEquiv___lam__0, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Nat_factorizationEquiv___closed__0_value)} };
static const lean_object* lp_mathlib_Nat_factorizationEquiv___closed__2 = (const lean_object*)&lp_mathlib_Nat_factorizationEquiv___closed__2_value;
static const lean_ctor_object lp_mathlib_Nat_factorizationEquiv___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Nat_factorizationEquiv___closed__1_value),((lean_object*)&lp_mathlib_Nat_factorizationEquiv___closed__2_value)}};
static const lean_object* lp_mathlib_Nat_factorizationEquiv___closed__3 = (const lean_object*)&lp_mathlib_Nat_factorizationEquiv___closed__3_value;
LEAN_EXPORT const lean_object* lp_mathlib_Nat_factorizationEquiv = (const lean_object*)&lp_mathlib_Nat_factorizationEquiv___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_prod___at___00Nat_factorizationEquiv_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_prod___at___00Finsupp_prod___at___00Nat_factorizationEquiv_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__0 = (const lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__0_value;
static const lean_string_object lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "termOrdProj[_]_"};
static const lean_object* lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__1 = (const lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__1_value;
static const lean_ctor_object lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__2_value_aux_0),((lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(186, 156, 7, 115, 204, 174, 239, 232)}};
static const lean_object* lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__2 = (const lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__2_value;
static const lean_string_object lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__3 = (const lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__4 = (const lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__4_value;
static const lean_string_object lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "ordProj["};
static const lean_object* lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__5 = (const lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__5_value;
static const lean_ctor_object lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__5_value)}};
static const lean_object* lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__6 = (const lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__6_value;
static const lean_string_object lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__7 = (const lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__7_value;
static const lean_ctor_object lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__7_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__8 = (const lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__8_value;
static const lean_ctor_object lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__9 = (const lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__9_value;
static const lean_ctor_object lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__4_value),((lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__6_value),((lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__9_value)}};
static const lean_object* lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__10 = (const lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__10_value;
static const lean_string_object lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "] "};
static const lean_object* lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__11 = (const lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__11_value;
static const lean_ctor_object lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__11_value)}};
static const lean_object* lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__12 = (const lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__12_value;
static const lean_ctor_object lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__4_value),((lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__10_value),((lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__12_value)}};
static const lean_object* lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__13 = (const lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__13_value;
static const lean_ctor_object lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__8_value),((lean_object*)(((size_t)(1023) << 1) | 1))}};
static const lean_object* lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__14 = (const lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__14_value;
static const lean_ctor_object lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__4_value),((lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__13_value),((lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__14_value)}};
static const lean_object* lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__15 = (const lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__15_value;
static const lean_ctor_object lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__2_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__15_value)}};
static const lean_object* lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__16 = (const lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__16_value;
LEAN_EXPORT const lean_object* lp_mathlib_Nat_termOrdProj_x5b___x5d__ = (const lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__16_value;
static const lean_string_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "term_^_"};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__0 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__0_value;
static const lean_ctor_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(71, 186, 108, 193, 152, 123, 33, 175)}};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__1 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__1_value;
static const lean_string_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "^"};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__2 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__2_value;
static const lean_string_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__3 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__3_value;
static const lean_string_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__4 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__4_value;
static const lean_string_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__5 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__5_value;
static const lean_string_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__6 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__6_value;
static const lean_ctor_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__7_value_aux_1),((lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__7_value_aux_2),((lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__7 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__7_value;
static const lean_string_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "Nat.factorization"};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__8 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__8_value;
static lean_once_cell_t lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__9;
static const lean_string_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "factorization"};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__10 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__10_value;
static const lean_ctor_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(136, 147, 255, 126, 22, 183, 173, 235)}};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__11 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__11_value;
static const lean_ctor_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__11_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__12 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__12_value;
static const lean_ctor_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__12_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__13 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__13_value;
static const lean_string_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__14 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__14_value;
static const lean_ctor_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__15 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__15_value;
LEAN_EXPORT lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Nat_termOrdCompl_x5b___x5d___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "termOrdCompl[_]_"};
static const lean_object* lp_mathlib_Nat_termOrdCompl_x5b___x5d___00__closed__0 = (const lean_object*)&lp_mathlib_Nat_termOrdCompl_x5b___x5d___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Nat_termOrdCompl_x5b___x5d___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib_Nat_termOrdCompl_x5b___x5d___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Nat_termOrdCompl_x5b___x5d___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Nat_termOrdCompl_x5b___x5d___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(55, 182, 164, 246, 118, 15, 234, 149)}};
static const lean_object* lp_mathlib_Nat_termOrdCompl_x5b___x5d___00__closed__1 = (const lean_object*)&lp_mathlib_Nat_termOrdCompl_x5b___x5d___00__closed__1_value;
static const lean_string_object lp_mathlib_Nat_termOrdCompl_x5b___x5d___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "ordCompl["};
static const lean_object* lp_mathlib_Nat_termOrdCompl_x5b___x5d___00__closed__2 = (const lean_object*)&lp_mathlib_Nat_termOrdCompl_x5b___x5d___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Nat_termOrdCompl_x5b___x5d___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Nat_termOrdCompl_x5b___x5d___00__closed__2_value)}};
static const lean_object* lp_mathlib_Nat_termOrdCompl_x5b___x5d___00__closed__3 = (const lean_object*)&lp_mathlib_Nat_termOrdCompl_x5b___x5d___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Nat_termOrdCompl_x5b___x5d___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__4_value),((lean_object*)&lp_mathlib_Nat_termOrdCompl_x5b___x5d___00__closed__3_value),((lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__9_value)}};
static const lean_object* lp_mathlib_Nat_termOrdCompl_x5b___x5d___00__closed__4 = (const lean_object*)&lp_mathlib_Nat_termOrdCompl_x5b___x5d___00__closed__4_value;
static const lean_ctor_object lp_mathlib_Nat_termOrdCompl_x5b___x5d___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__4_value),((lean_object*)&lp_mathlib_Nat_termOrdCompl_x5b___x5d___00__closed__4_value),((lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__12_value)}};
static const lean_object* lp_mathlib_Nat_termOrdCompl_x5b___x5d___00__closed__5 = (const lean_object*)&lp_mathlib_Nat_termOrdCompl_x5b___x5d___00__closed__5_value;
static const lean_ctor_object lp_mathlib_Nat_termOrdCompl_x5b___x5d___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__4_value),((lean_object*)&lp_mathlib_Nat_termOrdCompl_x5b___x5d___00__closed__5_value),((lean_object*)&lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__14_value)}};
static const lean_object* lp_mathlib_Nat_termOrdCompl_x5b___x5d___00__closed__6 = (const lean_object*)&lp_mathlib_Nat_termOrdCompl_x5b___x5d___00__closed__6_value;
static const lean_ctor_object lp_mathlib_Nat_termOrdCompl_x5b___x5d___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Nat_termOrdCompl_x5b___x5d___00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Nat_termOrdCompl_x5b___x5d___00__closed__6_value)}};
static const lean_object* lp_mathlib_Nat_termOrdCompl_x5b___x5d___00__closed__7 = (const lean_object*)&lp_mathlib_Nat_termOrdCompl_x5b___x5d___00__closed__7_value;
LEAN_EXPORT const lean_object* lp_mathlib_Nat_termOrdCompl_x5b___x5d__ = (const lean_object*)&lp_mathlib_Nat_termOrdCompl_x5b___x5d___00__closed__7_value;
static const lean_string_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdCompl_x5b___x5d____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "term_/_"};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdCompl_x5b___x5d____1___closed__0 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdCompl_x5b___x5d____1___closed__0_value;
static const lean_ctor_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdCompl_x5b___x5d____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdCompl_x5b___x5d____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(210, 175, 43, 55, 191, 201, 132, 176)}};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdCompl_x5b___x5d____1___closed__1 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdCompl_x5b___x5d____1___closed__1_value;
static const lean_string_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdCompl_x5b___x5d____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "/"};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdCompl_x5b___x5d____1___closed__2 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdCompl_x5b___x5d____1___closed__2_value;
static const lean_string_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdCompl_x5b___x5d____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdCompl_x5b___x5d____1___closed__3 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdCompl_x5b___x5d____1___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdCompl_x5b___x5d____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdCompl_x5b___x5d____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_factorizationLCMLeft___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_factorizationLCMLeft___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_factorizationLCMLeft___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Nat_factorizationLCMLeft___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Nat_factorizationLCMLeft___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Nat_factorizationLCMLeft___closed__0 = (const lean_object*)&lp_mathlib_Nat_factorizationLCMLeft___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Nat_factorizationLCMLeft(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_factorizationLCMRight___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_factorizationLCMRight___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_factorizationLCMRight(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_countP_go___at___00Multiset_countP___at___00Multiset_count___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__0_spec__1_spec__2(lean_object* v_a_1_, lean_object* v_a_2_, lean_object* v_a_3_){
_start:
{
if (lean_obj_tag(v_a_2_) == 0)
{
return v_a_3_;
}
else
{
lean_object* v_head_4_; lean_object* v_tail_5_; uint8_t v___x_6_; 
v_head_4_ = lean_ctor_get(v_a_2_, 0);
v_tail_5_ = lean_ctor_get(v_a_2_, 1);
v___x_6_ = lean_nat_dec_eq(v_a_1_, v_head_4_);
if (v___x_6_ == 0)
{
v_a_2_ = v_tail_5_;
goto _start;
}
else
{
lean_object* v___x_8_; lean_object* v___x_9_; 
v___x_8_ = lean_unsigned_to_nat(1u);
v___x_9_ = lean_nat_add(v_a_3_, v___x_8_);
lean_dec(v_a_3_);
v_a_2_ = v_tail_5_;
v_a_3_ = v___x_9_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_countP_go___at___00Multiset_countP___at___00Multiset_count___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__0_spec__1_spec__2___boxed(lean_object* v_a_11_, lean_object* v_a_12_, lean_object* v_a_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_List_countP_go___at___00Multiset_countP___at___00Multiset_count___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__0_spec__1_spec__2(v_a_11_, v_a_12_, v_a_13_);
lean_dec(v_a_12_);
lean_dec(v_a_11_);
return v_res_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_countP___at___00Multiset_count___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__0_spec__1___redArg(lean_object* v_a_15_, lean_object* v_s_16_){
_start:
{
lean_object* v___x_17_; lean_object* v___x_18_; 
v___x_17_ = lean_unsigned_to_nat(0u);
v___x_18_ = lp_mathlib_List_countP_go___at___00Multiset_countP___at___00Multiset_count___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__0_spec__1_spec__2(v_a_15_, v_s_16_, v___x_17_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_countP___at___00Multiset_count___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_a_19_, lean_object* v_s_20_){
_start:
{
lean_object* v_res_21_; 
v_res_21_ = lp_mathlib_Multiset_countP___at___00Multiset_count___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__0_spec__1___redArg(v_a_19_, v_s_20_);
lean_dec(v_s_20_);
lean_dec(v_a_19_);
return v_res_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_toFinsupp___at___00Nat_factorization_spec__0___lam__1(lean_object* v_s_22_, lean_object* v_a_23_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = lp_mathlib_Multiset_countP___at___00Multiset_count___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__0_spec__1___redArg(v_a_23_, v_s_22_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_toFinsupp___at___00Nat_factorization_spec__0___lam__1___boxed(lean_object* v_s_25_, lean_object* v_a_26_){
_start:
{
lean_object* v_res_27_; 
v_res_27_ = lp_mathlib_Multiset_toFinsupp___at___00Nat_factorization_spec__0___lam__1(v_s_25_, v_a_26_);
lean_dec(v_a_26_);
lean_dec(v_s_25_);
return v_res_27_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__1_spec__3_spec__5_spec__6_spec__7_spec__8___lam__0(lean_object* v___x_28_, uint8_t v___x_29_, lean_object* v___y_30_){
_start:
{
uint8_t v___x_31_; 
v___x_31_ = lean_nat_dec_eq(v___x_28_, v___y_30_);
if (v___x_31_ == 0)
{
uint8_t v___x_32_; 
v___x_32_ = 1;
return v___x_32_;
}
else
{
return v___x_29_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__1_spec__3_spec__5_spec__6_spec__7_spec__8___lam__0___boxed(lean_object* v___x_33_, lean_object* v___x_34_, lean_object* v___y_35_){
_start:
{
uint8_t v___x_407__boxed_36_; uint8_t v_res_37_; lean_object* v_r_38_; 
v___x_407__boxed_36_ = lean_unbox(v___x_34_);
v_res_37_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__1_spec__3_spec__5_spec__6_spec__7_spec__8___lam__0(v___x_33_, v___x_407__boxed_36_, v___y_35_);
lean_dec(v___y_35_);
lean_dec(v___x_33_);
v_r_38_ = lean_box(v_res_37_);
return v_r_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__1_spec__3_spec__5_spec__6_spec__7_spec__8(lean_object* v_as_39_, size_t v_i_40_, size_t v_stop_41_, lean_object* v_b_42_){
_start:
{
uint8_t v___x_43_; 
v___x_43_ = lean_usize_dec_eq(v_i_40_, v_stop_41_);
if (v___x_43_ == 0)
{
size_t v___x_44_; size_t v___x_45_; lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___f_48_; uint8_t v___x_49_; 
v___x_44_ = ((size_t)1ULL);
v___x_45_ = lean_usize_sub(v_i_40_, v___x_44_);
v___x_46_ = lean_array_uget_borrowed(v_as_39_, v___x_45_);
v___x_47_ = lean_box(v___x_43_);
lean_inc(v___x_46_);
v___f_48_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__1_spec__3_spec__5_spec__6_spec__7_spec__8___lam__0___boxed), 3, 2);
lean_closure_set(v___f_48_, 0, v___x_46_);
lean_closure_set(v___f_48_, 1, v___x_47_);
lean_inc(v_b_42_);
v___x_49_ = l_List_decidableBAll___redArg(v___f_48_, v_b_42_);
if (v___x_49_ == 0)
{
v_i_40_ = v___x_45_;
goto _start;
}
else
{
lean_object* v___x_51_; 
lean_inc(v___x_46_);
v___x_51_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_51_, 0, v___x_46_);
lean_ctor_set(v___x_51_, 1, v_b_42_);
v_i_40_ = v___x_45_;
v_b_42_ = v___x_51_;
goto _start;
}
}
else
{
return v_b_42_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__1_spec__3_spec__5_spec__6_spec__7_spec__8___boxed(lean_object* v_as_53_, lean_object* v_i_54_, lean_object* v_stop_55_, lean_object* v_b_56_){
_start:
{
size_t v_i_boxed_57_; size_t v_stop_boxed_58_; lean_object* v_res_59_; 
v_i_boxed_57_ = lean_unbox_usize(v_i_54_);
lean_dec(v_i_54_);
v_stop_boxed_58_ = lean_unbox_usize(v_stop_55_);
lean_dec(v_stop_55_);
v_res_59_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__1_spec__3_spec__5_spec__6_spec__7_spec__8(v_as_53_, v_i_boxed_57_, v_stop_boxed_58_, v_b_56_);
lean_dec_ref(v_as_53_);
return v_res_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldrTR___at___00List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__1_spec__3_spec__5_spec__6_spec__7(lean_object* v_init_60_, lean_object* v_l_61_){
_start:
{
lean_object* v___x_62_; lean_object* v___x_63_; lean_object* v___x_64_; uint8_t v___x_65_; 
v___x_62_ = lean_array_mk(v_l_61_);
v___x_63_ = lean_array_get_size(v___x_62_);
v___x_64_ = lean_unsigned_to_nat(0u);
v___x_65_ = lean_nat_dec_lt(v___x_64_, v___x_63_);
if (v___x_65_ == 0)
{
lean_dec_ref(v___x_62_);
return v_init_60_;
}
else
{
size_t v___x_66_; size_t v___x_67_; lean_object* v___x_68_; 
v___x_66_ = lean_usize_of_nat(v___x_63_);
v___x_67_ = ((size_t)0ULL);
v___x_68_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__1_spec__3_spec__5_spec__6_spec__7_spec__8(v___x_62_, v___x_66_, v___x_67_, v_init_60_);
lean_dec_ref(v___x_62_);
return v___x_68_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__1_spec__3_spec__5_spec__6___redArg(lean_object* v_l_69_){
_start:
{
lean_object* v___x_70_; lean_object* v___x_71_; 
v___x_70_ = lean_box(0);
v___x_71_ = lp_mathlib_List_foldrTR___at___00List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__1_spec__3_spec__5_spec__6_spec__7(v___x_70_, v_l_69_);
return v___x_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_dedup___at___00Multiset_toFinset___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__1_spec__3(lean_object* v_s_72_){
_start:
{
lean_object* v___x_73_; 
v___x_73_ = lp_mathlib_List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__1_spec__3_spec__5_spec__6___redArg(v_s_72_);
return v___x_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_toFinset___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__1(lean_object* v_s_74_){
_start:
{
lean_object* v___x_75_; 
v___x_75_ = lp_mathlib_List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__1_spec__3_spec__5_spec__6___redArg(v_s_74_);
return v___x_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_toFinsupp___at___00Nat_factorization_spec__0___lam__2(lean_object* v_s_76_){
_start:
{
lean_object* v___f_77_; lean_object* v___x_78_; lean_object* v___x_79_; 
lean_inc(v_s_76_);
v___f_77_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_toFinsupp___at___00Nat_factorization_spec__0___lam__1___boxed), 2, 1);
lean_closure_set(v___f_77_, 0, v_s_76_);
v___x_78_ = lp_mathlib_List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__1_spec__3_spec__5_spec__6___redArg(v_s_76_);
v___x_79_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_79_, 0, v___x_78_);
lean_ctor_set(v___x_79_, 1, v___f_77_);
return v___x_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_toFinsupp___at___00Nat_factorization_spec__0___lam__0(lean_object* v_f_80_){
_start:
{
lean_object* v___x_370__overap_81_; lean_object* v___x_82_; 
v___x_370__overap_81_ = lp_mathlib_Finsupp_toMultiset(lean_box(0));
v___x_82_ = lean_apply_1(v___x_370__overap_81_, v_f_80_);
return v___x_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_factorization(lean_object* v_n_89_){
_start:
{
lean_object* v___x_90_; lean_object* v_toFun_91_; lean_object* v___x_92_; lean_object* v___x_93_; 
v___x_90_ = ((lean_object*)(lp_mathlib_Multiset_toFinsupp___at___00Nat_factorization_spec__0));
v_toFun_91_ = lean_ctor_get(v___x_90_, 0);
v___x_92_ = lp_mathlib_Nat_primeFactorsList(v_n_89_);
lean_inc(v_toFun_91_);
v___x_93_ = lean_apply_1(v_toFun_91_, v___x_92_);
return v___x_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_factorization___boxed(lean_object* v_n_94_){
_start:
{
lean_object* v_res_95_; 
v_res_95_ = lp_mathlib_Nat_factorization(v_n_94_);
lean_dec(v_n_94_);
return v_res_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_count___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__0(lean_object* v_a_96_, lean_object* v_s_97_){
_start:
{
lean_object* v___x_98_; 
v___x_98_ = lp_mathlib_Multiset_countP___at___00Multiset_count___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__0_spec__1___redArg(v_a_96_, v_s_97_);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_count___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__0___boxed(lean_object* v_a_99_, lean_object* v_s_100_){
_start:
{
lean_object* v_res_101_; 
v_res_101_ = lp_mathlib_Multiset_count___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__0(v_a_99_, v_s_100_);
lean_dec(v_s_100_);
lean_dec(v_a_99_);
return v_res_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_countP___at___00Multiset_count___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__0_spec__1(lean_object* v_a_102_, lean_object* v_p_103_, lean_object* v_s_104_){
_start:
{
lean_object* v___x_105_; 
v___x_105_ = lp_mathlib_Multiset_countP___at___00Multiset_count___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__0_spec__1___redArg(v_a_102_, v_s_104_);
return v___x_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_countP___at___00Multiset_count___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__0_spec__1___boxed(lean_object* v_a_106_, lean_object* v_p_107_, lean_object* v_s_108_){
_start:
{
lean_object* v_res_109_; 
v_res_109_ = lp_mathlib_Multiset_countP___at___00Multiset_count___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__0_spec__1(v_a_106_, v_p_107_, v_s_108_);
lean_dec(v_s_108_);
lean_dec(v_a_106_);
return v_res_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__1_spec__3_spec__5(lean_object* v_l_110_){
_start:
{
lean_object* v___x_111_; 
v___x_111_ = lp_mathlib_List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__1_spec__3_spec__5_spec__6___redArg(v_l_110_);
return v___x_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__1_spec__3_spec__5_spec__6(lean_object* v_R_112_, lean_object* v_l_113_){
_start:
{
lean_object* v___x_114_; 
v___x_114_ = lp_mathlib_List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00Multiset_toFinsupp___at___00Nat_factorization_spec__0_spec__1_spec__3_spec__5_spec__6___redArg(v_l_113_);
return v___x_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_prod___at___00Nat_factorizationEquiv_spec__0___redArg___lam__0(lean_object* v_toFun_115_, lean_object* v_g_116_, lean_object* v_a_117_){
_start:
{
lean_object* v___x_118_; lean_object* v___x_119_; 
lean_inc(v_a_117_);
v___x_118_ = lean_apply_1(v_toFun_115_, v_a_117_);
v___x_119_ = lean_apply_2(v_g_116_, v_a_117_, v___x_118_);
return v___x_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_prod___at___00Finset_prod___at___00Finsupp_prod___at___00Nat_factorizationEquiv_spec__0_spec__0_spec__1(lean_object* v_s_121_){
_start:
{
lean_object* v___f_122_; lean_object* v___x_123_; lean_object* v___x_124_; 
v___f_122_ = ((lean_object*)(lp_mathlib_Multiset_prod___at___00Finset_prod___at___00Finsupp_prod___at___00Nat_factorizationEquiv_spec__0_spec__0_spec__1___closed__0));
v___x_123_ = lean_unsigned_to_nat(1u);
v___x_124_ = l_List_foldrTR___redArg(v___f_122_, v___x_123_, v_s_121_);
return v___x_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_prod___at___00Finsupp_prod___at___00Nat_factorizationEquiv_spec__0_spec__0___redArg(lean_object* v_s_125_, lean_object* v_f_126_){
_start:
{
lean_object* v___x_127_; lean_object* v___x_128_; 
v___x_127_ = lp_mathlib_Multiset_map___redArg(v_f_126_, v_s_125_);
v___x_128_ = lp_mathlib_Multiset_prod___at___00Finset_prod___at___00Finsupp_prod___at___00Nat_factorizationEquiv_spec__0_spec__0_spec__1(v___x_127_);
return v___x_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_prod___at___00Nat_factorizationEquiv_spec__0___redArg(lean_object* v_f_129_, lean_object* v_g_130_){
_start:
{
lean_object* v_support_131_; lean_object* v_toFun_132_; lean_object* v___f_133_; lean_object* v___x_134_; 
v_support_131_ = lean_ctor_get(v_f_129_, 0);
lean_inc(v_support_131_);
v_toFun_132_ = lean_ctor_get(v_f_129_, 1);
lean_inc(v_toFun_132_);
lean_dec_ref(v_f_129_);
v___f_133_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_prod___at___00Nat_factorizationEquiv_spec__0___redArg___lam__0), 3, 2);
lean_closure_set(v___f_133_, 0, v_toFun_132_);
lean_closure_set(v___f_133_, 1, v_g_130_);
v___x_134_ = lp_mathlib_Finset_prod___at___00Finsupp_prod___at___00Nat_factorizationEquiv_spec__0_spec__0___redArg(v_support_131_, v___f_133_);
return v___x_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_factorizationEquiv___lam__0(lean_object* v___f_135_, lean_object* v_x_136_){
_start:
{
lean_object* v___x_137_; 
v___x_137_ = lp_mathlib_Finsupp_prod___at___00Nat_factorizationEquiv_spec__0___redArg(v_x_136_, v___f_135_);
return v___x_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_prod___at___00Nat_factorizationEquiv_spec__0(lean_object* v_00_u03b1_146_, lean_object* v_f_147_, lean_object* v_g_148_){
_start:
{
lean_object* v___x_149_; 
v___x_149_ = lp_mathlib_Finsupp_prod___at___00Nat_factorizationEquiv_spec__0___redArg(v_f_147_, v_g_148_);
return v___x_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_prod___at___00Finsupp_prod___at___00Nat_factorizationEquiv_spec__0_spec__0(lean_object* v_00_u03b9_150_, lean_object* v_s_151_, lean_object* v_f_152_){
_start:
{
lean_object* v___x_153_; 
v___x_153_ = lp_mathlib_Finset_prod___at___00Finsupp_prod___at___00Nat_factorizationEquiv_spec__0_spec__0___redArg(v_s_151_, v_f_152_);
return v___x_153_;
}
}
static lean_object* _init_lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__9(void){
_start:
{
lean_object* v___x_208_; lean_object* v___x_209_; 
v___x_208_ = ((lean_object*)(lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__8));
v___x_209_ = l_String_toRawSubstring_x27(v___x_208_);
return v___x_209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1(lean_object* v_x_223_, lean_object* v_a_224_, lean_object* v_a_225_){
_start:
{
lean_object* v___x_226_; uint8_t v___x_227_; 
v___x_226_ = ((lean_object*)(lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__2));
lean_inc(v_x_223_);
v___x_227_ = l_Lean_Syntax_isOfKind(v_x_223_, v___x_226_);
if (v___x_227_ == 0)
{
lean_object* v___x_228_; lean_object* v___x_229_; 
lean_dec(v_x_223_);
v___x_228_ = lean_box(1);
v___x_229_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_229_, 0, v___x_228_);
lean_ctor_set(v___x_229_, 1, v_a_225_);
return v___x_229_;
}
else
{
lean_object* v_quotContext_230_; lean_object* v_currMacroScope_231_; lean_object* v_ref_232_; lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; uint8_t v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; 
v_quotContext_230_ = lean_ctor_get(v_a_224_, 1);
v_currMacroScope_231_ = lean_ctor_get(v_a_224_, 2);
v_ref_232_ = lean_ctor_get(v_a_224_, 5);
v___x_233_ = lean_unsigned_to_nat(1u);
v___x_234_ = l_Lean_Syntax_getArg(v_x_223_, v___x_233_);
v___x_235_ = lean_unsigned_to_nat(3u);
v___x_236_ = l_Lean_Syntax_getArg(v_x_223_, v___x_235_);
lean_dec(v_x_223_);
v___x_237_ = 0;
v___x_238_ = l_Lean_SourceInfo_fromRef(v_ref_232_, v___x_237_);
v___x_239_ = ((lean_object*)(lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__1));
v___x_240_ = ((lean_object*)(lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__2));
lean_inc_n(v___x_238_, 4);
v___x_241_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_241_, 0, v___x_238_);
lean_ctor_set(v___x_241_, 1, v___x_240_);
v___x_242_ = ((lean_object*)(lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__7));
v___x_243_ = lean_obj_once(&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__9, &lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__9_once, _init_lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__9);
v___x_244_ = ((lean_object*)(lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__11));
lean_inc(v_currMacroScope_231_);
lean_inc(v_quotContext_230_);
v___x_245_ = l_Lean_addMacroScope(v_quotContext_230_, v___x_244_, v_currMacroScope_231_);
v___x_246_ = ((lean_object*)(lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__13));
v___x_247_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_247_, 0, v___x_238_);
lean_ctor_set(v___x_247_, 1, v___x_243_);
lean_ctor_set(v___x_247_, 2, v___x_245_);
lean_ctor_set(v___x_247_, 3, v___x_246_);
v___x_248_ = ((lean_object*)(lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___closed__15));
lean_inc(v___x_234_);
v___x_249_ = l_Lean_Syntax_node2(v___x_238_, v___x_248_, v___x_236_, v___x_234_);
v___x_250_ = l_Lean_Syntax_node2(v___x_238_, v___x_242_, v___x_247_, v___x_249_);
v___x_251_ = l_Lean_Syntax_node3(v___x_238_, v___x_239_, v___x_234_, v___x_241_, v___x_250_);
v___x_252_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_252_, 0, v___x_251_);
lean_ctor_set(v___x_252_, 1, v_a_225_);
return v___x_252_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1___boxed(lean_object* v_x_253_, lean_object* v_a_254_, lean_object* v_a_255_){
_start:
{
lean_object* v_res_256_; 
v_res_256_ = lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdProj_x5b___x5d____1(v_x_253_, v_a_254_, v_a_255_);
lean_dec_ref(v_a_254_);
return v_res_256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdCompl_x5b___x5d____1(lean_object* v_x_286_, lean_object* v_a_287_, lean_object* v_a_288_){
_start:
{
lean_object* v___x_289_; uint8_t v___x_290_; 
v___x_289_ = ((lean_object*)(lp_mathlib_Nat_termOrdCompl_x5b___x5d___00__closed__1));
lean_inc(v_x_286_);
v___x_290_ = l_Lean_Syntax_isOfKind(v_x_286_, v___x_289_);
if (v___x_290_ == 0)
{
lean_object* v___x_291_; lean_object* v___x_292_; 
lean_dec(v_x_286_);
v___x_291_ = lean_box(1);
v___x_292_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_292_, 0, v___x_291_);
lean_ctor_set(v___x_292_, 1, v_a_288_);
return v___x_292_;
}
else
{
lean_object* v_ref_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; uint8_t v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; 
v_ref_293_ = lean_ctor_get(v_a_287_, 5);
v___x_294_ = lean_unsigned_to_nat(1u);
v___x_295_ = l_Lean_Syntax_getArg(v_x_286_, v___x_294_);
v___x_296_ = lean_unsigned_to_nat(3u);
v___x_297_ = l_Lean_Syntax_getArg(v_x_286_, v___x_296_);
lean_dec(v_x_286_);
v___x_298_ = 0;
v___x_299_ = l_Lean_SourceInfo_fromRef(v_ref_293_, v___x_298_);
v___x_300_ = ((lean_object*)(lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdCompl_x5b___x5d____1___closed__1));
v___x_301_ = ((lean_object*)(lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdCompl_x5b___x5d____1___closed__2));
lean_inc_n(v___x_299_, 4);
v___x_302_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_302_, 0, v___x_299_);
lean_ctor_set(v___x_302_, 1, v___x_301_);
v___x_303_ = ((lean_object*)(lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__2));
v___x_304_ = ((lean_object*)(lp_mathlib_Nat_termOrdProj_x5b___x5d___00__closed__5));
v___x_305_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_305_, 0, v___x_299_);
lean_ctor_set(v___x_305_, 1, v___x_304_);
v___x_306_ = ((lean_object*)(lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdCompl_x5b___x5d____1___closed__3));
v___x_307_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_307_, 0, v___x_299_);
lean_ctor_set(v___x_307_, 1, v___x_306_);
lean_inc(v___x_297_);
v___x_308_ = l_Lean_Syntax_node4(v___x_299_, v___x_303_, v___x_305_, v___x_295_, v___x_307_, v___x_297_);
v___x_309_ = l_Lean_Syntax_node3(v___x_299_, v___x_300_, v___x_297_, v___x_302_, v___x_308_);
v___x_310_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_310_, 0, v___x_309_);
lean_ctor_set(v___x_310_, 1, v_a_288_);
return v___x_310_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdCompl_x5b___x5d____1___boxed(lean_object* v_x_311_, lean_object* v_a_312_, lean_object* v_a_313_){
_start:
{
lean_object* v_res_314_; 
v_res_314_ = lp_mathlib_Nat___aux__Mathlib__Data__Nat__Factorization__Defs______macroRules__Nat__termOrdCompl_x5b___x5d____1(v_x_311_, v_a_312_, v_a_313_);
lean_dec_ref(v_a_312_);
return v_res_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_factorizationLCMLeft___lam__0(lean_object* v_self_315_, lean_object* v___y_316_){
_start:
{
lean_object* v_toFun_317_; lean_object* v___x_318_; 
v_toFun_317_ = lean_ctor_get(v_self_315_, 1);
lean_inc(v_toFun_317_);
lean_dec_ref(v_self_315_);
v___x_318_ = lean_apply_1(v_toFun_317_, v___y_316_);
return v___x_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_factorizationLCMLeft___lam__1(lean_object* v_b_319_, lean_object* v___f_320_, lean_object* v_a_321_, lean_object* v_p_322_, lean_object* v_n_323_){
_start:
{
lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; uint8_t v___x_328_; 
v___x_324_ = lp_mathlib_Nat_factorization(v_b_319_);
lean_inc_ref(v___f_320_);
lean_inc_n(v_p_322_, 2);
v___x_325_ = lean_apply_2(v___f_320_, v___x_324_, v_p_322_);
v___x_326_ = lp_mathlib_Nat_factorization(v_a_321_);
v___x_327_ = lean_apply_2(v___f_320_, v___x_326_, v_p_322_);
v___x_328_ = lean_nat_dec_le(v___x_325_, v___x_327_);
lean_dec(v___x_327_);
lean_dec(v___x_325_);
if (v___x_328_ == 0)
{
lean_object* v___x_329_; 
lean_dec(v_p_322_);
v___x_329_ = lean_unsigned_to_nat(1u);
return v___x_329_;
}
else
{
lean_object* v___x_330_; 
v___x_330_ = lean_nat_pow(v_p_322_, v_n_323_);
lean_dec(v_p_322_);
return v___x_330_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_factorizationLCMLeft___lam__1___boxed(lean_object* v_b_331_, lean_object* v___f_332_, lean_object* v_a_333_, lean_object* v_p_334_, lean_object* v_n_335_){
_start:
{
lean_object* v_res_336_; 
v_res_336_ = lp_mathlib_Nat_factorizationLCMLeft___lam__1(v_b_331_, v___f_332_, v_a_333_, v_p_334_, v_n_335_);
lean_dec(v_n_335_);
lean_dec(v_a_333_);
lean_dec(v_b_331_);
return v_res_336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_factorizationLCMLeft(lean_object* v_a_338_, lean_object* v_b_339_){
_start:
{
lean_object* v___f_340_; lean_object* v___f_341_; lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; 
v___f_340_ = ((lean_object*)(lp_mathlib_Nat_factorizationLCMLeft___closed__0));
lean_inc(v_a_338_);
lean_inc(v_b_339_);
v___f_341_ = lean_alloc_closure((void*)(lp_mathlib_Nat_factorizationLCMLeft___lam__1___boxed), 5, 3);
lean_closure_set(v___f_341_, 0, v_b_339_);
lean_closure_set(v___f_341_, 1, v___f_340_);
lean_closure_set(v___f_341_, 2, v_a_338_);
v___x_342_ = l_Nat_lcm(v_a_338_, v_b_339_);
lean_dec(v_b_339_);
lean_dec(v_a_338_);
v___x_343_ = lp_mathlib_Nat_factorization(v___x_342_);
lean_dec(v___x_342_);
v___x_344_ = lp_mathlib_Finsupp_prod___at___00Nat_factorizationEquiv_spec__0___redArg(v___x_343_, v___f_341_);
return v___x_344_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_factorizationLCMRight___lam__1(lean_object* v_b_345_, lean_object* v___f_346_, lean_object* v_a_347_, lean_object* v_p_348_, lean_object* v_n_349_){
_start:
{
lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___x_353_; uint8_t v___x_354_; 
v___x_350_ = lp_mathlib_Nat_factorization(v_b_345_);
lean_inc_ref(v___f_346_);
lean_inc_n(v_p_348_, 2);
v___x_351_ = lean_apply_2(v___f_346_, v___x_350_, v_p_348_);
v___x_352_ = lp_mathlib_Nat_factorization(v_a_347_);
v___x_353_ = lean_apply_2(v___f_346_, v___x_352_, v_p_348_);
v___x_354_ = lean_nat_dec_le(v___x_351_, v___x_353_);
lean_dec(v___x_353_);
lean_dec(v___x_351_);
if (v___x_354_ == 0)
{
lean_object* v___x_355_; 
v___x_355_ = lean_nat_pow(v_p_348_, v_n_349_);
lean_dec(v_p_348_);
return v___x_355_;
}
else
{
lean_object* v___x_356_; 
lean_dec(v_p_348_);
v___x_356_ = lean_unsigned_to_nat(1u);
return v___x_356_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_factorizationLCMRight___lam__1___boxed(lean_object* v_b_357_, lean_object* v___f_358_, lean_object* v_a_359_, lean_object* v_p_360_, lean_object* v_n_361_){
_start:
{
lean_object* v_res_362_; 
v_res_362_ = lp_mathlib_Nat_factorizationLCMRight___lam__1(v_b_357_, v___f_358_, v_a_359_, v_p_360_, v_n_361_);
lean_dec(v_n_361_);
lean_dec(v_a_359_);
lean_dec(v_b_357_);
return v_res_362_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_factorizationLCMRight(lean_object* v_a_363_, lean_object* v_b_364_){
_start:
{
lean_object* v___f_365_; lean_object* v___f_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; 
v___f_365_ = ((lean_object*)(lp_mathlib_Nat_factorizationLCMLeft___closed__0));
lean_inc(v_a_363_);
lean_inc(v_b_364_);
v___f_366_ = lean_alloc_closure((void*)(lp_mathlib_Nat_factorizationLCMRight___lam__1___boxed), 5, 3);
lean_closure_set(v___f_366_, 0, v_b_364_);
lean_closure_set(v___f_366_, 1, v___f_365_);
lean_closure_set(v___f_366_, 2, v_a_363_);
v___x_367_ = l_Nat_lcm(v_a_363_, v_b_364_);
lean_dec(v_b_364_);
lean_dec(v_a_363_);
v___x_368_ = lp_mathlib_Nat_factorization(v___x_367_);
lean_dec(v___x_367_);
v___x_369_ = lp_mathlib_Finsupp_prod___at___00Nat_factorizationEquiv_spec__0___redArg(v___x_368_, v___f_366_);
return v___x_369_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Data_List_Count(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finsupp_Multiset(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finsupp_Order(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_PrimeFin(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_NumberTheory_Padics_PadicVal_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Factorization_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_List_Count(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finsupp_Multiset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finsupp_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_PrimeFin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_NumberTheory_Padics_PadicVal_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Nat_Factorization_Defs(uint8_t builtin) {
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
lean_object* initialize_batteries_Batteries_Data_List_Count(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finsupp_Multiset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finsupp_Order(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_PrimeFin(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_NumberTheory_Padics_PadicVal_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Nat_Factorization_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_List_Count(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finsupp_Multiset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finsupp_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_PrimeFin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_NumberTheory_Padics_PadicVal_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Factorization_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Nat_Factorization_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Nat_Factorization_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
