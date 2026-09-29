// Lean compiler output
// Module: Mathlib.Algebra.Polynomial.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.AddChar public import Mathlib.Algebra.Group.Submonoid.Operations public import Mathlib.Algebra.MonoidAlgebra.Module public import Mathlib.Algebra.MonoidAlgebra.NoZeroDivisors public import Mathlib.Algebra.Order.Monoid.Unbundled.WithTop public import Mathlib.Algebra.Ring.Action.Rat public import Mathlib.Data.Finset.Sort public import Mathlib.Tactic.FastInstance public import Mathlib.LinearAlgebra.Finsupp.LSum public import Mathlib.Algebra.Order.Group.Nat
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
lean_object* l_instDecidableEqNat___boxed(lean_object*, lean_object*);
uint8_t lp_mathlib_AddMonoidAlgebra_instDecidableEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_Function_Injective_decidableEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instNatCastNat___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
lean_object* l_Std_instToFormatFormat___lam__0___boxed(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lp_mathlib_WithTop_natCast___redArg___lam__0(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Nat_decLe___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_sort___redArg(lean_object*, lean_object*);
lean_object* l_List_mapTR_loop___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_WithTop_decidableLE___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_string_length(lean_object*);
lean_object* lean_nat_to_int(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* l_Std_Format_joinSep___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Std_Format_fill(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* lp_mathlib_Finset_sum___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Polynomial_term___x5bX_x5d___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Polynomial"};
static const lean_object* lp_mathlib_Polynomial_term___x5bX_x5d___closed__0 = (const lean_object*)&lp_mathlib_Polynomial_term___x5bX_x5d___closed__0_value;
static const lean_string_object lp_mathlib_Polynomial_term___x5bX_x5d___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "term_[X]"};
static const lean_object* lp_mathlib_Polynomial_term___x5bX_x5d___closed__1 = (const lean_object*)&lp_mathlib_Polynomial_term___x5bX_x5d___closed__1_value;
static const lean_ctor_object lp_mathlib_Polynomial_term___x5bX_x5d___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Polynomial_term___x5bX_x5d___closed__0_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Polynomial_term___x5bX_x5d___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial_term___x5bX_x5d___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Polynomial_term___x5bX_x5d___closed__1_value),LEAN_SCALAR_PTR_LITERAL(252, 100, 91, 91, 58, 10, 246, 213)}};
static const lean_object* lp_mathlib_Polynomial_term___x5bX_x5d___closed__2 = (const lean_object*)&lp_mathlib_Polynomial_term___x5bX_x5d___closed__2_value;
static const lean_string_object lp_mathlib_Polynomial_term___x5bX_x5d___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "[X]"};
static const lean_object* lp_mathlib_Polynomial_term___x5bX_x5d___closed__3 = (const lean_object*)&lp_mathlib_Polynomial_term___x5bX_x5d___closed__3_value;
static const lean_ctor_object lp_mathlib_Polynomial_term___x5bX_x5d___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial_term___x5bX_x5d___closed__3_value)}};
static const lean_object* lp_mathlib_Polynomial_term___x5bX_x5d___closed__4 = (const lean_object*)&lp_mathlib_Polynomial_term___x5bX_x5d___closed__4_value;
static const lean_ctor_object lp_mathlib_Polynomial_term___x5bX_x5d___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial_term___x5bX_x5d___closed__2_value),((lean_object*)(((size_t)(9000) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Polynomial_term___x5bX_x5d___closed__4_value)}};
static const lean_object* lp_mathlib_Polynomial_term___x5bX_x5d___closed__5 = (const lean_object*)&lp_mathlib_Polynomial_term___x5bX_x5d___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Polynomial_term___x5bX_x5d = (const lean_object*)&lp_mathlib_Polynomial_term___x5bX_x5d___closed__5_value;
static const lean_string_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__0 = (const lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__0_value;
static const lean_string_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__1 = (const lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__1_value;
static const lean_string_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__2 = (const lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__2_value;
static const lean_string_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__3 = (const lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__4 = (const lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__4_value;
static lean_once_cell_t lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__5;
static const lean_ctor_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Polynomial_term___x5bX_x5d___closed__0_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__6 = (const lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__6_value;
static const lean_ctor_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__6_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__7 = (const lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__7_value;
static const lean_ctor_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__6_value)}};
static const lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__8 = (const lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__8_value;
static const lean_ctor_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__9 = (const lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__9_value;
static const lean_ctor_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__7_value),((lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__9_value)}};
static const lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__10 = (const lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__10_value;
static const lean_string_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__11 = (const lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__11_value;
static const lean_ctor_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__12 = (const lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______unexpand__Polynomial__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______unexpand__Polynomial__1___closed__0 = (const lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______unexpand__Polynomial__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______unexpand__Polynomial__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______unexpand__Polynomial__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______unexpand__Polynomial__1___closed__1 = (const lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______unexpand__Polynomial__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______unexpand__Polynomial__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______unexpand__Polynomial__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_toFinsuppIso___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_toFinsuppIso___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_toFinsuppIso___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_toFinsuppIso___lam__1___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Polynomial_toFinsuppIso___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Polynomial_toFinsuppIso___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Polynomial_toFinsuppIso___closed__0 = (const lean_object*)&lp_mathlib_Polynomial_toFinsuppIso___closed__0_value;
static const lean_closure_object lp_mathlib_Polynomial_toFinsuppIso___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Polynomial_toFinsuppIso___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Polynomial_toFinsuppIso___closed__1 = (const lean_object*)&lp_mathlib_Polynomial_toFinsuppIso___closed__1_value;
static const lean_ctor_object lp_mathlib_Polynomial_toFinsuppIso___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial_toFinsuppIso___closed__0_value),((lean_object*)&lp_mathlib_Polynomial_toFinsuppIso___closed__1_value)}};
static const lean_object* lp_mathlib_Polynomial_toFinsuppIso___closed__2 = (const lean_object*)&lp_mathlib_Polynomial_toFinsuppIso___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_toFinsuppIso(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_toFinsuppIso___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Polynomial_instDecidableEq___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_instDecidableEq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_instDecidableEq___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Polynomial_instDecidableEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_instDecidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Polynomial_instDecidableEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_instDecidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_toFinsuppIsoLinear___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_toFinsuppIsoLinear___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_toFinsuppIsoLinear(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_toFinsuppIsoLinear___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_support___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_support___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_support(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_support___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Polynomial_Basic_0__Polynomial_support_match__1_splitter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Polynomial_Basic_0__Polynomial_support_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Polynomial_Basic_0__Polynomial_support_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_coeff___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_coeff___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_coeff(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_coeff___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_sum___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_sum___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_sum___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_sum(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_sum___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_repr___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_repr___redArg___lam__0___boxed(lean_object*);
static const lean_string_object lp_mathlib_Polynomial_repr___redArg___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "C "};
static const lean_object* lp_mathlib_Polynomial_repr___redArg___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Polynomial_repr___redArg___lam__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Polynomial_repr___redArg___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial_repr___redArg___lam__1___closed__0_value)}};
static const lean_object* lp_mathlib_Polynomial_repr___redArg___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Polynomial_repr___redArg___lam__1___closed__1_value;
static const lean_string_object lp_mathlib_Polynomial_repr___redArg___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " * X ^ "};
static const lean_object* lp_mathlib_Polynomial_repr___redArg___lam__1___closed__2 = (const lean_object*)&lp_mathlib_Polynomial_repr___redArg___lam__1___closed__2_value;
static const lean_ctor_object lp_mathlib_Polynomial_repr___redArg___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial_repr___redArg___lam__1___closed__2_value)}};
static const lean_object* lp_mathlib_Polynomial_repr___redArg___lam__1___closed__3 = (const lean_object*)&lp_mathlib_Polynomial_repr___redArg___lam__1___closed__3_value;
static const lean_string_object lp_mathlib_Polynomial_repr___redArg___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "X ^ "};
static const lean_object* lp_mathlib_Polynomial_repr___redArg___lam__1___closed__4 = (const lean_object*)&lp_mathlib_Polynomial_repr___redArg___lam__1___closed__4_value;
static const lean_ctor_object lp_mathlib_Polynomial_repr___redArg___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial_repr___redArg___lam__1___closed__4_value)}};
static const lean_object* lp_mathlib_Polynomial_repr___redArg___lam__1___closed__5 = (const lean_object*)&lp_mathlib_Polynomial_repr___redArg___lam__1___closed__5_value;
static const lean_string_object lp_mathlib_Polynomial_repr___redArg___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " * X"};
static const lean_object* lp_mathlib_Polynomial_repr___redArg___lam__1___closed__6 = (const lean_object*)&lp_mathlib_Polynomial_repr___redArg___lam__1___closed__6_value;
static const lean_ctor_object lp_mathlib_Polynomial_repr___redArg___lam__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial_repr___redArg___lam__1___closed__6_value)}};
static const lean_object* lp_mathlib_Polynomial_repr___redArg___lam__1___closed__7 = (const lean_object*)&lp_mathlib_Polynomial_repr___redArg___lam__1___closed__7_value;
static const lean_string_object lp_mathlib_Polynomial_repr___redArg___lam__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "X"};
static const lean_object* lp_mathlib_Polynomial_repr___redArg___lam__1___closed__8 = (const lean_object*)&lp_mathlib_Polynomial_repr___redArg___lam__1___closed__8_value;
static const lean_ctor_object lp_mathlib_Polynomial_repr___redArg___lam__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial_repr___redArg___lam__1___closed__8_value)}};
static const lean_object* lp_mathlib_Polynomial_repr___redArg___lam__1___closed__9 = (const lean_object*)&lp_mathlib_Polynomial_repr___redArg___lam__1___closed__9_value;
static const lean_ctor_object lp_mathlib_Polynomial_repr___redArg___lam__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Polynomial_repr___redArg___lam__1___closed__9_value)}};
static const lean_object* lp_mathlib_Polynomial_repr___redArg___lam__1___closed__10 = (const lean_object*)&lp_mathlib_Polynomial_repr___redArg___lam__1___closed__10_value;
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_repr___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Polynomial_repr___redArg___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Nat_decLe___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Polynomial_repr___redArg___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Polynomial_repr___redArg___lam__2___closed__0_value;
static const lean_string_object lp_mathlib_Polynomial_repr___redArg___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "0"};
static const lean_object* lp_mathlib_Polynomial_repr___redArg___lam__2___closed__1 = (const lean_object*)&lp_mathlib_Polynomial_repr___redArg___lam__2___closed__1_value;
static const lean_ctor_object lp_mathlib_Polynomial_repr___redArg___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial_repr___redArg___lam__2___closed__1_value)}};
static const lean_object* lp_mathlib_Polynomial_repr___redArg___lam__2___closed__2 = (const lean_object*)&lp_mathlib_Polynomial_repr___redArg___lam__2___closed__2_value;
static const lean_string_object lp_mathlib_Polynomial_repr___redArg___lam__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Polynomial_repr___redArg___lam__2___closed__3 = (const lean_object*)&lp_mathlib_Polynomial_repr___redArg___lam__2___closed__3_value;
static const lean_string_object lp_mathlib_Polynomial_repr___redArg___lam__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Polynomial_repr___redArg___lam__2___closed__4 = (const lean_object*)&lp_mathlib_Polynomial_repr___redArg___lam__2___closed__4_value;
static lean_once_cell_t lp_mathlib_Polynomial_repr___redArg___lam__2___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Polynomial_repr___redArg___lam__2___closed__5;
static lean_once_cell_t lp_mathlib_Polynomial_repr___redArg___lam__2___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Polynomial_repr___redArg___lam__2___closed__6;
static const lean_ctor_object lp_mathlib_Polynomial_repr___redArg___lam__2___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial_repr___redArg___lam__2___closed__3_value)}};
static const lean_object* lp_mathlib_Polynomial_repr___redArg___lam__2___closed__7 = (const lean_object*)&lp_mathlib_Polynomial_repr___redArg___lam__2___closed__7_value;
static const lean_ctor_object lp_mathlib_Polynomial_repr___redArg___lam__2___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial_repr___redArg___lam__2___closed__4_value)}};
static const lean_object* lp_mathlib_Polynomial_repr___redArg___lam__2___closed__8 = (const lean_object*)&lp_mathlib_Polynomial_repr___redArg___lam__2___closed__8_value;
static const lean_string_object lp_mathlib_Polynomial_repr___redArg___lam__2___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " +"};
static const lean_object* lp_mathlib_Polynomial_repr___redArg___lam__2___closed__9 = (const lean_object*)&lp_mathlib_Polynomial_repr___redArg___lam__2___closed__9_value;
static const lean_ctor_object lp_mathlib_Polynomial_repr___redArg___lam__2___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial_repr___redArg___lam__2___closed__9_value)}};
static const lean_object* lp_mathlib_Polynomial_repr___redArg___lam__2___closed__10 = (const lean_object*)&lp_mathlib_Polynomial_repr___redArg___lam__2___closed__10_value;
static const lean_ctor_object lp_mathlib_Polynomial_repr___redArg___lam__2___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial_repr___redArg___lam__2___closed__10_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_mathlib_Polynomial_repr___redArg___lam__2___closed__11 = (const lean_object*)&lp_mathlib_Polynomial_repr___redArg___lam__2___closed__11_value;
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_repr___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Polynomial_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Polynomial_repr___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Polynomial_repr___redArg___closed__0 = (const lean_object*)&lp_mathlib_Polynomial_repr___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_Polynomial_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instNatCastNat___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Polynomial_repr___redArg___closed__1 = (const lean_object*)&lp_mathlib_Polynomial_repr___redArg___closed__1_value;
static const lean_closure_object lp_mathlib_Polynomial_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Std_instToFormatFormat___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Polynomial_repr___redArg___closed__2 = (const lean_object*)&lp_mathlib_Polynomial_repr___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_repr___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_repr(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__5(void){
_start:
{
lean_object* v___x_24_; lean_object* v___x_25_; 
v___x_24_ = ((lean_object*)(lp_mathlib_Polynomial_term___x5bX_x5d___closed__0));
v___x_25_ = l_String_toRawSubstring_x27(v___x_24_);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1(lean_object* v_x_42_, lean_object* v_a_43_, lean_object* v_a_44_){
_start:
{
lean_object* v___x_45_; uint8_t v___x_46_; 
v___x_45_ = ((lean_object*)(lp_mathlib_Polynomial_term___x5bX_x5d___closed__2));
lean_inc(v_x_42_);
v___x_46_ = l_Lean_Syntax_isOfKind(v_x_42_, v___x_45_);
if (v___x_46_ == 0)
{
lean_object* v___x_47_; lean_object* v___x_48_; 
lean_dec(v_x_42_);
v___x_47_ = lean_box(1);
v___x_48_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_48_, 0, v___x_47_);
lean_ctor_set(v___x_48_, 1, v_a_44_);
return v___x_48_;
}
else
{
lean_object* v_quotContext_49_; lean_object* v_currMacroScope_50_; lean_object* v_ref_51_; lean_object* v___x_52_; lean_object* v___x_53_; uint8_t v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; 
v_quotContext_49_ = lean_ctor_get(v_a_43_, 1);
v_currMacroScope_50_ = lean_ctor_get(v_a_43_, 2);
v_ref_51_ = lean_ctor_get(v_a_43_, 5);
v___x_52_ = lean_unsigned_to_nat(0u);
v___x_53_ = l_Lean_Syntax_getArg(v_x_42_, v___x_52_);
lean_dec(v_x_42_);
v___x_54_ = 0;
v___x_55_ = l_Lean_SourceInfo_fromRef(v_ref_51_, v___x_54_);
v___x_56_ = ((lean_object*)(lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__4));
v___x_57_ = lean_obj_once(&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__5, &lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__5_once, _init_lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__5);
v___x_58_ = ((lean_object*)(lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__6));
lean_inc(v_currMacroScope_50_);
lean_inc(v_quotContext_49_);
v___x_59_ = l_Lean_addMacroScope(v_quotContext_49_, v___x_58_, v_currMacroScope_50_);
v___x_60_ = ((lean_object*)(lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__10));
lean_inc_n(v___x_55_, 2);
v___x_61_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_61_, 0, v___x_55_);
lean_ctor_set(v___x_61_, 1, v___x_57_);
lean_ctor_set(v___x_61_, 2, v___x_59_);
lean_ctor_set(v___x_61_, 3, v___x_60_);
v___x_62_ = ((lean_object*)(lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__12));
v___x_63_ = l_Lean_Syntax_node1(v___x_55_, v___x_62_, v___x_53_);
v___x_64_ = l_Lean_Syntax_node2(v___x_55_, v___x_56_, v___x_61_, v___x_63_);
v___x_65_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_65_, 0, v___x_64_);
lean_ctor_set(v___x_65_, 1, v_a_44_);
return v___x_65_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___boxed(lean_object* v_x_66_, lean_object* v_a_67_, lean_object* v_a_68_){
_start:
{
lean_object* v_res_69_; 
v_res_69_ = lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1(v_x_66_, v_a_67_, v_a_68_);
lean_dec_ref(v_a_67_);
return v_res_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______unexpand__Polynomial__1(lean_object* v_x_73_, lean_object* v_a_74_, lean_object* v_a_75_){
_start:
{
lean_object* v___x_76_; uint8_t v___x_77_; 
v___x_76_ = ((lean_object*)(lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______macroRules__Polynomial__term___x5bX_x5d__1___closed__4));
lean_inc(v_x_73_);
v___x_77_ = l_Lean_Syntax_isOfKind(v_x_73_, v___x_76_);
if (v___x_77_ == 0)
{
lean_object* v___x_78_; lean_object* v___x_79_; 
lean_dec(v_x_73_);
v___x_78_ = lean_box(0);
v___x_79_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_79_, 0, v___x_78_);
lean_ctor_set(v___x_79_, 1, v_a_75_);
return v___x_79_;
}
else
{
lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; uint8_t v___x_83_; 
v___x_80_ = lean_unsigned_to_nat(0u);
v___x_81_ = l_Lean_Syntax_getArg(v_x_73_, v___x_80_);
v___x_82_ = ((lean_object*)(lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______unexpand__Polynomial__1___closed__1));
lean_inc(v___x_81_);
v___x_83_ = l_Lean_Syntax_isOfKind(v___x_81_, v___x_82_);
if (v___x_83_ == 0)
{
lean_object* v___x_84_; lean_object* v___x_85_; 
lean_dec(v___x_81_);
lean_dec(v_x_73_);
v___x_84_ = lean_box(0);
v___x_85_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_85_, 0, v___x_84_);
lean_ctor_set(v___x_85_, 1, v_a_75_);
return v___x_85_;
}
else
{
lean_object* v___x_86_; lean_object* v___x_87_; uint8_t v___x_88_; 
v___x_86_ = lean_unsigned_to_nat(1u);
v___x_87_ = l_Lean_Syntax_getArg(v_x_73_, v___x_86_);
lean_dec(v_x_73_);
lean_inc(v___x_87_);
v___x_88_ = l_Lean_Syntax_matchesNull(v___x_87_, v___x_86_);
if (v___x_88_ == 0)
{
lean_object* v___x_89_; lean_object* v___x_90_; 
lean_dec(v___x_87_);
lean_dec(v___x_81_);
v___x_89_ = lean_box(0);
v___x_90_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_90_, 0, v___x_89_);
lean_ctor_set(v___x_90_, 1, v_a_75_);
return v___x_90_;
}
else
{
lean_object* v___x_91_; lean_object* v_ref_92_; uint8_t v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; 
v___x_91_ = l_Lean_Syntax_getArg(v___x_87_, v___x_80_);
lean_dec(v___x_87_);
v_ref_92_ = l_Lean_replaceRef(v___x_81_, v_a_74_);
lean_dec(v___x_81_);
v___x_93_ = 0;
v___x_94_ = l_Lean_SourceInfo_fromRef(v_ref_92_, v___x_93_);
lean_dec(v_ref_92_);
v___x_95_ = ((lean_object*)(lp_mathlib_Polynomial_term___x5bX_x5d___closed__2));
v___x_96_ = ((lean_object*)(lp_mathlib_Polynomial_term___x5bX_x5d___closed__3));
lean_inc(v___x_94_);
v___x_97_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_97_, 0, v___x_94_);
lean_ctor_set(v___x_97_, 1, v___x_96_);
v___x_98_ = l_Lean_Syntax_node2(v___x_94_, v___x_95_, v___x_91_, v___x_97_);
v___x_99_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_99_, 0, v___x_98_);
lean_ctor_set(v___x_99_, 1, v_a_75_);
return v___x_99_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______unexpand__Polynomial__1___boxed(lean_object* v_x_100_, lean_object* v_a_101_, lean_object* v_a_102_){
_start:
{
lean_object* v_res_103_; 
v_res_103_ = lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Basic______unexpand__Polynomial__1(v_x_100_, v_a_101_, v_a_102_);
lean_dec(v_a_101_);
return v_res_103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_toFinsuppIso___lam__0(lean_object* v_self_104_){
_start:
{
lean_inc_ref(v_self_104_);
return v_self_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_toFinsuppIso___lam__0___boxed(lean_object* v_self_105_){
_start:
{
lean_object* v_res_106_; 
v_res_106_ = lp_mathlib_Polynomial_toFinsuppIso___lam__0(v_self_105_);
lean_dec_ref(v_self_105_);
return v_res_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_toFinsuppIso___lam__1(lean_object* v_toFinsupp_107_){
_start:
{
lean_inc_ref(v_toFinsupp_107_);
return v_toFinsupp_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_toFinsuppIso___lam__1___boxed(lean_object* v_toFinsupp_108_){
_start:
{
lean_object* v_res_109_; 
v_res_109_ = lp_mathlib_Polynomial_toFinsuppIso___lam__1(v_toFinsupp_108_);
lean_dec_ref(v_toFinsupp_108_);
return v_res_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_toFinsuppIso(lean_object* v_R_115_, lean_object* v_inst_116_){
_start:
{
lean_object* v___x_117_; 
v___x_117_ = ((lean_object*)(lp_mathlib_Polynomial_toFinsuppIso___closed__2));
return v___x_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_toFinsuppIso___boxed(lean_object* v_R_118_, lean_object* v_inst_119_){
_start:
{
lean_object* v_res_120_; 
v_res_120_ = lp_mathlib_Polynomial_toFinsuppIso(v_R_118_, v_inst_119_);
lean_dec_ref(v_inst_119_);
return v_res_120_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Polynomial_instDecidableEq___redArg___lam__0(lean_object* v_inst_121_, lean_object* v_inst_122_, lean_object* v_a_123_, lean_object* v_b_124_){
_start:
{
lean_object* v___x_125_; uint8_t v___x_126_; 
v___x_125_ = lean_alloc_closure((void*)(l_instDecidableEqNat___boxed), 2, 0);
v___x_126_ = lp_mathlib_AddMonoidAlgebra_instDecidableEq___redArg(v_inst_121_, v_inst_122_, v___x_125_, v_a_123_, v_b_124_);
return v___x_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_instDecidableEq___redArg___lam__0___boxed(lean_object* v_inst_127_, lean_object* v_inst_128_, lean_object* v_a_129_, lean_object* v_b_130_){
_start:
{
uint8_t v_res_131_; lean_object* v_r_132_; 
v_res_131_ = lp_mathlib_Polynomial_instDecidableEq___redArg___lam__0(v_inst_127_, v_inst_128_, v_a_129_, v_b_130_);
lean_dec_ref(v_inst_127_);
v_r_132_ = lean_box(v_res_131_);
return v_r_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_instDecidableEq___redArg___lam__1(lean_object* v___x_133_, lean_object* v___y_134_){
_start:
{
lean_object* v_toFun_135_; lean_object* v___x_136_; 
v_toFun_135_ = lean_ctor_get(v___x_133_, 0);
lean_inc(v_toFun_135_);
lean_dec_ref(v___x_133_);
v___x_136_ = lean_apply_1(v_toFun_135_, v___y_134_);
return v___x_136_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Polynomial_instDecidableEq___redArg(lean_object* v_inst_137_, lean_object* v_inst_138_, lean_object* v_a_139_, lean_object* v_b_140_){
_start:
{
lean_object* v___f_141_; lean_object* v___x_142_; lean_object* v___f_143_; uint8_t v___x_144_; 
lean_inc_ref(v_inst_137_);
v___f_141_ = lean_alloc_closure((void*)(lp_mathlib_Polynomial_instDecidableEq___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_141_, 0, v_inst_137_);
lean_closure_set(v___f_141_, 1, v_inst_138_);
v___x_142_ = lp_mathlib_Polynomial_toFinsuppIso(lean_box(0), v_inst_137_);
lean_dec_ref(v_inst_137_);
v___f_143_ = lean_alloc_closure((void*)(lp_mathlib_Polynomial_instDecidableEq___redArg___lam__1), 2, 1);
lean_closure_set(v___f_143_, 0, v___x_142_);
v___x_144_ = lp_mathlib_Function_Injective_decidableEq___redArg(v___f_143_, v___f_141_, v_a_139_, v_b_140_);
return v___x_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_instDecidableEq___redArg___boxed(lean_object* v_inst_145_, lean_object* v_inst_146_, lean_object* v_a_147_, lean_object* v_b_148_){
_start:
{
uint8_t v_res_149_; lean_object* v_r_150_; 
v_res_149_ = lp_mathlib_Polynomial_instDecidableEq___redArg(v_inst_145_, v_inst_146_, v_a_147_, v_b_148_);
v_r_150_ = lean_box(v_res_149_);
return v_r_150_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Polynomial_instDecidableEq(lean_object* v_R_151_, lean_object* v_inst_152_, lean_object* v_inst_153_, lean_object* v_a_154_, lean_object* v_b_155_){
_start:
{
uint8_t v___x_156_; 
v___x_156_ = lp_mathlib_Polynomial_instDecidableEq___redArg(v_inst_152_, v_inst_153_, v_a_154_, v_b_155_);
return v___x_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_instDecidableEq___boxed(lean_object* v_R_157_, lean_object* v_inst_158_, lean_object* v_inst_159_, lean_object* v_a_160_, lean_object* v_b_161_){
_start:
{
uint8_t v_res_162_; lean_object* v_r_163_; 
v_res_162_ = lp_mathlib_Polynomial_instDecidableEq(v_R_157_, v_inst_158_, v_inst_159_, v_a_160_, v_b_161_);
v_r_163_ = lean_box(v_res_162_);
return v_r_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_toFinsuppIsoLinear___redArg(lean_object* v_inst_164_){
_start:
{
lean_object* v___x_165_; lean_object* v_toFun_166_; lean_object* v_invFun_167_; lean_object* v___x_169_; uint8_t v_isShared_170_; uint8_t v_isSharedCheck_174_; 
v___x_165_ = lp_mathlib_Polynomial_toFinsuppIso(lean_box(0), v_inst_164_);
v_toFun_166_ = lean_ctor_get(v___x_165_, 0);
v_invFun_167_ = lean_ctor_get(v___x_165_, 1);
v_isSharedCheck_174_ = !lean_is_exclusive(v___x_165_);
if (v_isSharedCheck_174_ == 0)
{
v___x_169_ = v___x_165_;
v_isShared_170_ = v_isSharedCheck_174_;
goto v_resetjp_168_;
}
else
{
lean_inc(v_invFun_167_);
lean_inc(v_toFun_166_);
lean_dec(v___x_165_);
v___x_169_ = lean_box(0);
v_isShared_170_ = v_isSharedCheck_174_;
goto v_resetjp_168_;
}
v_resetjp_168_:
{
lean_object* v___x_172_; 
if (v_isShared_170_ == 0)
{
v___x_172_ = v___x_169_;
goto v_reusejp_171_;
}
else
{
lean_object* v_reuseFailAlloc_173_; 
v_reuseFailAlloc_173_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_173_, 0, v_toFun_166_);
lean_ctor_set(v_reuseFailAlloc_173_, 1, v_invFun_167_);
v___x_172_ = v_reuseFailAlloc_173_;
goto v_reusejp_171_;
}
v_reusejp_171_:
{
return v___x_172_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_toFinsuppIsoLinear___redArg___boxed(lean_object* v_inst_175_){
_start:
{
lean_object* v_res_176_; 
v_res_176_ = lp_mathlib_Polynomial_toFinsuppIsoLinear___redArg(v_inst_175_);
lean_dec_ref(v_inst_175_);
return v_res_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_toFinsuppIsoLinear(lean_object* v_R_177_, lean_object* v_inst_178_){
_start:
{
lean_object* v___x_179_; 
v___x_179_ = lp_mathlib_Polynomial_toFinsuppIsoLinear___redArg(v_inst_178_);
return v___x_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_toFinsuppIsoLinear___boxed(lean_object* v_R_180_, lean_object* v_inst_181_){
_start:
{
lean_object* v_res_182_; 
v_res_182_ = lp_mathlib_Polynomial_toFinsuppIsoLinear(v_R_180_, v_inst_181_);
lean_dec_ref(v_inst_181_);
return v_res_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_support___redArg(lean_object* v_x_183_){
_start:
{
lean_object* v_support_184_; 
v_support_184_ = lean_ctor_get(v_x_183_, 0);
lean_inc(v_support_184_);
return v_support_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_support___redArg___boxed(lean_object* v_x_185_){
_start:
{
lean_object* v_res_186_; 
v_res_186_ = lp_mathlib_Polynomial_support___redArg(v_x_185_);
lean_dec_ref(v_x_185_);
return v_res_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_support(lean_object* v_R_187_, lean_object* v_inst_188_, lean_object* v_x_189_){
_start:
{
lean_object* v_support_190_; 
v_support_190_ = lean_ctor_get(v_x_189_, 0);
lean_inc(v_support_190_);
return v_support_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_support___boxed(lean_object* v_R_191_, lean_object* v_inst_192_, lean_object* v_x_193_){
_start:
{
lean_object* v_res_194_; 
v_res_194_ = lp_mathlib_Polynomial_support(v_R_191_, v_inst_192_, v_x_193_);
lean_dec_ref(v_x_193_);
lean_dec_ref(v_inst_192_);
return v_res_194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Polynomial_Basic_0__Polynomial_support_match__1_splitter___redArg(lean_object* v_x_195_, lean_object* v_h__1_196_){
_start:
{
lean_object* v___x_197_; 
v___x_197_ = lean_apply_1(v_h__1_196_, v_x_195_);
return v___x_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Polynomial_Basic_0__Polynomial_support_match__1_splitter(lean_object* v_R_198_, lean_object* v_inst_199_, lean_object* v_motive_200_, lean_object* v_x_201_, lean_object* v_h__1_202_){
_start:
{
lean_object* v___x_203_; 
v___x_203_ = lean_apply_1(v_h__1_202_, v_x_201_);
return v___x_203_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Polynomial_Basic_0__Polynomial_support_match__1_splitter___boxed(lean_object* v_R_204_, lean_object* v_inst_205_, lean_object* v_motive_206_, lean_object* v_x_207_, lean_object* v_h__1_208_){
_start:
{
lean_object* v_res_209_; 
v_res_209_ = lp_mathlib___private_Mathlib_Algebra_Polynomial_Basic_0__Polynomial_support_match__1_splitter(v_R_204_, v_inst_205_, v_motive_206_, v_x_207_, v_h__1_208_);
lean_dec_ref(v_inst_205_);
return v_res_209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_coeff___redArg(lean_object* v_x_210_){
_start:
{
lean_inc_ref(v_x_210_);
return v_x_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_coeff___redArg___boxed(lean_object* v_x_211_){
_start:
{
lean_object* v_res_212_; 
v_res_212_ = lp_mathlib_Polynomial_coeff___redArg(v_x_211_);
lean_dec_ref(v_x_211_);
return v_res_212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_coeff(lean_object* v_R_213_, lean_object* v_inst_214_, lean_object* v_x_215_){
_start:
{
lean_inc_ref(v_x_215_);
return v_x_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_coeff___boxed(lean_object* v_R_216_, lean_object* v_inst_217_, lean_object* v_x_218_){
_start:
{
lean_object* v_res_219_; 
v_res_219_ = lp_mathlib_Polynomial_coeff(v_R_216_, v_inst_217_, v_x_218_);
lean_dec_ref(v_x_218_);
lean_dec_ref(v_inst_217_);
return v_res_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_sum___redArg___lam__0(lean_object* v_toFun_220_, lean_object* v_f_221_, lean_object* v_n_222_){
_start:
{
lean_object* v___x_223_; lean_object* v___x_224_; 
lean_inc(v_n_222_);
v___x_223_ = lean_apply_1(v_toFun_220_, v_n_222_);
v___x_224_ = lean_apply_2(v_f_221_, v_n_222_, v___x_223_);
return v___x_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_sum___redArg(lean_object* v_inst_225_, lean_object* v_p_226_, lean_object* v_f_227_){
_start:
{
lean_object* v_support_228_; lean_object* v_toFun_229_; lean_object* v___f_230_; lean_object* v___x_231_; 
v_support_228_ = lean_ctor_get(v_p_226_, 0);
lean_inc(v_support_228_);
v_toFun_229_ = lean_ctor_get(v_p_226_, 1);
lean_inc(v_toFun_229_);
lean_dec_ref(v_p_226_);
v___f_230_ = lean_alloc_closure((void*)(lp_mathlib_Polynomial_sum___redArg___lam__0), 3, 2);
lean_closure_set(v___f_230_, 0, v_toFun_229_);
lean_closure_set(v___f_230_, 1, v_f_227_);
v___x_231_ = lp_mathlib_Finset_sum___redArg(v_inst_225_, v_support_228_, v___f_230_);
return v___x_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_sum___redArg___boxed(lean_object* v_inst_232_, lean_object* v_p_233_, lean_object* v_f_234_){
_start:
{
lean_object* v_res_235_; 
v_res_235_ = lp_mathlib_Polynomial_sum___redArg(v_inst_232_, v_p_233_, v_f_234_);
lean_dec_ref(v_inst_232_);
return v_res_235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_sum(lean_object* v_R_236_, lean_object* v_inst_237_, lean_object* v_S_238_, lean_object* v_inst_239_, lean_object* v_p_240_, lean_object* v_f_241_){
_start:
{
lean_object* v___x_242_; 
v___x_242_ = lp_mathlib_Polynomial_sum___redArg(v_inst_239_, v_p_240_, v_f_241_);
return v___x_242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_sum___boxed(lean_object* v_R_243_, lean_object* v_inst_244_, lean_object* v_S_245_, lean_object* v_inst_246_, lean_object* v_p_247_, lean_object* v_f_248_){
_start:
{
lean_object* v_res_249_; 
v_res_249_ = lp_mathlib_Polynomial_sum(v_R_243_, v_inst_244_, v_S_245_, v_inst_246_, v_p_247_, v_f_248_);
lean_dec_ref(v_inst_246_);
lean_dec_ref(v_inst_244_);
return v_res_249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_repr___redArg___lam__0(lean_object* v_self_250_){
_start:
{
lean_object* v_snd_251_; 
v_snd_251_ = lean_ctor_get(v_self_250_, 1);
lean_inc(v_snd_251_);
return v_snd_251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_repr___redArg___lam__0___boxed(lean_object* v_self_252_){
_start:
{
lean_object* v_res_253_; 
v_res_253_ = lp_mathlib_Polynomial_repr___redArg___lam__0(v_self_252_);
lean_dec_ref(v_self_252_);
return v_res_253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_repr___redArg___lam__1(lean_object* v_toFun_272_, lean_object* v_inst_273_, lean_object* v_toOne_274_, lean_object* v___f_275_, lean_object* v_inst_276_, lean_object* v_x_277_){
_start:
{
lean_object* v___x_278_; uint8_t v___x_279_; 
v___x_278_ = lean_unsigned_to_nat(0u);
v___x_279_ = lean_nat_dec_eq(v_x_277_, v___x_278_);
if (v___x_279_ == 0)
{
lean_object* v___x_280_; uint8_t v___x_281_; 
v___x_280_ = lean_unsigned_to_nat(1u);
v___x_281_ = lean_nat_dec_eq(v_x_277_, v___x_280_);
if (v___x_281_ == 0)
{
lean_object* v___x_282_; lean_object* v___x_283_; uint8_t v___x_284_; 
lean_inc(v_x_277_);
v___x_282_ = lean_apply_1(v_toFun_272_, v_x_277_);
lean_inc(v___x_282_);
v___x_283_ = lean_apply_2(v_inst_273_, v___x_282_, v_toOne_274_);
v___x_284_ = lean_unbox(v___x_283_);
if (v___x_284_ == 0)
{
lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; 
v___x_285_ = lean_unsigned_to_nat(70u);
v___x_286_ = lp_mathlib_WithTop_natCast___redArg___lam__0(v___f_275_, v___x_285_);
v___x_287_ = ((lean_object*)(lp_mathlib_Polynomial_repr___redArg___lam__1___closed__1));
v___x_288_ = lean_unsigned_to_nat(1024u);
v___x_289_ = lean_apply_2(v_inst_276_, v___x_282_, v___x_288_);
v___x_290_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_290_, 0, v___x_287_);
lean_ctor_set(v___x_290_, 1, v___x_289_);
v___x_291_ = ((lean_object*)(lp_mathlib_Polynomial_repr___redArg___lam__1___closed__3));
v___x_292_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_292_, 0, v___x_290_);
lean_ctor_set(v___x_292_, 1, v___x_291_);
v___x_293_ = l_Nat_reprFast(v_x_277_);
v___x_294_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_294_, 0, v___x_293_);
v___x_295_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_295_, 0, v___x_292_);
lean_ctor_set(v___x_295_, 1, v___x_294_);
v___x_296_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_296_, 0, v___x_286_);
lean_ctor_set(v___x_296_, 1, v___x_295_);
return v___x_296_;
}
else
{
lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; 
lean_dec(v___x_282_);
lean_dec_ref(v_inst_276_);
v___x_297_ = lean_unsigned_to_nat(80u);
v___x_298_ = lp_mathlib_WithTop_natCast___redArg___lam__0(v___f_275_, v___x_297_);
v___x_299_ = ((lean_object*)(lp_mathlib_Polynomial_repr___redArg___lam__1___closed__5));
v___x_300_ = l_Nat_reprFast(v_x_277_);
v___x_301_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_301_, 0, v___x_300_);
v___x_302_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_302_, 0, v___x_299_);
lean_ctor_set(v___x_302_, 1, v___x_301_);
v___x_303_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_303_, 0, v___x_298_);
lean_ctor_set(v___x_303_, 1, v___x_302_);
return v___x_303_;
}
}
else
{
lean_object* v___x_304_; lean_object* v___x_305_; uint8_t v___x_306_; 
lean_dec(v_x_277_);
v___x_304_ = lean_apply_1(v_toFun_272_, v___x_280_);
lean_inc(v___x_304_);
v___x_305_ = lean_apply_2(v_inst_273_, v___x_304_, v_toOne_274_);
v___x_306_ = lean_unbox(v___x_305_);
if (v___x_306_ == 0)
{
lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; 
v___x_307_ = lean_unsigned_to_nat(70u);
v___x_308_ = lp_mathlib_WithTop_natCast___redArg___lam__0(v___f_275_, v___x_307_);
v___x_309_ = ((lean_object*)(lp_mathlib_Polynomial_repr___redArg___lam__1___closed__1));
v___x_310_ = lean_unsigned_to_nat(1024u);
v___x_311_ = lean_apply_2(v_inst_276_, v___x_304_, v___x_310_);
v___x_312_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_312_, 0, v___x_309_);
lean_ctor_set(v___x_312_, 1, v___x_311_);
v___x_313_ = ((lean_object*)(lp_mathlib_Polynomial_repr___redArg___lam__1___closed__7));
v___x_314_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_314_, 0, v___x_312_);
lean_ctor_set(v___x_314_, 1, v___x_313_);
v___x_315_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_315_, 0, v___x_308_);
lean_ctor_set(v___x_315_, 1, v___x_314_);
return v___x_315_;
}
else
{
lean_object* v___x_316_; 
lean_dec(v___x_304_);
lean_dec_ref(v_inst_276_);
lean_dec_ref(v___f_275_);
v___x_316_ = ((lean_object*)(lp_mathlib_Polynomial_repr___redArg___lam__1___closed__10));
return v___x_316_;
}
}
}
else
{
lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_323_; 
lean_dec(v_x_277_);
lean_dec(v_toOne_274_);
lean_dec_ref(v_inst_273_);
v___x_317_ = lean_unsigned_to_nat(1024u);
v___x_318_ = lp_mathlib_WithTop_natCast___redArg___lam__0(v___f_275_, v___x_317_);
v___x_319_ = ((lean_object*)(lp_mathlib_Polynomial_repr___redArg___lam__1___closed__1));
v___x_320_ = lean_apply_1(v_toFun_272_, v___x_278_);
v___x_321_ = lean_apply_2(v_inst_276_, v___x_320_, v___x_317_);
v___x_322_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_322_, 0, v___x_319_);
lean_ctor_set(v___x_322_, 1, v___x_321_);
v___x_323_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_323_, 0, v___x_318_);
lean_ctor_set(v___x_323_, 1, v___x_322_);
return v___x_323_;
}
}
}
static lean_object* _init_lp_mathlib_Polynomial_repr___redArg___lam__2___closed__5(void){
_start:
{
lean_object* v___x_330_; lean_object* v___x_331_; 
v___x_330_ = ((lean_object*)(lp_mathlib_Polynomial_repr___redArg___lam__2___closed__3));
v___x_331_ = lean_string_length(v___x_330_);
return v___x_331_;
}
}
static lean_object* _init_lp_mathlib_Polynomial_repr___redArg___lam__2___closed__6(void){
_start:
{
lean_object* v___x_332_; lean_object* v___x_333_; 
v___x_332_ = lean_obj_once(&lp_mathlib_Polynomial_repr___redArg___lam__2___closed__5, &lp_mathlib_Polynomial_repr___redArg___lam__2___closed__5_once, _init_lp_mathlib_Polynomial_repr___redArg___lam__2___closed__5);
v___x_333_ = lean_nat_to_int(v___x_332_);
return v___x_333_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_repr___redArg___lam__2(lean_object* v_inst_344_, lean_object* v_toOne_345_, lean_object* v___f_346_, lean_object* v_inst_347_, lean_object* v___f_348_, lean_object* v___f_349_, lean_object* v_p_350_, lean_object* v_prec_351_){
_start:
{
lean_object* v_support_352_; lean_object* v_toFun_353_; lean_object* v___x_355_; uint8_t v_isShared_356_; uint8_t v_isSharedCheck_417_; 
v_support_352_ = lean_ctor_get(v_p_350_, 0);
v_toFun_353_ = lean_ctor_get(v_p_350_, 1);
v_isSharedCheck_417_ = !lean_is_exclusive(v_p_350_);
if (v_isSharedCheck_417_ == 0)
{
v___x_355_ = v_p_350_;
v_isShared_356_ = v_isSharedCheck_417_;
goto v_resetjp_354_;
}
else
{
lean_inc(v_toFun_353_);
lean_inc(v_support_352_);
lean_dec(v_p_350_);
v___x_355_ = lean_box(0);
v_isShared_356_ = v_isSharedCheck_417_;
goto v_resetjp_354_;
}
v_resetjp_354_:
{
lean_object* v___f_357_; lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; lean_object* v_termPrecAndReprs_361_; 
lean_inc_ref(v___f_346_);
v___f_357_ = lean_alloc_closure((void*)(lp_mathlib_Polynomial_repr___redArg___lam__1), 6, 5);
lean_closure_set(v___f_357_, 0, v_toFun_353_);
lean_closure_set(v___f_357_, 1, v_inst_344_);
lean_closure_set(v___f_357_, 2, v_toOne_345_);
lean_closure_set(v___f_357_, 3, v___f_346_);
lean_closure_set(v___f_357_, 4, v_inst_347_);
v___x_358_ = ((lean_object*)(lp_mathlib_Polynomial_repr___redArg___lam__2___closed__0));
v___x_359_ = lp_mathlib_Multiset_sort___redArg(v_support_352_, v___x_358_);
v___x_360_ = lean_box(0);
v_termPrecAndReprs_361_ = l_List_mapTR_loop___redArg(v___f_357_, v___x_359_, v___x_360_);
if (lean_obj_tag(v_termPrecAndReprs_361_) == 0)
{
lean_object* v___x_362_; 
lean_del_object(v___x_355_);
lean_dec(v_prec_351_);
lean_dec_ref(v___f_349_);
lean_dec_ref(v___f_348_);
lean_dec_ref(v___f_346_);
v___x_362_ = ((lean_object*)(lp_mathlib_Polynomial_repr___redArg___lam__2___closed__2));
return v___x_362_;
}
else
{
lean_object* v_head_363_; lean_object* v_tail_364_; 
v_head_363_ = lean_ctor_get(v_termPrecAndReprs_361_, 0);
lean_inc(v_head_363_);
v_tail_364_ = lean_ctor_get(v_termPrecAndReprs_361_, 1);
lean_inc(v_tail_364_);
if (lean_obj_tag(v_tail_364_) == 0)
{
lean_object* v___x_366_; uint8_t v_isShared_367_; uint8_t v_isSharedCheck_390_; 
lean_dec_ref(v___f_349_);
lean_dec_ref(v___f_348_);
v_isSharedCheck_390_ = !lean_is_exclusive(v_termPrecAndReprs_361_);
if (v_isSharedCheck_390_ == 0)
{
lean_object* v_unused_391_; lean_object* v_unused_392_; 
v_unused_391_ = lean_ctor_get(v_termPrecAndReprs_361_, 1);
lean_dec(v_unused_391_);
v_unused_392_ = lean_ctor_get(v_termPrecAndReprs_361_, 0);
lean_dec(v_unused_392_);
v___x_366_ = v_termPrecAndReprs_361_;
v_isShared_367_ = v_isSharedCheck_390_;
goto v_resetjp_365_;
}
else
{
lean_dec(v_termPrecAndReprs_361_);
v___x_366_ = lean_box(0);
v_isShared_367_ = v_isSharedCheck_390_;
goto v_resetjp_365_;
}
v_resetjp_365_:
{
lean_object* v_fst_368_; lean_object* v_snd_369_; lean_object* v___x_371_; uint8_t v_isShared_372_; uint8_t v_isSharedCheck_389_; 
v_fst_368_ = lean_ctor_get(v_head_363_, 0);
v_snd_369_ = lean_ctor_get(v_head_363_, 1);
v_isSharedCheck_389_ = !lean_is_exclusive(v_head_363_);
if (v_isSharedCheck_389_ == 0)
{
v___x_371_ = v_head_363_;
v_isShared_372_ = v_isSharedCheck_389_;
goto v_resetjp_370_;
}
else
{
lean_inc(v_snd_369_);
lean_inc(v_fst_368_);
lean_dec(v_head_363_);
v___x_371_ = lean_box(0);
v_isShared_372_ = v_isSharedCheck_389_;
goto v_resetjp_370_;
}
v_resetjp_370_:
{
lean_object* v___x_373_; uint8_t v___x_374_; 
v___x_373_ = lp_mathlib_WithTop_natCast___redArg___lam__0(v___f_346_, v_prec_351_);
v___x_374_ = lp_mathlib_WithTop_decidableLE___redArg(v___x_358_, v_fst_368_, v___x_373_);
if (v___x_374_ == 0)
{
lean_del_object(v___x_371_);
lean_del_object(v___x_366_);
lean_del_object(v___x_355_);
return v_snd_369_;
}
else
{
lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_378_; 
v___x_375_ = lean_obj_once(&lp_mathlib_Polynomial_repr___redArg___lam__2___closed__6, &lp_mathlib_Polynomial_repr___redArg___lam__2___closed__6_once, _init_lp_mathlib_Polynomial_repr___redArg___lam__2___closed__6);
v___x_376_ = ((lean_object*)(lp_mathlib_Polynomial_repr___redArg___lam__2___closed__7));
if (v_isShared_372_ == 0)
{
lean_ctor_set_tag(v___x_371_, 5);
lean_ctor_set(v___x_371_, 0, v___x_376_);
v___x_378_ = v___x_371_;
goto v_reusejp_377_;
}
else
{
lean_object* v_reuseFailAlloc_388_; 
v_reuseFailAlloc_388_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_388_, 0, v___x_376_);
lean_ctor_set(v_reuseFailAlloc_388_, 1, v_snd_369_);
v___x_378_ = v_reuseFailAlloc_388_;
goto v_reusejp_377_;
}
v_reusejp_377_:
{
lean_object* v___x_379_; lean_object* v___x_381_; 
v___x_379_ = ((lean_object*)(lp_mathlib_Polynomial_repr___redArg___lam__2___closed__8));
if (v_isShared_367_ == 0)
{
lean_ctor_set_tag(v___x_366_, 5);
lean_ctor_set(v___x_366_, 1, v___x_379_);
lean_ctor_set(v___x_366_, 0, v___x_378_);
v___x_381_ = v___x_366_;
goto v_reusejp_380_;
}
else
{
lean_object* v_reuseFailAlloc_387_; 
v_reuseFailAlloc_387_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_387_, 0, v___x_378_);
lean_ctor_set(v_reuseFailAlloc_387_, 1, v___x_379_);
v___x_381_ = v_reuseFailAlloc_387_;
goto v_reusejp_380_;
}
v_reusejp_380_:
{
lean_object* v___x_383_; 
if (v_isShared_356_ == 0)
{
lean_ctor_set_tag(v___x_355_, 4);
lean_ctor_set(v___x_355_, 1, v___x_381_);
lean_ctor_set(v___x_355_, 0, v___x_375_);
v___x_383_ = v___x_355_;
goto v_reusejp_382_;
}
else
{
lean_object* v_reuseFailAlloc_386_; 
v_reuseFailAlloc_386_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v_reuseFailAlloc_386_, 0, v___x_375_);
lean_ctor_set(v_reuseFailAlloc_386_, 1, v___x_381_);
v___x_383_ = v_reuseFailAlloc_386_;
goto v_reusejp_382_;
}
v_reusejp_382_:
{
uint8_t v___x_384_; lean_object* v___x_385_; 
v___x_384_ = 0;
v___x_385_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_385_, 0, v___x_383_);
lean_ctor_set_uint8(v___x_385_, sizeof(void*)*1, v___x_384_);
return v___x_385_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_394_; uint8_t v_isShared_395_; uint8_t v_isSharedCheck_414_; 
lean_dec(v_tail_364_);
lean_dec_ref(v___f_346_);
v_isSharedCheck_414_ = !lean_is_exclusive(v_head_363_);
if (v_isSharedCheck_414_ == 0)
{
lean_object* v_unused_415_; lean_object* v_unused_416_; 
v_unused_415_ = lean_ctor_get(v_head_363_, 1);
lean_dec(v_unused_415_);
v_unused_416_ = lean_ctor_get(v_head_363_, 0);
lean_dec(v_unused_416_);
v___x_394_ = v_head_363_;
v_isShared_395_ = v_isSharedCheck_414_;
goto v_resetjp_393_;
}
else
{
lean_dec(v_head_363_);
v___x_394_ = lean_box(0);
v_isShared_395_ = v_isSharedCheck_414_;
goto v_resetjp_393_;
}
v_resetjp_393_:
{
lean_object* v___x_396_; uint8_t v___x_397_; lean_object* v___x_398_; lean_object* v___x_399_; lean_object* v___x_400_; lean_object* v___x_401_; 
v___x_396_ = lean_unsigned_to_nat(65u);
v___x_397_ = lean_nat_dec_le(v___x_396_, v_prec_351_);
lean_dec(v_prec_351_);
v___x_398_ = l_List_mapTR_loop___redArg(v___f_348_, v_termPrecAndReprs_361_, v___x_360_);
v___x_399_ = ((lean_object*)(lp_mathlib_Polynomial_repr___redArg___lam__2___closed__11));
v___x_400_ = l_Std_Format_joinSep___redArg(v___f_349_, v___x_398_, v___x_399_);
v___x_401_ = l_Std_Format_fill(v___x_400_);
if (v___x_397_ == 0)
{
lean_del_object(v___x_394_);
lean_del_object(v___x_355_);
return v___x_401_;
}
else
{
lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_405_; 
v___x_402_ = lean_obj_once(&lp_mathlib_Polynomial_repr___redArg___lam__2___closed__6, &lp_mathlib_Polynomial_repr___redArg___lam__2___closed__6_once, _init_lp_mathlib_Polynomial_repr___redArg___lam__2___closed__6);
v___x_403_ = ((lean_object*)(lp_mathlib_Polynomial_repr___redArg___lam__2___closed__7));
if (v_isShared_395_ == 0)
{
lean_ctor_set_tag(v___x_394_, 5);
lean_ctor_set(v___x_394_, 1, v___x_401_);
lean_ctor_set(v___x_394_, 0, v___x_403_);
v___x_405_ = v___x_394_;
goto v_reusejp_404_;
}
else
{
lean_object* v_reuseFailAlloc_413_; 
v_reuseFailAlloc_413_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_413_, 0, v___x_403_);
lean_ctor_set(v_reuseFailAlloc_413_, 1, v___x_401_);
v___x_405_ = v_reuseFailAlloc_413_;
goto v_reusejp_404_;
}
v_reusejp_404_:
{
lean_object* v___x_406_; lean_object* v___x_408_; 
v___x_406_ = ((lean_object*)(lp_mathlib_Polynomial_repr___redArg___lam__2___closed__8));
if (v_isShared_356_ == 0)
{
lean_ctor_set_tag(v___x_355_, 5);
lean_ctor_set(v___x_355_, 1, v___x_406_);
lean_ctor_set(v___x_355_, 0, v___x_405_);
v___x_408_ = v___x_355_;
goto v_reusejp_407_;
}
else
{
lean_object* v_reuseFailAlloc_412_; 
v_reuseFailAlloc_412_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_412_, 0, v___x_405_);
lean_ctor_set(v_reuseFailAlloc_412_, 1, v___x_406_);
v___x_408_ = v_reuseFailAlloc_412_;
goto v_reusejp_407_;
}
v_reusejp_407_:
{
lean_object* v___x_409_; uint8_t v___x_410_; lean_object* v___x_411_; 
v___x_409_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_409_, 0, v___x_402_);
lean_ctor_set(v___x_409_, 1, v___x_408_);
v___x_410_ = 0;
v___x_411_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_411_, 0, v___x_409_);
lean_ctor_set_uint8(v___x_411_, sizeof(void*)*1, v___x_410_);
return v___x_411_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_repr___redArg(lean_object* v_inst_421_, lean_object* v_inst_422_, lean_object* v_inst_423_){
_start:
{
lean_object* v___x_424_; lean_object* v___x_425_; lean_object* v_toOne_426_; lean_object* v___f_427_; lean_object* v___f_428_; lean_object* v___f_429_; lean_object* v___f_430_; 
v___x_424_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_421_);
v___x_425_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v___x_424_);
v_toOne_426_ = lean_ctor_get(v___x_425_, 2);
lean_inc(v_toOne_426_);
lean_dec_ref(v___x_425_);
v___f_427_ = ((lean_object*)(lp_mathlib_Polynomial_repr___redArg___closed__0));
v___f_428_ = ((lean_object*)(lp_mathlib_Polynomial_repr___redArg___closed__1));
v___f_429_ = ((lean_object*)(lp_mathlib_Polynomial_repr___redArg___closed__2));
v___f_430_ = lean_alloc_closure((void*)(lp_mathlib_Polynomial_repr___redArg___lam__2), 8, 6);
lean_closure_set(v___f_430_, 0, v_inst_423_);
lean_closure_set(v___f_430_, 1, v_toOne_426_);
lean_closure_set(v___f_430_, 2, v___f_428_);
lean_closure_set(v___f_430_, 3, v_inst_422_);
lean_closure_set(v___f_430_, 4, v___f_427_);
lean_closure_set(v___f_430_, 5, v___f_429_);
return v___f_430_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_repr(lean_object* v_R_431_, lean_object* v_inst_432_, lean_object* v_inst_433_, lean_object* v_inst_434_){
_start:
{
lean_object* v___x_435_; 
v___x_435_ = lp_mathlib_Polynomial_repr___redArg(v_inst_432_, v_inst_433_, v_inst_434_);
return v___x_435_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_AddChar(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Module(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_NoZeroDivisors(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_WithTop(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Action_Rat(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Sort(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FastInstance(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_LSum(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_AddChar(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_NoZeroDivisors(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_WithTop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Action_Rat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Sort(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FastInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_LSum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Polynomial_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_AddChar(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Module(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_NoZeroDivisors(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_WithTop(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Action_Rat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Sort(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FastInstance(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_LSum(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_AddChar(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_NoZeroDivisors(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_WithTop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Action_Rat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Sort(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FastInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_LSum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Polynomial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Polynomial_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
