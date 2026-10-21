// Lean compiler output
// Module: Mathlib.Data.Nat.Totient
// Imports: public import Init public meta import Init public import Mathlib.Algebra.BigOperators.Ring.Finset public import Mathlib.Algebra.CharP.Two public import Mathlib.Algebra.Order.AbsoluteValue.Basic public import Mathlib.Algebra.Order.BigOperators.Group.LocallyFinite public import Mathlib.Algebra.Order.BigOperators.GroupWithZero.Finset public import Mathlib.Data.Nat.Cast.Field public import Mathlib.Data.Nat.Factorization.Basic public import Mathlib.Data.Nat.Factorization.Induction public import Mathlib.Data.Nat.Periodic public import Mathlib.Tactic.Ring
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
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* lean_nat_gcd(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Level_ofNat(lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_Positivity_core(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_lit___override(lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_List_range(lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00Multiset_filter___at___00Finset_filter___at___00Nat_totient_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00Multiset_filter___at___00Finset_filter___at___00Nat_totient_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_filter___at___00Finset_filter___at___00Nat_totient_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_filter___at___00Finset_filter___at___00Nat_totient_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_totient(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_filter___at___00Nat_totient_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_filter___at___00Nat_totient_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_filter___at___00Nat_totient_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_filter___at___00Nat_totient_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_filter___at___00Finset_filter___at___00Nat_totient_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_filter___at___00Finset_filter___at___00Nat_totient_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Nat_term_u03c6___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_mathlib_Nat_term_u03c6___closed__0 = (const lean_object*)&lp_mathlib_Nat_term_u03c6___closed__0_value;
static const lean_string_object lp_mathlib_Nat_term_u03c6___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 5, .m_data = "termφ"};
static const lean_object* lp_mathlib_Nat_term_u03c6___closed__1 = (const lean_object*)&lp_mathlib_Nat_term_u03c6___closed__1_value;
static const lean_ctor_object lp_mathlib_Nat_term_u03c6___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Nat_term_u03c6___closed__0_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib_Nat_term_u03c6___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Nat_term_u03c6___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Nat_term_u03c6___closed__1_value),LEAN_SCALAR_PTR_LITERAL(131, 136, 5, 40, 237, 34, 162, 146)}};
static const lean_object* lp_mathlib_Nat_term_u03c6___closed__2 = (const lean_object*)&lp_mathlib_Nat_term_u03c6___closed__2_value;
static const lean_string_object lp_mathlib_Nat_term_u03c6___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 1, .m_data = "φ"};
static const lean_object* lp_mathlib_Nat_term_u03c6___closed__3 = (const lean_object*)&lp_mathlib_Nat_term_u03c6___closed__3_value;
static const lean_ctor_object lp_mathlib_Nat_term_u03c6___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Nat_term_u03c6___closed__3_value)}};
static const lean_object* lp_mathlib_Nat_term_u03c6___closed__4 = (const lean_object*)&lp_mathlib_Nat_term_u03c6___closed__4_value;
static const lean_ctor_object lp_mathlib_Nat_term_u03c6___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Nat_term_u03c6___closed__2_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Nat_term_u03c6___closed__4_value)}};
static const lean_object* lp_mathlib_Nat_term_u03c6___closed__5 = (const lean_object*)&lp_mathlib_Nat_term_u03c6___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Nat_term_u03c6 = (const lean_object*)&lp_mathlib_Nat_term_u03c6___closed__5_value;
static const lean_string_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______macroRules__Nat__term_u03c6__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Nat.totient"};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______macroRules__Nat__term_u03c6__1___closed__0 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______macroRules__Nat__term_u03c6__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______macroRules__Nat__term_u03c6__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______macroRules__Nat__term_u03c6__1___closed__1;
static const lean_string_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______macroRules__Nat__term_u03c6__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "totient"};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______macroRules__Nat__term_u03c6__1___closed__2 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______macroRules__Nat__term_u03c6__1___closed__2_value;
static const lean_ctor_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______macroRules__Nat__term_u03c6__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Nat_term_u03c6___closed__0_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______macroRules__Nat__term_u03c6__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______macroRules__Nat__term_u03c6__1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______macroRules__Nat__term_u03c6__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(157, 71, 93, 156, 233, 227, 208, 109)}};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______macroRules__Nat__term_u03c6__1___closed__3 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______macroRules__Nat__term_u03c6__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______macroRules__Nat__term_u03c6__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______macroRules__Nat__term_u03c6__1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______macroRules__Nat__term_u03c6__1___closed__4 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______macroRules__Nat__term_u03c6__1___closed__4_value;
static const lean_ctor_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______macroRules__Nat__term_u03c6__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______macroRules__Nat__term_u03c6__1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______macroRules__Nat__term_u03c6__1___closed__5 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______macroRules__Nat__term_u03c6__1___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______macroRules__Nat__term_u03c6__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______macroRules__Nat__term_u03c6__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______unexpand__Nat__totient__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______unexpand__Nat__totient__1___closed__0 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______unexpand__Nat__totient__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______unexpand__Nat__totient__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______unexpand__Nat__totient__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______unexpand__Nat__totient__1___closed__1 = (const lean_object*)&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______unexpand__Nat__totient__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______unexpand__Nat__totient__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______unexpand__Nat__totient__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__0___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__1(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "not Nat.totient"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__1;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Nat_term_u03c6___closed__0_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__3;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__5;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "LT"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "lt"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__6_value),LEAN_SCALAR_PTR_LITERAL(71, 235, 154, 184, 62, 135, 30, 248)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__7_value),LEAN_SCALAR_PTR_LITERAL(54, 235, 251, 9, 4, 74, 57, 164)}};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "OfNat"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ofNat"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__9_value),LEAN_SCALAR_PTR_LITERAL(135, 241, 166, 108, 243, 216, 193, 244)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__10_value),LEAN_SCALAR_PTR_LITERAL(2, 108, 58, 34, 100, 49, 50, 216)}};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__12_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__13;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Iff"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "mpr"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__14_value),LEAN_SCALAR_PTR_LITERAL(19, 54, 203, 28, 77, 25, 163, 137)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__16_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__15_value),LEAN_SCALAR_PTR_LITERAL(14, 81, 9, 215, 230, 198, 87, 3)}};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__16_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__17;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "instLTNat"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__18_value),LEAN_SCALAR_PTR_LITERAL(141, 27, 201, 217, 48, 203, 85, 203)}};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__19_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__20;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "instOfNatNat"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__21_value),LEAN_SCALAR_PTR_LITERAL(217, 8, 172, 44, 179, 254, 147, 95)}};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__22_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__23;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__24;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__25;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "totient_pos"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__27_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Nat_term_u03c6___closed__0_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__27_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__26_value),LEAN_SCALAR_PTR_LITERAL(227, 177, 46, 29, 67, 185, 23, 217)}};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__27_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__28;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__29_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__30;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___boxed, .m_arity = 10, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00Multiset_filter___at___00Finset_filter___at___00Nat_totient_spec__0_spec__0_spec__1(lean_object* v_n_1_, lean_object* v_a_2_, lean_object* v_a_3_){
_start:
{
if (lean_obj_tag(v_a_2_) == 0)
{
lean_object* v___x_4_; 
v___x_4_ = l_List_reverse___redArg(v_a_3_);
return v___x_4_;
}
else
{
lean_object* v_head_5_; lean_object* v_tail_6_; lean_object* v___x_8_; uint8_t v_isShared_9_; uint8_t v_isSharedCheck_18_; 
v_head_5_ = lean_ctor_get(v_a_2_, 0);
v_tail_6_ = lean_ctor_get(v_a_2_, 1);
v_isSharedCheck_18_ = !lean_is_exclusive(v_a_2_);
if (v_isSharedCheck_18_ == 0)
{
v___x_8_ = v_a_2_;
v_isShared_9_ = v_isSharedCheck_18_;
goto v_resetjp_7_;
}
else
{
lean_inc(v_tail_6_);
lean_inc(v_head_5_);
lean_dec(v_a_2_);
v___x_8_ = lean_box(0);
v_isShared_9_ = v_isSharedCheck_18_;
goto v_resetjp_7_;
}
v_resetjp_7_:
{
lean_object* v___x_10_; lean_object* v___x_11_; uint8_t v___x_12_; 
v___x_10_ = lean_nat_gcd(v_n_1_, v_head_5_);
v___x_11_ = lean_unsigned_to_nat(1u);
v___x_12_ = lean_nat_dec_eq(v___x_10_, v___x_11_);
lean_dec(v___x_10_);
if (v___x_12_ == 0)
{
lean_del_object(v___x_8_);
lean_dec(v_head_5_);
v_a_2_ = v_tail_6_;
goto _start;
}
else
{
lean_object* v___x_15_; 
if (v_isShared_9_ == 0)
{
lean_ctor_set(v___x_8_, 1, v_a_3_);
v___x_15_ = v___x_8_;
goto v_reusejp_14_;
}
else
{
lean_object* v_reuseFailAlloc_17_; 
v_reuseFailAlloc_17_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_17_, 0, v_head_5_);
lean_ctor_set(v_reuseFailAlloc_17_, 1, v_a_3_);
v___x_15_ = v_reuseFailAlloc_17_;
goto v_reusejp_14_;
}
v_reusejp_14_:
{
v_a_2_ = v_tail_6_;
v_a_3_ = v___x_15_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00Multiset_filter___at___00Finset_filter___at___00Nat_totient_spec__0_spec__0_spec__1___boxed(lean_object* v_n_19_, lean_object* v_a_20_, lean_object* v_a_21_){
_start:
{
lean_object* v_res_22_; 
v_res_22_ = lp_mathlib_List_filterTR_loop___at___00Multiset_filter___at___00Finset_filter___at___00Nat_totient_spec__0_spec__0_spec__1(v_n_19_, v_a_20_, v_a_21_);
lean_dec(v_n_19_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_filter___at___00Finset_filter___at___00Nat_totient_spec__0_spec__0___redArg(lean_object* v_n_23_, lean_object* v_s_24_){
_start:
{
lean_object* v___x_25_; lean_object* v___x_26_; 
v___x_25_ = lean_box(0);
v___x_26_ = lp_mathlib_List_filterTR_loop___at___00Multiset_filter___at___00Finset_filter___at___00Nat_totient_spec__0_spec__0_spec__1(v_n_23_, v_s_24_, v___x_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_filter___at___00Finset_filter___at___00Nat_totient_spec__0_spec__0___redArg___boxed(lean_object* v_n_27_, lean_object* v_s_28_){
_start:
{
lean_object* v_res_29_; 
v_res_29_ = lp_mathlib_Multiset_filter___at___00Finset_filter___at___00Nat_totient_spec__0_spec__0___redArg(v_n_27_, v_s_28_);
lean_dec(v_n_27_);
return v_res_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_totient(lean_object* v_n_30_){
_start:
{
lean_object* v___x_31_; lean_object* v___x_32_; lean_object* v___x_33_; 
lean_inc(v_n_30_);
v___x_31_ = l_List_range(v_n_30_);
v___x_32_ = lp_mathlib_Multiset_filter___at___00Finset_filter___at___00Nat_totient_spec__0_spec__0___redArg(v_n_30_, v___x_31_);
lean_dec(v_n_30_);
v___x_33_ = l_List_lengthTR___redArg(v___x_32_);
lean_dec(v___x_32_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_filter___at___00Nat_totient_spec__0___redArg(lean_object* v_n_34_, lean_object* v_s_35_){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lp_mathlib_Multiset_filter___at___00Finset_filter___at___00Nat_totient_spec__0_spec__0___redArg(v_n_34_, v_s_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_filter___at___00Nat_totient_spec__0___redArg___boxed(lean_object* v_n_37_, lean_object* v_s_38_){
_start:
{
lean_object* v_res_39_; 
v_res_39_ = lp_mathlib_Finset_filter___at___00Nat_totient_spec__0___redArg(v_n_37_, v_s_38_);
lean_dec(v_n_37_);
return v_res_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_filter___at___00Nat_totient_spec__0(lean_object* v_n_40_, lean_object* v_p_41_, lean_object* v_s_42_){
_start:
{
lean_object* v___x_43_; 
v___x_43_ = lp_mathlib_Multiset_filter___at___00Finset_filter___at___00Nat_totient_spec__0_spec__0___redArg(v_n_40_, v_s_42_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_filter___at___00Nat_totient_spec__0___boxed(lean_object* v_n_44_, lean_object* v_p_45_, lean_object* v_s_46_){
_start:
{
lean_object* v_res_47_; 
v_res_47_ = lp_mathlib_Finset_filter___at___00Nat_totient_spec__0(v_n_44_, v_p_45_, v_s_46_);
lean_dec(v_n_44_);
return v_res_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_filter___at___00Finset_filter___at___00Nat_totient_spec__0_spec__0(lean_object* v_n_48_, lean_object* v_p_49_, lean_object* v_s_50_){
_start:
{
lean_object* v___x_51_; 
v___x_51_ = lp_mathlib_Multiset_filter___at___00Finset_filter___at___00Nat_totient_spec__0_spec__0___redArg(v_n_48_, v_s_50_);
return v___x_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_filter___at___00Finset_filter___at___00Nat_totient_spec__0_spec__0___boxed(lean_object* v_n_52_, lean_object* v_p_53_, lean_object* v_s_54_){
_start:
{
lean_object* v_res_55_; 
v_res_55_ = lp_mathlib_Multiset_filter___at___00Finset_filter___at___00Nat_totient_spec__0_spec__0(v_n_52_, v_p_53_, v_s_54_);
lean_dec(v_n_52_);
return v_res_55_;
}
}
static lean_object* _init_lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______macroRules__Nat__term_u03c6__1___closed__1(void){
_start:
{
lean_object* v___x_70_; lean_object* v___x_71_; 
v___x_70_ = ((lean_object*)(lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______macroRules__Nat__term_u03c6__1___closed__0));
v___x_71_ = l_String_toRawSubstring_x27(v___x_70_);
return v___x_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______macroRules__Nat__term_u03c6__1(lean_object* v_x_82_, lean_object* v_a_83_, lean_object* v_a_84_){
_start:
{
lean_object* v___x_85_; uint8_t v___x_86_; 
v___x_85_ = ((lean_object*)(lp_mathlib_Nat_term_u03c6___closed__2));
v___x_86_ = l_Lean_Syntax_isOfKind(v_x_82_, v___x_85_);
if (v___x_86_ == 0)
{
lean_object* v___x_87_; lean_object* v___x_88_; 
v___x_87_ = lean_box(1);
v___x_88_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_88_, 0, v___x_87_);
lean_ctor_set(v___x_88_, 1, v_a_84_);
return v___x_88_;
}
else
{
lean_object* v_quotContext_89_; lean_object* v_currMacroScope_90_; lean_object* v_ref_91_; uint8_t v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; 
v_quotContext_89_ = lean_ctor_get(v_a_83_, 1);
v_currMacroScope_90_ = lean_ctor_get(v_a_83_, 2);
v_ref_91_ = lean_ctor_get(v_a_83_, 5);
v___x_92_ = 0;
v___x_93_ = l_Lean_SourceInfo_fromRef(v_ref_91_, v___x_92_);
v___x_94_ = lean_obj_once(&lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______macroRules__Nat__term_u03c6__1___closed__1, &lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______macroRules__Nat__term_u03c6__1___closed__1_once, _init_lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______macroRules__Nat__term_u03c6__1___closed__1);
v___x_95_ = ((lean_object*)(lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______macroRules__Nat__term_u03c6__1___closed__3));
lean_inc(v_currMacroScope_90_);
lean_inc(v_quotContext_89_);
v___x_96_ = l_Lean_addMacroScope(v_quotContext_89_, v___x_95_, v_currMacroScope_90_);
v___x_97_ = ((lean_object*)(lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______macroRules__Nat__term_u03c6__1___closed__5));
v___x_98_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_98_, 0, v___x_93_);
lean_ctor_set(v___x_98_, 1, v___x_94_);
lean_ctor_set(v___x_98_, 2, v___x_96_);
lean_ctor_set(v___x_98_, 3, v___x_97_);
v___x_99_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_99_, 0, v___x_98_);
lean_ctor_set(v___x_99_, 1, v_a_84_);
return v___x_99_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______macroRules__Nat__term_u03c6__1___boxed(lean_object* v_x_100_, lean_object* v_a_101_, lean_object* v_a_102_){
_start:
{
lean_object* v_res_103_; 
v_res_103_ = lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______macroRules__Nat__term_u03c6__1(v_x_100_, v_a_101_, v_a_102_);
lean_dec_ref(v_a_101_);
return v_res_103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______unexpand__Nat__totient__1(lean_object* v_x_107_, lean_object* v_a_108_, lean_object* v_a_109_){
_start:
{
lean_object* v___x_110_; uint8_t v___x_111_; 
v___x_110_ = ((lean_object*)(lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______unexpand__Nat__totient__1___closed__1));
lean_inc(v_x_107_);
v___x_111_ = l_Lean_Syntax_isOfKind(v_x_107_, v___x_110_);
if (v___x_111_ == 0)
{
lean_object* v___x_112_; lean_object* v___x_113_; 
lean_dec(v_x_107_);
v___x_112_ = lean_box(0);
v___x_113_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_113_, 0, v___x_112_);
lean_ctor_set(v___x_113_, 1, v_a_109_);
return v___x_113_;
}
else
{
lean_object* v_ref_114_; uint8_t v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; 
v_ref_114_ = l_Lean_replaceRef(v_x_107_, v_a_108_);
lean_dec(v_x_107_);
v___x_115_ = 0;
v___x_116_ = l_Lean_SourceInfo_fromRef(v_ref_114_, v___x_115_);
lean_dec(v_ref_114_);
v___x_117_ = ((lean_object*)(lp_mathlib_Nat_term_u03c6___closed__2));
v___x_118_ = ((lean_object*)(lp_mathlib_Nat_term_u03c6___closed__3));
lean_inc(v___x_116_);
v___x_119_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_119_, 0, v___x_116_);
lean_ctor_set(v___x_119_, 1, v___x_118_);
v___x_120_ = l_Lean_Syntax_node1(v___x_116_, v___x_117_, v___x_119_);
v___x_121_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_121_, 0, v___x_120_);
lean_ctor_set(v___x_121_, 1, v_a_109_);
return v___x_121_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______unexpand__Nat__totient__1___boxed(lean_object* v_x_122_, lean_object* v_a_123_, lean_object* v_a_124_){
_start:
{
lean_object* v_res_125_; 
v_res_125_ = lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______unexpand__Nat__totient__1(v_x_122_, v_a_123_, v_a_124_);
lean_dec(v_a_123_);
return v_res_125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__0___redArg(lean_object* v_k_126_, uint8_t v_allowLevelAssignments_127_, lean_object* v___y_128_, lean_object* v___y_129_, lean_object* v___y_130_, lean_object* v___y_131_){
_start:
{
lean_object* v___x_133_; 
v___x_133_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), v_allowLevelAssignments_127_, v_k_126_, v___y_128_, v___y_129_, v___y_130_, v___y_131_);
if (lean_obj_tag(v___x_133_) == 0)
{
lean_object* v_a_134_; lean_object* v___x_136_; uint8_t v_isShared_137_; uint8_t v_isSharedCheck_141_; 
v_a_134_ = lean_ctor_get(v___x_133_, 0);
v_isSharedCheck_141_ = !lean_is_exclusive(v___x_133_);
if (v_isSharedCheck_141_ == 0)
{
v___x_136_ = v___x_133_;
v_isShared_137_ = v_isSharedCheck_141_;
goto v_resetjp_135_;
}
else
{
lean_inc(v_a_134_);
lean_dec(v___x_133_);
v___x_136_ = lean_box(0);
v_isShared_137_ = v_isSharedCheck_141_;
goto v_resetjp_135_;
}
v_resetjp_135_:
{
lean_object* v___x_139_; 
if (v_isShared_137_ == 0)
{
v___x_139_ = v___x_136_;
goto v_reusejp_138_;
}
else
{
lean_object* v_reuseFailAlloc_140_; 
v_reuseFailAlloc_140_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_140_, 0, v_a_134_);
v___x_139_ = v_reuseFailAlloc_140_;
goto v_reusejp_138_;
}
v_reusejp_138_:
{
return v___x_139_;
}
}
}
else
{
lean_object* v_a_142_; lean_object* v___x_144_; uint8_t v_isShared_145_; uint8_t v_isSharedCheck_149_; 
v_a_142_ = lean_ctor_get(v___x_133_, 0);
v_isSharedCheck_149_ = !lean_is_exclusive(v___x_133_);
if (v_isSharedCheck_149_ == 0)
{
v___x_144_ = v___x_133_;
v_isShared_145_ = v_isSharedCheck_149_;
goto v_resetjp_143_;
}
else
{
lean_inc(v_a_142_);
lean_dec(v___x_133_);
v___x_144_ = lean_box(0);
v_isShared_145_ = v_isSharedCheck_149_;
goto v_resetjp_143_;
}
v_resetjp_143_:
{
lean_object* v___x_147_; 
if (v_isShared_145_ == 0)
{
v___x_147_ = v___x_144_;
goto v_reusejp_146_;
}
else
{
lean_object* v_reuseFailAlloc_148_; 
v_reuseFailAlloc_148_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_148_, 0, v_a_142_);
v___x_147_ = v_reuseFailAlloc_148_;
goto v_reusejp_146_;
}
v_reusejp_146_:
{
return v___x_147_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__0___redArg___boxed(lean_object* v_k_150_, lean_object* v_allowLevelAssignments_151_, lean_object* v___y_152_, lean_object* v___y_153_, lean_object* v___y_154_, lean_object* v___y_155_, lean_object* v___y_156_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_157_; lean_object* v_res_158_; 
v_allowLevelAssignments_boxed_157_ = lean_unbox(v_allowLevelAssignments_151_);
v_res_158_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__0___redArg(v_k_150_, v_allowLevelAssignments_boxed_157_, v___y_152_, v___y_153_, v___y_154_, v___y_155_);
lean_dec(v___y_155_);
lean_dec_ref(v___y_154_);
lean_dec(v___y_153_);
lean_dec_ref(v___y_152_);
return v_res_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__0(lean_object* v_00_u03b1_159_, lean_object* v_k_160_, uint8_t v_allowLevelAssignments_161_, lean_object* v___y_162_, lean_object* v___y_163_, lean_object* v___y_164_, lean_object* v___y_165_){
_start:
{
lean_object* v___x_167_; 
v___x_167_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__0___redArg(v_k_160_, v_allowLevelAssignments_161_, v___y_162_, v___y_163_, v___y_164_, v___y_165_);
return v___x_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__0___boxed(lean_object* v_00_u03b1_168_, lean_object* v_k_169_, lean_object* v_allowLevelAssignments_170_, lean_object* v___y_171_, lean_object* v___y_172_, lean_object* v___y_173_, lean_object* v___y_174_, lean_object* v___y_175_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_176_; lean_object* v_res_177_; 
v_allowLevelAssignments_boxed_176_ = lean_unbox(v_allowLevelAssignments_170_);
v_res_177_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__0(v_00_u03b1_168_, v_k_169_, v_allowLevelAssignments_boxed_176_, v___y_171_, v___y_172_, v___y_173_, v___y_174_);
lean_dec(v___y_174_);
lean_dec_ref(v___y_173_);
lean_dec(v___y_172_);
lean_dec_ref(v___y_171_);
return v_res_177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__2___redArg(lean_object* v_e_178_, lean_object* v___y_179_){
_start:
{
uint8_t v___x_181_; 
v___x_181_ = l_Lean_Expr_hasMVar(v_e_178_);
if (v___x_181_ == 0)
{
lean_object* v___x_182_; 
v___x_182_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_182_, 0, v_e_178_);
return v___x_182_;
}
else
{
lean_object* v___x_183_; lean_object* v_mctx_184_; lean_object* v___x_185_; lean_object* v_fst_186_; lean_object* v_snd_187_; lean_object* v___x_188_; lean_object* v_cache_189_; lean_object* v_zetaDeltaFVarIds_190_; lean_object* v_postponed_191_; lean_object* v_diag_192_; lean_object* v___x_194_; uint8_t v_isShared_195_; uint8_t v_isSharedCheck_201_; 
v___x_183_ = lean_st_ref_get(v___y_179_);
v_mctx_184_ = lean_ctor_get(v___x_183_, 0);
lean_inc_ref(v_mctx_184_);
lean_dec(v___x_183_);
v___x_185_ = l_Lean_instantiateMVarsCore(v_mctx_184_, v_e_178_);
v_fst_186_ = lean_ctor_get(v___x_185_, 0);
lean_inc(v_fst_186_);
v_snd_187_ = lean_ctor_get(v___x_185_, 1);
lean_inc(v_snd_187_);
lean_dec_ref(v___x_185_);
v___x_188_ = lean_st_ref_take(v___y_179_);
v_cache_189_ = lean_ctor_get(v___x_188_, 1);
v_zetaDeltaFVarIds_190_ = lean_ctor_get(v___x_188_, 2);
v_postponed_191_ = lean_ctor_get(v___x_188_, 3);
v_diag_192_ = lean_ctor_get(v___x_188_, 4);
v_isSharedCheck_201_ = !lean_is_exclusive(v___x_188_);
if (v_isSharedCheck_201_ == 0)
{
lean_object* v_unused_202_; 
v_unused_202_ = lean_ctor_get(v___x_188_, 0);
lean_dec(v_unused_202_);
v___x_194_ = v___x_188_;
v_isShared_195_ = v_isSharedCheck_201_;
goto v_resetjp_193_;
}
else
{
lean_inc(v_diag_192_);
lean_inc(v_postponed_191_);
lean_inc(v_zetaDeltaFVarIds_190_);
lean_inc(v_cache_189_);
lean_dec(v___x_188_);
v___x_194_ = lean_box(0);
v_isShared_195_ = v_isSharedCheck_201_;
goto v_resetjp_193_;
}
v_resetjp_193_:
{
lean_object* v___x_197_; 
if (v_isShared_195_ == 0)
{
lean_ctor_set(v___x_194_, 0, v_snd_187_);
v___x_197_ = v___x_194_;
goto v_reusejp_196_;
}
else
{
lean_object* v_reuseFailAlloc_200_; 
v_reuseFailAlloc_200_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_200_, 0, v_snd_187_);
lean_ctor_set(v_reuseFailAlloc_200_, 1, v_cache_189_);
lean_ctor_set(v_reuseFailAlloc_200_, 2, v_zetaDeltaFVarIds_190_);
lean_ctor_set(v_reuseFailAlloc_200_, 3, v_postponed_191_);
lean_ctor_set(v_reuseFailAlloc_200_, 4, v_diag_192_);
v___x_197_ = v_reuseFailAlloc_200_;
goto v_reusejp_196_;
}
v_reusejp_196_:
{
lean_object* v___x_198_; lean_object* v___x_199_; 
v___x_198_ = lean_st_ref_set(v___y_179_, v___x_197_);
v___x_199_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_199_, 0, v_fst_186_);
return v___x_199_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__2___redArg___boxed(lean_object* v_e_203_, lean_object* v___y_204_, lean_object* v___y_205_){
_start:
{
lean_object* v_res_206_; 
v_res_206_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__2___redArg(v_e_203_, v___y_204_);
lean_dec(v___y_204_);
return v_res_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__2(lean_object* v_e_207_, lean_object* v___y_208_, lean_object* v___y_209_, lean_object* v___y_210_, lean_object* v___y_211_){
_start:
{
lean_object* v___x_213_; 
v___x_213_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__2___redArg(v_e_207_, v___y_209_);
return v___x_213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__2___boxed(lean_object* v_e_214_, lean_object* v___y_215_, lean_object* v___y_216_, lean_object* v___y_217_, lean_object* v___y_218_, lean_object* v___y_219_){
_start:
{
lean_object* v_res_220_; 
v_res_220_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__2(v_e_214_, v___y_215_, v___y_216_, v___y_217_, v___y_218_);
lean_dec(v___y_218_);
lean_dec_ref(v___y_217_);
lean_dec(v___y_216_);
lean_dec_ref(v___y_215_);
return v_res_220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__0(uint8_t v___x_221_, lean_object* v___x_222_, lean_object* v_00_u03b1_223_, lean_object* v___y_224_, lean_object* v___y_225_, lean_object* v___y_226_, lean_object* v___y_227_){
_start:
{
lean_object* v_keyedConfig_229_; uint8_t v_trackZetaDelta_230_; lean_object* v_zetaDeltaSet_231_; lean_object* v_lctx_232_; lean_object* v_localInstances_233_; lean_object* v_defEqCtx_x3f_234_; lean_object* v_synthPendingDepth_235_; lean_object* v_customCanUnfoldPredicate_x3f_236_; uint8_t v_univApprox_237_; uint8_t v_inTypeClassResolution_238_; uint8_t v_cacheInferType_239_; lean_object* v___x_241_; uint8_t v_isShared_242_; uint8_t v_isSharedCheck_248_; 
v_keyedConfig_229_ = lean_ctor_get(v___y_224_, 0);
v_trackZetaDelta_230_ = lean_ctor_get_uint8(v___y_224_, sizeof(void*)*7);
v_zetaDeltaSet_231_ = lean_ctor_get(v___y_224_, 1);
v_lctx_232_ = lean_ctor_get(v___y_224_, 2);
v_localInstances_233_ = lean_ctor_get(v___y_224_, 3);
v_defEqCtx_x3f_234_ = lean_ctor_get(v___y_224_, 4);
v_synthPendingDepth_235_ = lean_ctor_get(v___y_224_, 5);
v_customCanUnfoldPredicate_x3f_236_ = lean_ctor_get(v___y_224_, 6);
v_univApprox_237_ = lean_ctor_get_uint8(v___y_224_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_238_ = lean_ctor_get_uint8(v___y_224_, sizeof(void*)*7 + 2);
v_cacheInferType_239_ = lean_ctor_get_uint8(v___y_224_, sizeof(void*)*7 + 3);
v_isSharedCheck_248_ = !lean_is_exclusive(v___y_224_);
if (v_isSharedCheck_248_ == 0)
{
v___x_241_ = v___y_224_;
v_isShared_242_ = v_isSharedCheck_248_;
goto v_resetjp_240_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_236_);
lean_inc(v_synthPendingDepth_235_);
lean_inc(v_defEqCtx_x3f_234_);
lean_inc(v_localInstances_233_);
lean_inc(v_lctx_232_);
lean_inc(v_zetaDeltaSet_231_);
lean_inc(v_keyedConfig_229_);
lean_dec(v___y_224_);
v___x_241_ = lean_box(0);
v_isShared_242_ = v_isSharedCheck_248_;
goto v_resetjp_240_;
}
v_resetjp_240_:
{
lean_object* v___x_243_; lean_object* v___x_245_; 
v___x_243_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_221_, v_keyedConfig_229_);
if (v_isShared_242_ == 0)
{
lean_ctor_set(v___x_241_, 0, v___x_243_);
v___x_245_ = v___x_241_;
goto v_reusejp_244_;
}
else
{
lean_object* v_reuseFailAlloc_247_; 
v_reuseFailAlloc_247_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_247_, 0, v___x_243_);
lean_ctor_set(v_reuseFailAlloc_247_, 1, v_zetaDeltaSet_231_);
lean_ctor_set(v_reuseFailAlloc_247_, 2, v_lctx_232_);
lean_ctor_set(v_reuseFailAlloc_247_, 3, v_localInstances_233_);
lean_ctor_set(v_reuseFailAlloc_247_, 4, v_defEqCtx_x3f_234_);
lean_ctor_set(v_reuseFailAlloc_247_, 5, v_synthPendingDepth_235_);
lean_ctor_set(v_reuseFailAlloc_247_, 6, v_customCanUnfoldPredicate_x3f_236_);
lean_ctor_set_uint8(v_reuseFailAlloc_247_, sizeof(void*)*7, v_trackZetaDelta_230_);
lean_ctor_set_uint8(v_reuseFailAlloc_247_, sizeof(void*)*7 + 1, v_univApprox_237_);
lean_ctor_set_uint8(v_reuseFailAlloc_247_, sizeof(void*)*7 + 2, v_inTypeClassResolution_238_);
lean_ctor_set_uint8(v_reuseFailAlloc_247_, sizeof(void*)*7 + 3, v_cacheInferType_239_);
v___x_245_ = v_reuseFailAlloc_247_;
goto v_reusejp_244_;
}
v_reusejp_244_:
{
lean_object* v___x_246_; 
v___x_246_ = l_Lean_Meta_isExprDefEq(v___x_222_, v_00_u03b1_223_, v___x_245_, v___y_225_, v___y_226_, v___y_227_);
lean_dec_ref(v___x_245_);
return v___x_246_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__0___boxed(lean_object* v___x_249_, lean_object* v___x_250_, lean_object* v_00_u03b1_251_, lean_object* v___y_252_, lean_object* v___y_253_, lean_object* v___y_254_, lean_object* v___y_255_, lean_object* v___y_256_){
_start:
{
uint8_t v___x_5876__boxed_257_; lean_object* v_res_258_; 
v___x_5876__boxed_257_ = lean_unbox(v___x_249_);
v_res_258_ = lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__0(v___x_5876__boxed_257_, v___x_250_, v_00_u03b1_251_, v___y_252_, v___y_253_, v___y_254_, v___y_255_);
lean_dec(v___y_255_);
lean_dec_ref(v___y_254_);
lean_dec(v___y_253_);
return v_res_258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__1(lean_object* v___x_259_, uint8_t v___x_260_, lean_object* v___x_261_, lean_object* v___x_262_, lean_object* v___x_263_, uint8_t v___x_264_, lean_object* v_e_265_, uint8_t v___x_266_, uint8_t v_a_267_, lean_object* v___y_268_, lean_object* v___y_269_, lean_object* v___y_270_, lean_object* v___y_271_){
_start:
{
lean_object* v___x_273_; 
v___x_273_ = l_Lean_Meta_mkFreshExprMVar(v___x_259_, v___x_260_, v___x_261_, v___y_268_, v___y_269_, v___y_270_, v___y_271_);
if (lean_obj_tag(v___x_273_) == 0)
{
lean_object* v_a_274_; lean_object* v_keyedConfig_275_; uint8_t v_trackZetaDelta_276_; lean_object* v_zetaDeltaSet_277_; lean_object* v_lctx_278_; lean_object* v_localInstances_279_; lean_object* v_defEqCtx_x3f_280_; lean_object* v_synthPendingDepth_281_; lean_object* v_customCanUnfoldPredicate_x3f_282_; uint8_t v_univApprox_283_; uint8_t v_inTypeClassResolution_284_; uint8_t v_cacheInferType_285_; lean_object* v___x_287_; uint8_t v_isShared_288_; uint8_t v_isSharedCheck_328_; 
v_a_274_ = lean_ctor_get(v___x_273_, 0);
lean_inc(v_a_274_);
lean_dec_ref_known(v___x_273_, 1);
v_keyedConfig_275_ = lean_ctor_get(v___y_268_, 0);
v_trackZetaDelta_276_ = lean_ctor_get_uint8(v___y_268_, sizeof(void*)*7);
v_zetaDeltaSet_277_ = lean_ctor_get(v___y_268_, 1);
v_lctx_278_ = lean_ctor_get(v___y_268_, 2);
v_localInstances_279_ = lean_ctor_get(v___y_268_, 3);
v_defEqCtx_x3f_280_ = lean_ctor_get(v___y_268_, 4);
v_synthPendingDepth_281_ = lean_ctor_get(v___y_268_, 5);
v_customCanUnfoldPredicate_x3f_282_ = lean_ctor_get(v___y_268_, 6);
v_univApprox_283_ = lean_ctor_get_uint8(v___y_268_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_284_ = lean_ctor_get_uint8(v___y_268_, sizeof(void*)*7 + 2);
v_cacheInferType_285_ = lean_ctor_get_uint8(v___y_268_, sizeof(void*)*7 + 3);
v_isSharedCheck_328_ = !lean_is_exclusive(v___y_268_);
if (v_isSharedCheck_328_ == 0)
{
v___x_287_ = v___y_268_;
v_isShared_288_ = v_isSharedCheck_328_;
goto v_resetjp_286_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_282_);
lean_inc(v_synthPendingDepth_281_);
lean_inc(v_defEqCtx_x3f_280_);
lean_inc(v_localInstances_279_);
lean_inc(v_lctx_278_);
lean_inc(v_zetaDeltaSet_277_);
lean_inc(v_keyedConfig_275_);
lean_dec(v___y_268_);
v___x_287_ = lean_box(0);
v_isShared_288_ = v_isSharedCheck_328_;
goto v_resetjp_286_;
}
v_resetjp_286_:
{
lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_295_; 
v___x_289_ = ((lean_object*)(lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______macroRules__Nat__term_u03c6__1___closed__2));
v___x_290_ = l_Lean_Name_mkStr2(v___x_262_, v___x_289_);
v___x_291_ = l_Lean_Expr_const___override(v___x_290_, v___x_263_);
lean_inc(v_a_274_);
v___x_292_ = l_Lean_Expr_app___override(v___x_291_, v_a_274_);
v___x_293_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_264_, v_keyedConfig_275_);
if (v_isShared_288_ == 0)
{
lean_ctor_set(v___x_287_, 0, v___x_293_);
v___x_295_ = v___x_287_;
goto v_reusejp_294_;
}
else
{
lean_object* v_reuseFailAlloc_327_; 
v_reuseFailAlloc_327_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_327_, 0, v___x_293_);
lean_ctor_set(v_reuseFailAlloc_327_, 1, v_zetaDeltaSet_277_);
lean_ctor_set(v_reuseFailAlloc_327_, 2, v_lctx_278_);
lean_ctor_set(v_reuseFailAlloc_327_, 3, v_localInstances_279_);
lean_ctor_set(v_reuseFailAlloc_327_, 4, v_defEqCtx_x3f_280_);
lean_ctor_set(v_reuseFailAlloc_327_, 5, v_synthPendingDepth_281_);
lean_ctor_set(v_reuseFailAlloc_327_, 6, v_customCanUnfoldPredicate_x3f_282_);
lean_ctor_set_uint8(v_reuseFailAlloc_327_, sizeof(void*)*7, v_trackZetaDelta_276_);
lean_ctor_set_uint8(v_reuseFailAlloc_327_, sizeof(void*)*7 + 1, v_univApprox_283_);
lean_ctor_set_uint8(v_reuseFailAlloc_327_, sizeof(void*)*7 + 2, v_inTypeClassResolution_284_);
lean_ctor_set_uint8(v_reuseFailAlloc_327_, sizeof(void*)*7 + 3, v_cacheInferType_285_);
v___x_295_ = v_reuseFailAlloc_327_;
goto v_reusejp_294_;
}
v_reusejp_294_:
{
lean_object* v___x_296_; 
v___x_296_ = l_Lean_Meta_isExprDefEq(v___x_292_, v_e_265_, v___x_295_, v___y_269_, v___y_270_, v___y_271_);
lean_dec_ref(v___x_295_);
if (lean_obj_tag(v___x_296_) == 0)
{
lean_object* v_a_297_; lean_object* v___x_299_; uint8_t v_isShared_300_; uint8_t v_isSharedCheck_318_; 
v_a_297_ = lean_ctor_get(v___x_296_, 0);
v_isSharedCheck_318_ = !lean_is_exclusive(v___x_296_);
if (v_isSharedCheck_318_ == 0)
{
v___x_299_ = v___x_296_;
v_isShared_300_ = v_isSharedCheck_318_;
goto v_resetjp_298_;
}
else
{
lean_inc(v_a_297_);
lean_dec(v___x_296_);
v___x_299_ = lean_box(0);
v_isShared_300_ = v_isSharedCheck_318_;
goto v_resetjp_298_;
}
v_resetjp_298_:
{
uint8_t v___x_301_; 
v___x_301_ = lean_unbox(v_a_297_);
lean_dec(v_a_297_);
if (v___x_301_ == 0)
{
lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v___x_305_; 
v___x_302_ = lean_box(v___x_266_);
v___x_303_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_303_, 0, v_a_274_);
lean_ctor_set(v___x_303_, 1, v___x_302_);
if (v_isShared_300_ == 0)
{
lean_ctor_set(v___x_299_, 0, v___x_303_);
v___x_305_ = v___x_299_;
goto v_reusejp_304_;
}
else
{
lean_object* v_reuseFailAlloc_306_; 
v_reuseFailAlloc_306_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_306_, 0, v___x_303_);
v___x_305_ = v_reuseFailAlloc_306_;
goto v_reusejp_304_;
}
v_reusejp_304_:
{
return v___x_305_;
}
}
else
{
lean_object* v___x_307_; lean_object* v_a_308_; lean_object* v___x_310_; uint8_t v_isShared_311_; uint8_t v_isSharedCheck_317_; 
lean_del_object(v___x_299_);
v___x_307_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__2___redArg(v_a_274_, v___y_269_);
v_a_308_ = lean_ctor_get(v___x_307_, 0);
v_isSharedCheck_317_ = !lean_is_exclusive(v___x_307_);
if (v_isSharedCheck_317_ == 0)
{
v___x_310_ = v___x_307_;
v_isShared_311_ = v_isSharedCheck_317_;
goto v_resetjp_309_;
}
else
{
lean_inc(v_a_308_);
lean_dec(v___x_307_);
v___x_310_ = lean_box(0);
v_isShared_311_ = v_isSharedCheck_317_;
goto v_resetjp_309_;
}
v_resetjp_309_:
{
lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_315_; 
v___x_312_ = lean_box(v_a_267_);
v___x_313_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_313_, 0, v_a_308_);
lean_ctor_set(v___x_313_, 1, v___x_312_);
if (v_isShared_311_ == 0)
{
lean_ctor_set(v___x_310_, 0, v___x_313_);
v___x_315_ = v___x_310_;
goto v_reusejp_314_;
}
else
{
lean_object* v_reuseFailAlloc_316_; 
v_reuseFailAlloc_316_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_316_, 0, v___x_313_);
v___x_315_ = v_reuseFailAlloc_316_;
goto v_reusejp_314_;
}
v_reusejp_314_:
{
return v___x_315_;
}
}
}
}
}
else
{
lean_object* v_a_319_; lean_object* v___x_321_; uint8_t v_isShared_322_; uint8_t v_isSharedCheck_326_; 
lean_dec(v_a_274_);
v_a_319_ = lean_ctor_get(v___x_296_, 0);
v_isSharedCheck_326_ = !lean_is_exclusive(v___x_296_);
if (v_isSharedCheck_326_ == 0)
{
v___x_321_ = v___x_296_;
v_isShared_322_ = v_isSharedCheck_326_;
goto v_resetjp_320_;
}
else
{
lean_inc(v_a_319_);
lean_dec(v___x_296_);
v___x_321_ = lean_box(0);
v_isShared_322_ = v_isSharedCheck_326_;
goto v_resetjp_320_;
}
v_resetjp_320_:
{
lean_object* v___x_324_; 
if (v_isShared_322_ == 0)
{
v___x_324_ = v___x_321_;
goto v_reusejp_323_;
}
else
{
lean_object* v_reuseFailAlloc_325_; 
v_reuseFailAlloc_325_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_325_, 0, v_a_319_);
v___x_324_ = v_reuseFailAlloc_325_;
goto v_reusejp_323_;
}
v_reusejp_323_:
{
return v___x_324_;
}
}
}
}
}
}
else
{
lean_object* v_a_329_; lean_object* v___x_331_; uint8_t v_isShared_332_; uint8_t v_isSharedCheck_336_; 
lean_dec_ref(v___y_268_);
lean_dec_ref(v_e_265_);
lean_dec(v___x_263_);
lean_dec_ref(v___x_262_);
v_a_329_ = lean_ctor_get(v___x_273_, 0);
v_isSharedCheck_336_ = !lean_is_exclusive(v___x_273_);
if (v_isSharedCheck_336_ == 0)
{
v___x_331_ = v___x_273_;
v_isShared_332_ = v_isSharedCheck_336_;
goto v_resetjp_330_;
}
else
{
lean_inc(v_a_329_);
lean_dec(v___x_273_);
v___x_331_ = lean_box(0);
v_isShared_332_ = v_isSharedCheck_336_;
goto v_resetjp_330_;
}
v_resetjp_330_:
{
lean_object* v___x_334_; 
if (v_isShared_332_ == 0)
{
v___x_334_ = v___x_331_;
goto v_reusejp_333_;
}
else
{
lean_object* v_reuseFailAlloc_335_; 
v_reuseFailAlloc_335_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_335_, 0, v_a_329_);
v___x_334_ = v_reuseFailAlloc_335_;
goto v_reusejp_333_;
}
v_reusejp_333_:
{
return v___x_334_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__1___boxed(lean_object* v___x_337_, lean_object* v___x_338_, lean_object* v___x_339_, lean_object* v___x_340_, lean_object* v___x_341_, lean_object* v___x_342_, lean_object* v_e_343_, lean_object* v___x_344_, lean_object* v_a_345_, lean_object* v___y_346_, lean_object* v___y_347_, lean_object* v___y_348_, lean_object* v___y_349_, lean_object* v___y_350_){
_start:
{
uint8_t v___x_5917__boxed_351_; uint8_t v___x_5921__boxed_352_; uint8_t v___x_5922__boxed_353_; uint8_t v_a_5923__boxed_354_; lean_object* v_res_355_; 
v___x_5917__boxed_351_ = lean_unbox(v___x_338_);
v___x_5921__boxed_352_ = lean_unbox(v___x_342_);
v___x_5922__boxed_353_ = lean_unbox(v___x_344_);
v_a_5923__boxed_354_ = lean_unbox(v_a_345_);
v_res_355_ = lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__1(v___x_337_, v___x_5917__boxed_351_, v___x_339_, v___x_340_, v___x_341_, v___x_5921__boxed_352_, v_e_343_, v___x_5922__boxed_353_, v_a_5923__boxed_354_, v___y_346_, v___y_347_, v___y_348_, v___y_349_);
lean_dec(v___y_349_);
lean_dec_ref(v___y_348_);
lean_dec(v___y_347_);
return v_res_355_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__1_spec__1(lean_object* v_msgData_356_, lean_object* v___y_357_, lean_object* v___y_358_, lean_object* v___y_359_, lean_object* v___y_360_){
_start:
{
lean_object* v___x_362_; lean_object* v_env_363_; lean_object* v___x_364_; lean_object* v_mctx_365_; lean_object* v_lctx_366_; lean_object* v_options_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; 
v___x_362_ = lean_st_ref_get(v___y_360_);
v_env_363_ = lean_ctor_get(v___x_362_, 0);
lean_inc_ref(v_env_363_);
lean_dec(v___x_362_);
v___x_364_ = lean_st_ref_get(v___y_358_);
v_mctx_365_ = lean_ctor_get(v___x_364_, 0);
lean_inc_ref(v_mctx_365_);
lean_dec(v___x_364_);
v_lctx_366_ = lean_ctor_get(v___y_357_, 2);
v_options_367_ = lean_ctor_get(v___y_359_, 2);
lean_inc_ref(v_options_367_);
lean_inc_ref(v_lctx_366_);
v___x_368_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_368_, 0, v_env_363_);
lean_ctor_set(v___x_368_, 1, v_mctx_365_);
lean_ctor_set(v___x_368_, 2, v_lctx_366_);
lean_ctor_set(v___x_368_, 3, v_options_367_);
v___x_369_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_369_, 0, v___x_368_);
lean_ctor_set(v___x_369_, 1, v_msgData_356_);
v___x_370_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_370_, 0, v___x_369_);
return v___x_370_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__1_spec__1___boxed(lean_object* v_msgData_371_, lean_object* v___y_372_, lean_object* v___y_373_, lean_object* v___y_374_, lean_object* v___y_375_, lean_object* v___y_376_){
_start:
{
lean_object* v_res_377_; 
v_res_377_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__1_spec__1(v_msgData_371_, v___y_372_, v___y_373_, v___y_374_, v___y_375_);
lean_dec(v___y_375_);
lean_dec_ref(v___y_374_);
lean_dec(v___y_373_);
lean_dec_ref(v___y_372_);
return v_res_377_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__1___redArg(lean_object* v_msg_378_, lean_object* v___y_379_, lean_object* v___y_380_, lean_object* v___y_381_, lean_object* v___y_382_){
_start:
{
lean_object* v_ref_384_; lean_object* v___x_385_; lean_object* v_a_386_; lean_object* v___x_388_; uint8_t v_isShared_389_; uint8_t v_isSharedCheck_394_; 
v_ref_384_ = lean_ctor_get(v___y_381_, 5);
v___x_385_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__1_spec__1(v_msg_378_, v___y_379_, v___y_380_, v___y_381_, v___y_382_);
v_a_386_ = lean_ctor_get(v___x_385_, 0);
v_isSharedCheck_394_ = !lean_is_exclusive(v___x_385_);
if (v_isSharedCheck_394_ == 0)
{
v___x_388_ = v___x_385_;
v_isShared_389_ = v_isSharedCheck_394_;
goto v_resetjp_387_;
}
else
{
lean_inc(v_a_386_);
lean_dec(v___x_385_);
v___x_388_ = lean_box(0);
v_isShared_389_ = v_isSharedCheck_394_;
goto v_resetjp_387_;
}
v_resetjp_387_:
{
lean_object* v___x_390_; lean_object* v___x_392_; 
lean_inc(v_ref_384_);
v___x_390_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_390_, 0, v_ref_384_);
lean_ctor_set(v___x_390_, 1, v_a_386_);
if (v_isShared_389_ == 0)
{
lean_ctor_set_tag(v___x_388_, 1);
lean_ctor_set(v___x_388_, 0, v___x_390_);
v___x_392_ = v___x_388_;
goto v_reusejp_391_;
}
else
{
lean_object* v_reuseFailAlloc_393_; 
v_reuseFailAlloc_393_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_393_, 0, v___x_390_);
v___x_392_ = v_reuseFailAlloc_393_;
goto v_reusejp_391_;
}
v_reusejp_391_:
{
return v___x_392_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__1___redArg___boxed(lean_object* v_msg_395_, lean_object* v___y_396_, lean_object* v___y_397_, lean_object* v___y_398_, lean_object* v___y_399_, lean_object* v___y_400_){
_start:
{
lean_object* v_res_401_; 
v_res_401_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__1___redArg(v_msg_395_, v___y_396_, v___y_397_, v___y_398_, v___y_399_);
lean_dec(v___y_399_);
lean_dec_ref(v___y_398_);
lean_dec(v___y_397_);
lean_dec_ref(v___y_396_);
return v_res_401_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__1(void){
_start:
{
lean_object* v___x_403_; lean_object* v___x_404_; 
v___x_403_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__0));
v___x_404_ = l_Lean_stringToMessageData(v___x_403_);
return v___x_404_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__3(void){
_start:
{
lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_409_; 
v___x_407_ = lean_box(0);
v___x_408_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__2));
v___x_409_ = l_Lean_Expr_const___override(v___x_408_, v___x_407_);
return v___x_409_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__4(void){
_start:
{
lean_object* v___x_410_; lean_object* v___x_411_; 
v___x_410_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__3, &lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__3_once, _init_lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__3);
v___x_411_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_411_, 0, v___x_410_);
return v___x_411_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__5(void){
_start:
{
lean_object* v___x_412_; lean_object* v___x_413_; 
v___x_412_ = lean_unsigned_to_nat(0u);
v___x_413_ = l_Lean_Level_ofNat(v___x_412_);
return v___x_413_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__13(void){
_start:
{
lean_object* v___x_426_; lean_object* v___x_427_; 
v___x_426_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__12));
v___x_427_ = l_Lean_Expr_lit___override(v___x_426_);
return v___x_427_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__17(void){
_start:
{
lean_object* v___x_433_; lean_object* v___x_434_; lean_object* v___x_435_; 
v___x_433_ = lean_box(0);
v___x_434_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__16));
v___x_435_ = l_Lean_Expr_const___override(v___x_434_, v___x_433_);
return v___x_435_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__20(void){
_start:
{
lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; 
v___x_439_ = lean_box(0);
v___x_440_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__19));
v___x_441_ = l_Lean_Expr_const___override(v___x_440_, v___x_439_);
return v___x_441_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__23(void){
_start:
{
lean_object* v___x_445_; lean_object* v___x_446_; lean_object* v___x_447_; 
v___x_445_ = lean_box(0);
v___x_446_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__22));
v___x_447_ = l_Lean_Expr_const___override(v___x_446_, v___x_445_);
return v___x_447_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__24(void){
_start:
{
lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_450_; 
v___x_448_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__13, &lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__13_once, _init_lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__13);
v___x_449_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__23, &lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__23_once, _init_lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__23);
v___x_450_ = l_Lean_Expr_app___override(v___x_449_, v___x_448_);
return v___x_450_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__25(void){
_start:
{
lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; 
v___x_451_ = lean_box(0);
v___x_452_ = ((lean_object*)(lp_mathlib_Nat___aux__Mathlib__Data__Nat__Totient______macroRules__Nat__term_u03c6__1___closed__3));
v___x_453_ = l_Lean_Expr_const___override(v___x_452_, v___x_451_);
return v___x_453_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__28(void){
_start:
{
lean_object* v___x_458_; lean_object* v___x_459_; lean_object* v___x_460_; 
v___x_458_ = lean_box(0);
v___x_459_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__27));
v___x_460_ = l_Lean_Expr_const___override(v___x_459_, v___x_458_);
return v___x_460_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__30(void){
_start:
{
lean_object* v___x_462_; lean_object* v___x_463_; 
v___x_462_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__29));
v___x_463_ = l_Lean_stringToMessageData(v___x_462_);
return v___x_463_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2(lean_object* v_u_464_, lean_object* v_00_u03b1_465_, lean_object* v_z_466_, lean_object* v_p_467_, lean_object* v_e_468_, lean_object* v___y_469_, lean_object* v___y_470_, lean_object* v___y_471_, lean_object* v___y_472_){
_start:
{
if (lean_obj_tag(v_p_467_) == 0)
{
lean_object* v___x_474_; lean_object* v___x_475_; 
lean_dec_ref(v_e_468_);
lean_dec_ref(v_z_466_);
lean_dec_ref(v_00_u03b1_465_);
lean_dec(v_u_464_);
v___x_474_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_474_, 0, v_p_467_);
v___x_475_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_475_, 0, v___x_474_);
return v___x_475_;
}
else
{
lean_object* v_val_476_; lean_object* v___x_477_; 
v_val_476_ = lean_ctor_get(v_p_467_, 0);
lean_inc(v_val_476_);
v___x_477_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__1, &lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__1_once, _init_lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__1);
if (lean_obj_tag(v_u_464_) == 0)
{
lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v___x_480_; uint8_t v___x_481_; lean_object* v___x_482_; lean_object* v___f_483_; uint8_t v___x_484_; lean_object* v___x_485_; 
v___x_478_ = ((lean_object*)(lp_mathlib_Nat_term_u03c6___closed__0));
v___x_479_ = lean_box(0);
v___x_480_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__3, &lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__3_once, _init_lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__3);
v___x_481_ = 2;
v___x_482_ = lean_box(v___x_481_);
lean_inc_ref(v_00_u03b1_465_);
v___f_483_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__0___boxed), 8, 3);
lean_closure_set(v___f_483_, 0, v___x_482_);
lean_closure_set(v___f_483_, 1, v___x_480_);
lean_closure_set(v___f_483_, 2, v_00_u03b1_465_);
v___x_484_ = 0;
v___x_485_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__0___redArg(v___f_483_, v___x_484_, v___y_469_, v___y_470_, v___y_471_, v___y_472_);
if (lean_obj_tag(v___x_485_) == 0)
{
lean_object* v_a_486_; uint8_t v___x_487_; 
v_a_486_ = lean_ctor_get(v___x_485_, 0);
lean_inc(v_a_486_);
lean_dec_ref_known(v___x_485_, 1);
v___x_487_ = lean_unbox(v_a_486_);
if (v___x_487_ == 0)
{
lean_object* v___x_488_; 
lean_dec(v_a_486_);
lean_dec_ref_known(v_p_467_, 1);
lean_dec(v_val_476_);
lean_dec_ref(v_e_468_);
lean_dec_ref(v_z_466_);
lean_dec_ref(v_00_u03b1_465_);
v___x_488_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__1___redArg(v___x_477_, v___y_469_, v___y_470_, v___y_471_, v___y_472_);
return v___x_488_;
}
else
{
lean_object* v___x_489_; uint8_t v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v___x_494_; lean_object* v___f_495_; lean_object* v___x_496_; 
v___x_489_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__4, &lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__4_once, _init_lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__4);
v___x_490_ = 0;
v___x_491_ = lean_box(0);
v___x_492_ = lean_box(v___x_490_);
v___x_493_ = lean_box(v___x_481_);
v___x_494_ = lean_box(v___x_484_);
v___f_495_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__1___boxed), 14, 9);
lean_closure_set(v___f_495_, 0, v___x_489_);
lean_closure_set(v___f_495_, 1, v___x_492_);
lean_closure_set(v___f_495_, 2, v___x_491_);
lean_closure_set(v___f_495_, 3, v___x_478_);
lean_closure_set(v___f_495_, 4, v___x_479_);
lean_closure_set(v___f_495_, 5, v___x_493_);
lean_closure_set(v___f_495_, 6, v_e_468_);
lean_closure_set(v___f_495_, 7, v___x_494_);
lean_closure_set(v___f_495_, 8, v_a_486_);
v___x_496_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__0___redArg(v___f_495_, v___x_484_, v___y_469_, v___y_470_, v___y_471_, v___y_472_);
if (lean_obj_tag(v___x_496_) == 0)
{
lean_object* v_a_497_; lean_object* v_snd_498_; uint8_t v___x_499_; 
v_a_497_ = lean_ctor_get(v___x_496_, 0);
lean_inc(v_a_497_);
lean_dec_ref_known(v___x_496_, 1);
v_snd_498_ = lean_ctor_get(v_a_497_, 1);
v___x_499_ = lean_unbox(v_snd_498_);
if (v___x_499_ == 0)
{
lean_object* v___x_500_; 
lean_dec(v_a_497_);
lean_dec(v_val_476_);
lean_dec_ref_known(v_p_467_, 1);
lean_dec_ref(v_z_466_);
lean_dec_ref(v_00_u03b1_465_);
v___x_500_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__1___redArg(v___x_477_, v___y_469_, v___y_470_, v___y_471_, v___y_472_);
return v___x_500_;
}
else
{
lean_object* v_fst_501_; lean_object* v___x_503_; uint8_t v_isShared_504_; uint8_t v_isSharedCheck_553_; 
v_fst_501_ = lean_ctor_get(v_a_497_, 0);
v_isSharedCheck_553_ = !lean_is_exclusive(v_a_497_);
if (v_isSharedCheck_553_ == 0)
{
lean_object* v_unused_554_; 
v_unused_554_ = lean_ctor_get(v_a_497_, 1);
lean_dec(v_unused_554_);
v___x_503_ = v_a_497_;
v_isShared_504_ = v_isSharedCheck_553_;
goto v_resetjp_502_;
}
else
{
lean_inc(v_fst_501_);
lean_dec(v_a_497_);
v___x_503_ = lean_box(0);
v_isShared_504_ = v_isSharedCheck_553_;
goto v_resetjp_502_;
}
v_resetjp_502_:
{
lean_object* v___x_505_; lean_object* v___x_506_; 
v___x_505_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__5, &lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__5_once, _init_lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__5);
lean_inc(v_fst_501_);
v___x_506_ = lp_mathlib_Mathlib_Meta_Positivity_core(v___x_505_, v_00_u03b1_465_, v_z_466_, v_p_467_, v_fst_501_, v___y_469_, v___y_470_, v___y_471_, v___y_472_);
if (lean_obj_tag(v___x_506_) == 0)
{
lean_object* v_a_507_; lean_object* v___x_509_; uint8_t v_isShared_510_; uint8_t v_isSharedCheck_552_; 
v_a_507_ = lean_ctor_get(v___x_506_, 0);
v_isSharedCheck_552_ = !lean_is_exclusive(v___x_506_);
if (v_isSharedCheck_552_ == 0)
{
v___x_509_ = v___x_506_;
v_isShared_510_ = v_isSharedCheck_552_;
goto v_resetjp_508_;
}
else
{
lean_inc(v_a_507_);
lean_dec(v___x_506_);
v___x_509_ = lean_box(0);
v_isShared_510_ = v_isSharedCheck_552_;
goto v_resetjp_508_;
}
v_resetjp_508_:
{
if (lean_obj_tag(v_a_507_) == 0)
{
lean_object* v_pf_511_; lean_object* v___x_513_; uint8_t v_isShared_514_; uint8_t v_isSharedCheck_548_; 
v_pf_511_ = lean_ctor_get(v_a_507_, 1);
v_isSharedCheck_548_ = !lean_is_exclusive(v_a_507_);
if (v_isSharedCheck_548_ == 0)
{
lean_object* v_unused_549_; 
v_unused_549_ = lean_ctor_get(v_a_507_, 0);
lean_dec(v_unused_549_);
v___x_513_ = v_a_507_;
v_isShared_514_ = v_isSharedCheck_548_;
goto v_resetjp_512_;
}
else
{
lean_inc(v_pf_511_);
lean_dec(v_a_507_);
v___x_513_ = lean_box(0);
v_isShared_514_ = v_isSharedCheck_548_;
goto v_resetjp_512_;
}
v_resetjp_512_:
{
lean_object* v___x_515_; lean_object* v___x_516_; lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_520_; 
v___x_515_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__8));
v___x_516_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__11));
v___x_517_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__13, &lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__13_once, _init_lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__13);
v___x_518_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__17, &lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__17_once, _init_lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__17);
if (v_isShared_504_ == 0)
{
lean_ctor_set_tag(v___x_503_, 1);
lean_ctor_set(v___x_503_, 1, v___x_479_);
lean_ctor_set(v___x_503_, 0, v_u_464_);
v___x_520_ = v___x_503_;
goto v_reusejp_519_;
}
else
{
lean_object* v_reuseFailAlloc_547_; 
v_reuseFailAlloc_547_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_547_, 0, v_u_464_);
lean_ctor_set(v_reuseFailAlloc_547_, 1, v___x_479_);
v___x_520_ = v_reuseFailAlloc_547_;
goto v_reusejp_519_;
}
v_reusejp_519_:
{
lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; lean_object* v___x_526_; lean_object* v___x_527_; lean_object* v___x_528_; lean_object* v___x_529_; lean_object* v___x_530_; lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_542_; 
lean_inc_ref(v___x_520_);
v___x_521_ = l_Lean_Expr_const___override(v___x_515_, v___x_520_);
v___x_522_ = l_Lean_Expr_app___override(v___x_521_, v___x_480_);
v___x_523_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__20, &lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__20_once, _init_lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__20);
v___x_524_ = l_Lean_Expr_app___override(v___x_522_, v___x_523_);
v___x_525_ = l_Lean_Expr_const___override(v___x_516_, v___x_520_);
v___x_526_ = l_Lean_Expr_app___override(v___x_525_, v___x_480_);
v___x_527_ = l_Lean_Expr_app___override(v___x_526_, v___x_517_);
v___x_528_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__24, &lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__24_once, _init_lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__24);
v___x_529_ = l_Lean_Expr_app___override(v___x_527_, v___x_528_);
v___x_530_ = l_Lean_Expr_app___override(v___x_524_, v___x_529_);
v___x_531_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__25, &lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__25_once, _init_lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__25);
lean_inc_n(v_fst_501_, 2);
v___x_532_ = l_Lean_Expr_app___override(v___x_531_, v_fst_501_);
lean_inc_ref(v___x_530_);
v___x_533_ = l_Lean_Expr_app___override(v___x_530_, v___x_532_);
v___x_534_ = l_Lean_Expr_app___override(v___x_518_, v___x_533_);
v___x_535_ = l_Lean_Expr_app___override(v___x_530_, v_fst_501_);
v___x_536_ = l_Lean_Expr_app___override(v___x_534_, v___x_535_);
v___x_537_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__28, &lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__28_once, _init_lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__28);
v___x_538_ = l_Lean_Expr_app___override(v___x_537_, v_fst_501_);
v___x_539_ = l_Lean_Expr_app___override(v___x_536_, v___x_538_);
v___x_540_ = l_Lean_Expr_app___override(v___x_539_, v_pf_511_);
if (v_isShared_514_ == 0)
{
lean_ctor_set(v___x_513_, 1, v___x_540_);
lean_ctor_set(v___x_513_, 0, v_val_476_);
v___x_542_ = v___x_513_;
goto v_reusejp_541_;
}
else
{
lean_object* v_reuseFailAlloc_546_; 
v_reuseFailAlloc_546_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_546_, 0, v_val_476_);
lean_ctor_set(v_reuseFailAlloc_546_, 1, v___x_540_);
v___x_542_ = v_reuseFailAlloc_546_;
goto v_reusejp_541_;
}
v_reusejp_541_:
{
lean_object* v___x_544_; 
if (v_isShared_510_ == 0)
{
lean_ctor_set(v___x_509_, 0, v___x_542_);
v___x_544_ = v___x_509_;
goto v_reusejp_543_;
}
else
{
lean_object* v_reuseFailAlloc_545_; 
v_reuseFailAlloc_545_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_545_, 0, v___x_542_);
v___x_544_ = v_reuseFailAlloc_545_;
goto v_reusejp_543_;
}
v_reusejp_543_:
{
return v___x_544_;
}
}
}
}
}
else
{
lean_object* v___x_550_; lean_object* v___x_551_; 
lean_del_object(v___x_509_);
lean_dec(v_a_507_);
lean_del_object(v___x_503_);
lean_dec(v_fst_501_);
lean_dec(v_val_476_);
v___x_550_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__30, &lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__30_once, _init_lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___closed__30);
v___x_551_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__1___redArg(v___x_550_, v___y_469_, v___y_470_, v___y_471_, v___y_472_);
return v___x_551_;
}
}
}
else
{
lean_del_object(v___x_503_);
lean_dec(v_fst_501_);
lean_dec(v_val_476_);
return v___x_506_;
}
}
}
}
else
{
lean_object* v_a_555_; lean_object* v___x_557_; uint8_t v_isShared_558_; uint8_t v_isSharedCheck_562_; 
lean_dec(v_val_476_);
lean_dec_ref_known(v_p_467_, 1);
lean_dec_ref(v_z_466_);
lean_dec_ref(v_00_u03b1_465_);
v_a_555_ = lean_ctor_get(v___x_496_, 0);
v_isSharedCheck_562_ = !lean_is_exclusive(v___x_496_);
if (v_isSharedCheck_562_ == 0)
{
v___x_557_ = v___x_496_;
v_isShared_558_ = v_isSharedCheck_562_;
goto v_resetjp_556_;
}
else
{
lean_inc(v_a_555_);
lean_dec(v___x_496_);
v___x_557_ = lean_box(0);
v_isShared_558_ = v_isSharedCheck_562_;
goto v_resetjp_556_;
}
v_resetjp_556_:
{
lean_object* v___x_560_; 
if (v_isShared_558_ == 0)
{
v___x_560_ = v___x_557_;
goto v_reusejp_559_;
}
else
{
lean_object* v_reuseFailAlloc_561_; 
v_reuseFailAlloc_561_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_561_, 0, v_a_555_);
v___x_560_ = v_reuseFailAlloc_561_;
goto v_reusejp_559_;
}
v_reusejp_559_:
{
return v___x_560_;
}
}
}
}
}
else
{
lean_object* v_a_563_; lean_object* v___x_565_; uint8_t v_isShared_566_; uint8_t v_isSharedCheck_570_; 
lean_dec_ref_known(v_p_467_, 1);
lean_dec(v_val_476_);
lean_dec_ref(v_e_468_);
lean_dec_ref(v_z_466_);
lean_dec_ref(v_00_u03b1_465_);
v_a_563_ = lean_ctor_get(v___x_485_, 0);
v_isSharedCheck_570_ = !lean_is_exclusive(v___x_485_);
if (v_isSharedCheck_570_ == 0)
{
v___x_565_ = v___x_485_;
v_isShared_566_ = v_isSharedCheck_570_;
goto v_resetjp_564_;
}
else
{
lean_inc(v_a_563_);
lean_dec(v___x_485_);
v___x_565_ = lean_box(0);
v_isShared_566_ = v_isSharedCheck_570_;
goto v_resetjp_564_;
}
v_resetjp_564_:
{
lean_object* v___x_568_; 
if (v_isShared_566_ == 0)
{
v___x_568_ = v___x_565_;
goto v_reusejp_567_;
}
else
{
lean_object* v_reuseFailAlloc_569_; 
v_reuseFailAlloc_569_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_569_, 0, v_a_563_);
v___x_568_ = v_reuseFailAlloc_569_;
goto v_reusejp_567_;
}
v_reusejp_567_:
{
return v___x_568_;
}
}
}
}
else
{
lean_object* v___x_571_; 
lean_dec(v_val_476_);
lean_dec_ref_known(v_p_467_, 1);
lean_dec_ref(v_e_468_);
lean_dec_ref(v_z_466_);
lean_dec_ref(v_00_u03b1_465_);
lean_dec(v_u_464_);
v___x_571_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__1___redArg(v___x_477_, v___y_469_, v___y_470_, v___y_471_, v___y_472_);
return v___x_571_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2___boxed(lean_object* v_u_572_, lean_object* v_00_u03b1_573_, lean_object* v_z_574_, lean_object* v_p_575_, lean_object* v_e_576_, lean_object* v___y_577_, lean_object* v___y_578_, lean_object* v___y_579_, lean_object* v___y_580_, lean_object* v___y_581_){
_start:
{
lean_object* v_res_582_; 
v_res_582_ = lp_mathlib_Mathlib_Meta_Positivity_evalNatTotient___lam__2(v_u_572_, v_00_u03b1_573_, v_z_574_, v_p_575_, v_e_576_, v___y_577_, v___y_578_, v___y_579_, v___y_580_);
lean_dec(v___y_580_);
lean_dec_ref(v___y_579_);
lean_dec(v___y_578_);
lean_dec_ref(v___y_577_);
return v_res_582_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__1(lean_object* v_00_u03b1_585_, lean_object* v_msg_586_, lean_object* v___y_587_, lean_object* v___y_588_, lean_object* v___y_589_, lean_object* v___y_590_){
_start:
{
lean_object* v___x_592_; 
v___x_592_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__1___redArg(v_msg_586_, v___y_587_, v___y_588_, v___y_589_, v___y_590_);
return v___x_592_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__1___boxed(lean_object* v_00_u03b1_593_, lean_object* v_msg_594_, lean_object* v___y_595_, lean_object* v___y_596_, lean_object* v___y_597_, lean_object* v___y_598_, lean_object* v___y_599_){
_start:
{
lean_object* v_res_600_; 
v_res_600_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_Positivity_evalNatTotient_spec__1(v_00_u03b1_593_, v_msg_594_, v___y_595_, v___y_596_, v___y_597_, v___y_598_);
lean_dec(v___y_598_);
lean_dec_ref(v___y_597_);
lean_dec(v___y_596_);
lean_dec_ref(v___y_595_);
return v_res_600_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Ring_Finset(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_CharP_Two(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_AbsoluteValue_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_BigOperators_Group_LocallyFinite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_BigOperators_GroupWithZero_Finset(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Field(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Factorization_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Factorization_Induction(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Periodic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Ring(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Totient(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Ring_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_CharP_Two(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_AbsoluteValue_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_BigOperators_Group_LocallyFinite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_BigOperators_GroupWithZero_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Field(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Factorization_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Factorization_Induction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Periodic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Nat_Totient(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Ring_Finset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_CharP_Two(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_AbsoluteValue_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_BigOperators_Group_LocallyFinite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_BigOperators_GroupWithZero_Finset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Cast_Field(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Factorization_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Factorization_Induction(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Periodic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Ring(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Nat_Totient(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Ring_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_CharP_Two(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_AbsoluteValue_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_BigOperators_Group_LocallyFinite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_BigOperators_GroupWithZero_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Cast_Field(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Factorization_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Factorization_Induction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Periodic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Totient(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Nat_Totient(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Nat_Totient(builtin);
}
#ifdef __cplusplus
}
#endif
