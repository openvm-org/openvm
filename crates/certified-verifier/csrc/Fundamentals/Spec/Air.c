// Lean compiler output
// Module: Fundamentals.Spec.Air
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Field.Defs public import Mathlib.Algebra.MvPolynomial.CommRing public import Mathlib.Data.Fintype.Basic public import Mathlib.Data.Vector.Basic public import Mathlib.RingTheory.MvPolynomial.WeightedHomogeneous public import Fundamentals.Spec.Bus.Core
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
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lp_mathlib_Field_toSemifield___redArg(lean_object*);
lean_object* lp_mathlib_instMulZeroClassOfSemiring___redArg(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
lean_object* l_List_getD___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_instDistribOfSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Finsupp_prod___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finsupp_sum___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_mapTR_loop___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_get___redArg(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* l_instDecidableEqNat___boxed(lean_object*, lean_object*);
uint8_t l_Option_instDecidableEq___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_instDecidableEqList___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
uint8_t lp_mathlib_Finsupp_instDecidableEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_Multiset_ndunion___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_filter___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
extern lean_object* lp_mathlib_Nat_instAddCancelCommMonoid;
lean_object* l_nsmulRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Repr_addAppParen(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lp_mathlib_AddGroupWithOne_toAddGroup___redArg(lean_object*);
lean_object* lp_plausible_List_foldr___at___00List_sum___at___00__private_Plausible_Gen_0__Plausible_Gen_sumFst_spec__1_spec__1(lean_object*, lean_object*);
lean_object* lean_string_length(lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Option_repr___at___00Lean_Doc_Parser_instReprBlockCtxt_repr_spec__0(lean_object*, lean_object*);
lean_object* l_Std_Format_joinSep___at___00Array_repr___at___00Lean_Elab_Term_instReprElabElimInfo_repr_spec__0_spec__0(lean_object*, lean_object*);
lean_object* l_Std_Format_fill(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
extern lean_object* lp_mathlib_Nat_instMulZeroClass;
lean_object* l_List_finRange(lean_object*);
lean_object* l___private_Init_Data_List_Impl_0__List_flatMapTR_go___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_filterMapTR_go___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_RowRef_ctorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_RowRef_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_RowRef_ctorElim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_RowRef_ctorElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_RowRef_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_RowRef_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_RowRef_local_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_RowRef_local_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_RowRef_local_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_RowRef_local_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_RowRef_next_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_RowRef_next_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_RowRef_next_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_RowRef_next_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Air_RowRef_ofNat(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_RowRef_ofNat___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqRowRef(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqRowRef___boxed(lean_object*, lean_object*);
static const lean_string_object lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "Fundamentals.Air.RowRef.local"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__0_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__0_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__1 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__1_value;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "Fundamentals.Air.RowRef.next"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__2 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__2_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__2_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__3 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__3_value;
static lean_once_cell_t lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__4;
static lean_once_cell_t lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__5;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_ctorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_ctorElim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_ctorElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_isFirst_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_isFirst_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_isFirst_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_isFirst_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_isLast_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_isLast_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_isLast_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_isLast_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_isTransition_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_isTransition_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_isTransition_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_isTransition_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Air_Selector_ofNat(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_ofNat___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqSelector(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqSelector___boxed(lean_object*, lean_object*);
static const lean_string_object lp_swirl_x2dfv_Fundamentals_Air_instReprSelector_repr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "Fundamentals.Air.Selector.isFirst"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprSelector_repr___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprSelector_repr___closed__0_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Air_instReprSelector_repr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprSelector_repr___closed__0_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprSelector_repr___closed__1 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprSelector_repr___closed__1_value;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_Air_instReprSelector_repr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "Fundamentals.Air.Selector.isLast"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprSelector_repr___closed__2 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprSelector_repr___closed__2_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Air_instReprSelector_repr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprSelector_repr___closed__2_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprSelector_repr___closed__3 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprSelector_repr___closed__3_value;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_Air_instReprSelector_repr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 39, .m_capacity = 39, .m_length = 38, .m_data = "Fundamentals.Air.Selector.isTransition"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprSelector_repr___closed__4 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprSelector_repr___closed__4_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Air_instReprSelector_repr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprSelector_repr___closed__4_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprSelector_repr___closed__5 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprSelector_repr___closed__5_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprSelector_repr(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprSelector_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_swirl_x2dfv_Fundamentals_Air_instReprSelector___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2dfv_Fundamentals_Air_instReprSelector_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprSelector___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprSelector___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprSelector = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprSelector___closed__0_value;
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqTraceLayout_decEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqTraceLayout_decEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqTraceLayout(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqTraceLayout___boxed(lean_object*, lean_object*);
static const lean_string_object lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "[]"};
static const lean_object* lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__1 = (const lean_object*)&lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__1_value;
static const lean_string_object lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__2 = (const lean_object*)&lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__2_value;
static const lean_string_object lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__3 = (const lean_object*)&lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__3_value;
static const lean_ctor_object lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__3_value)}};
static const lean_object* lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__4 = (const lean_object*)&lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__4_value;
static const lean_ctor_object lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__4_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__5 = (const lean_object*)&lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__5_value;
static const lean_string_object lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__6 = (const lean_object*)&lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__6_value;
static lean_once_cell_t lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__7;
static lean_once_cell_t lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__8;
static const lean_ctor_object lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__2_value)}};
static const lean_object* lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__9 = (const lean_object*)&lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__9_value;
static const lean_ctor_object lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__6_value)}};
static const lean_object* lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__10 = (const lean_object*)&lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__10_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg(lean_object*);
static const lean_string_object lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "{ "};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__0_value;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "preprocessedWidth"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__1_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__1_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__2 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__2_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__2_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__3 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__3_value;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " := "};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__4 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__4_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__4_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__5 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__5_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__3_value),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__6 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__6_value;
static lean_once_cell_t lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__7;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "cachedMainWidths"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__8 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__8_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__8_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__9 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__9_value;
static lean_once_cell_t lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__10;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "commonMainWidth"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__11 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__11_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__11_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__12 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__12_value;
static lean_once_cell_t lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__13;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "publicValueCount"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__14 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__14_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__14_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__15 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__15_value;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " }"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__16 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__16_value;
static lean_once_cell_t lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__17;
static lean_once_cell_t lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__18;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__19 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__19_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__16_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__20 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__20_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TraceLayout_cachedMainWidth(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TraceLayout_cachedMainWidth___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_sum___at___00Fundamentals_Air_TraceLayout_totalCachedMainWidth_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_sum___at___00Fundamentals_Air_TraceLayout_totalCachedMainWidth_spec__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TraceLayout_totalCachedMainWidth(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TraceLayout_totalCachedMainWidth___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TraceLayout_committedWidth(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TraceLayout_committedWidth___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TraceLayout_singleMain(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TraceLayout_ofWidths(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TracePart_ctorIdx___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TracePart_ctorIdx___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TracePart_ctorIdx(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TracePart_ctorIdx___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TracePart_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TracePart_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TracePart_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TracePart_preprocessed_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TracePart_preprocessed_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TracePart_preprocessed_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TracePart_cachedMain_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TracePart_cachedMain_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TracePart_cachedMain_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TracePart_commonMain_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TracePart_commonMain_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TracePart_commonMain_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqTracePart_decEq___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqTracePart_decEq___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqTracePart_decEq(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqTracePart_decEq___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqTracePart___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqTracePart___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqTracePart(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqTracePart___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 40, .m_capacity = 40, .m_length = 39, .m_data = "Fundamentals.Air.TracePart.preprocessed"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___redArg___closed__1_value;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 38, .m_capacity = 38, .m_length = 37, .m_data = "Fundamentals.Air.TracePart.commonMain"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___redArg___closed__2 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___redArg___closed__2_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___redArg___closed__2_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___redArg___closed__3 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___redArg___closed__3_value;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 38, .m_capacity = 38, .m_length = 37, .m_data = "Fundamentals.Air.TracePart.cachedMain"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___redArg___closed__4 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___redArg___closed__4_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___redArg___closed__4_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___redArg___closed__5 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___redArg___closed__5_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___redArg___closed__5_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___redArg___closed__6 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___redArg___closed__6_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TracePart_width(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TracePart_width___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_preprocessed___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_preprocessed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_preprocessed___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_cachedMain___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_cachedMain(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_cachedMain___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_instDecidableEq___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_instDecidableEq___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_instDecidableEq(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_instDecidableEq___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Row_get___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Row_get(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Row_get___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_ctorIdx___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_ctorIdx___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_ctorIdx(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_ctorIdx___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_selector_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_selector_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_selector_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_cell_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_cell_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_cell_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_publicValue_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_publicValue_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_publicValue_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqVar_decEq___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqVar_decEq___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqVar_decEq(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqVar_decEq___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqVar___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqVar___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqVar(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqVar___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Expr_ofPolynomial___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Expr_ofPolynomial___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Expr_ofPolynomial(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Expr_ofPolynomial___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Expr_eval___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Expr_eval___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Expr_eval___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Expr_eval___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Expr_eval(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Expr_eval___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_finsuppSingle___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_finsuppSingle___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_finsuppSingle___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_finsuppSingle(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppAdd___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppAdd___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppAdd___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppAdd___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppAdd___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppAdd___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppAdd___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppAdd___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppAdd___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppAdd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppNeg___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppNeg___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppNeg___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppNeg___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppNeg___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppNeg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppNeg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_add___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_add___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_add___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_add(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_neg___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_neg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_neg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_monomial___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_monomial(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instZeroFinsuppAccumulator___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instZeroFinsuppAccumulator___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instZeroFinsuppAccumulator___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instZeroFinsuppAccumulator___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instZeroFinsuppAccumulator(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instZeroFinsuppAccumulator___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instAddFinsuppAccumulator___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instAddFinsuppAccumulator___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instAddFinsuppAccumulator___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instAddFinsuppAccumulator(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instAddFinsuppAccumulator___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instAddCommMonoidFinsuppAccumulator___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instAddCommMonoidFinsuppAccumulator___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instAddCommMonoidFinsuppAccumulator(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instAddCommMonoidFinsuppAccumulator___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_mul___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_mul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_mul___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_mul___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_mul___redArg___closed__0;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_mul___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_mul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_ofVariable___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_ofVariable(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_constant___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_constant___redArg___lam__0___boxed(lean_object*);
static const lean_closure_object lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_constant___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_constant___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_constant___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_constant___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_constant___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_constant___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_constant___redArg___closed__1 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_constant___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_constant___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_constant(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_traceWeight___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_traceWeight___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_traceWeight(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_traceWeight___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Trace_get___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Trace_get___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Trace_get___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Trace_get(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Trace_get___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Trace_nextRowIndex___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Trace_nextRowIndex___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Trace_nextRowIndex(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Trace_nextRowIndex___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Trace_col___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Trace_col(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Trace_col___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Trace_colNext___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Trace_colNext___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Trace_colNext(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Trace_colNext___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_publicValueCount___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_publicValueCount___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_publicValueCount(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_publicValueCount___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_get___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_get(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_get___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_EvalCtx_localRow___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_EvalCtx_localRow(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_EvalCtx_localRow___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_nextRowIndex___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_nextRowIndex___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_nextRowIndex(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_nextRowIndex___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_col___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_col(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_col___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_colNext___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_colNext___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_colNext(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_colNext___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_EvalCtx_nextRow___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_EvalCtx_nextRow(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_EvalCtx_nextRow___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_evalSelector___redArg(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_evalSelector___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_evalSelector(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_evalSelector___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_evalVar___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_evalVar(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_evalVar___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_Expr_evalAt___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_Expr_evalAt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMessage___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMessage___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMessage___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMessage(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMessage___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMultiplicity___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMultiplicity___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMultiplicity(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMultiplicity___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMessageAt___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMessageAt___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMessageAt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMessageAt___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMultiplicityAt___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMultiplicityAt___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMultiplicityAt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMultiplicityAt___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_toBusEventAt___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_toBusEventAt___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_toBusEventAt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_toBusEventAt___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_toIndexedBusEventAt___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_toIndexedBusEventAt___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_toIndexedBusEventAt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_toIndexedBusEventAt___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_eventsAt___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_eventsAt___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_eventsAt___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_eventsAt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_events___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_events___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_events___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_events___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_events___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_events(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_eventsAtForBus___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_eventsAtForBus___redArg___lam__0___boxed(lean_object*, lean_object*);
static const lean_array_object lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_eventsAtForBus___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_eventsAtForBus___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_eventsAtForBus___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_eventsAtForBus___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_eventsAtForBus(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_eventsForBus___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_eventsForBus(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_RowRef_ctorIdx(uint8_t v_x_1_){
_start:
{
if (v_x_1_ == 0)
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(0u);
return v___x_2_;
}
else
{
lean_object* v___x_3_; 
v___x_3_ = lean_unsigned_to_nat(1u);
return v___x_3_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_RowRef_ctorIdx___boxed(lean_object* v_x_4_){
_start:
{
uint8_t v_x_boxed_5_; lean_object* v_res_6_; 
v_x_boxed_5_ = lean_unbox(v_x_4_);
v_res_6_ = lp_swirl_x2dfv_Fundamentals_Air_RowRef_ctorIdx(v_x_boxed_5_);
return v_res_6_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_RowRef_ctorElim___redArg(lean_object* v_k_7_){
_start:
{
lean_inc(v_k_7_);
return v_k_7_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_RowRef_ctorElim___redArg___boxed(lean_object* v_k_8_){
_start:
{
lean_object* v_res_9_; 
v_res_9_ = lp_swirl_x2dfv_Fundamentals_Air_RowRef_ctorElim___redArg(v_k_8_);
lean_dec(v_k_8_);
return v_res_9_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_RowRef_ctorElim(lean_object* v_motive_10_, lean_object* v_ctorIdx_11_, uint8_t v_t_12_, lean_object* v_h_13_, lean_object* v_k_14_){
_start:
{
lean_inc(v_k_14_);
return v_k_14_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_RowRef_ctorElim___boxed(lean_object* v_motive_15_, lean_object* v_ctorIdx_16_, lean_object* v_t_17_, lean_object* v_h_18_, lean_object* v_k_19_){
_start:
{
uint8_t v_t_boxed_20_; lean_object* v_res_21_; 
v_t_boxed_20_ = lean_unbox(v_t_17_);
v_res_21_ = lp_swirl_x2dfv_Fundamentals_Air_RowRef_ctorElim(v_motive_15_, v_ctorIdx_16_, v_t_boxed_20_, v_h_18_, v_k_19_);
lean_dec(v_k_19_);
lean_dec(v_ctorIdx_16_);
return v_res_21_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_RowRef_local_elim___redArg(lean_object* v_local_22_){
_start:
{
lean_inc(v_local_22_);
return v_local_22_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_RowRef_local_elim___redArg___boxed(lean_object* v_local_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_swirl_x2dfv_Fundamentals_Air_RowRef_local_elim___redArg(v_local_23_);
lean_dec(v_local_23_);
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_RowRef_local_elim(lean_object* v_motive_25_, uint8_t v_t_26_, lean_object* v_h_27_, lean_object* v_local_28_){
_start:
{
lean_inc(v_local_28_);
return v_local_28_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_RowRef_local_elim___boxed(lean_object* v_motive_29_, lean_object* v_t_30_, lean_object* v_h_31_, lean_object* v_local_32_){
_start:
{
uint8_t v_t_boxed_33_; lean_object* v_res_34_; 
v_t_boxed_33_ = lean_unbox(v_t_30_);
v_res_34_ = lp_swirl_x2dfv_Fundamentals_Air_RowRef_local_elim(v_motive_29_, v_t_boxed_33_, v_h_31_, v_local_32_);
lean_dec(v_local_32_);
return v_res_34_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_RowRef_next_elim___redArg(lean_object* v_next_35_){
_start:
{
lean_inc(v_next_35_);
return v_next_35_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_RowRef_next_elim___redArg___boxed(lean_object* v_next_36_){
_start:
{
lean_object* v_res_37_; 
v_res_37_ = lp_swirl_x2dfv_Fundamentals_Air_RowRef_next_elim___redArg(v_next_36_);
lean_dec(v_next_36_);
return v_res_37_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_RowRef_next_elim(lean_object* v_motive_38_, uint8_t v_t_39_, lean_object* v_h_40_, lean_object* v_next_41_){
_start:
{
lean_inc(v_next_41_);
return v_next_41_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_RowRef_next_elim___boxed(lean_object* v_motive_42_, lean_object* v_t_43_, lean_object* v_h_44_, lean_object* v_next_45_){
_start:
{
uint8_t v_t_boxed_46_; lean_object* v_res_47_; 
v_t_boxed_46_ = lean_unbox(v_t_43_);
v_res_47_ = lp_swirl_x2dfv_Fundamentals_Air_RowRef_next_elim(v_motive_42_, v_t_boxed_46_, v_h_44_, v_next_45_);
lean_dec(v_next_45_);
return v_res_47_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Air_RowRef_ofNat(lean_object* v_n_48_){
_start:
{
lean_object* v___x_49_; uint8_t v___x_50_; 
v___x_49_ = lean_unsigned_to_nat(0u);
v___x_50_ = lean_nat_dec_le(v_n_48_, v___x_49_);
if (v___x_50_ == 0)
{
uint8_t v___x_51_; 
v___x_51_ = 1;
return v___x_51_;
}
else
{
uint8_t v___x_52_; 
v___x_52_ = 0;
return v___x_52_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_RowRef_ofNat___boxed(lean_object* v_n_53_){
_start:
{
uint8_t v_res_54_; lean_object* v_r_55_; 
v_res_54_ = lp_swirl_x2dfv_Fundamentals_Air_RowRef_ofNat(v_n_53_);
lean_dec(v_n_53_);
v_r_55_ = lean_box(v_res_54_);
return v_r_55_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqRowRef(uint8_t v_x_56_, uint8_t v_y_57_){
_start:
{
lean_object* v___x_58_; lean_object* v___x_59_; uint8_t v___x_60_; 
v___x_58_ = lp_swirl_x2dfv_Fundamentals_Air_RowRef_ctorIdx(v_x_56_);
v___x_59_ = lp_swirl_x2dfv_Fundamentals_Air_RowRef_ctorIdx(v_y_57_);
v___x_60_ = lean_nat_dec_eq(v___x_58_, v___x_59_);
lean_dec(v___x_59_);
lean_dec(v___x_58_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqRowRef___boxed(lean_object* v_x_61_, lean_object* v_y_62_){
_start:
{
uint8_t v_x_13__boxed_63_; uint8_t v_y_14__boxed_64_; uint8_t v_res_65_; lean_object* v_r_66_; 
v_x_13__boxed_63_ = lean_unbox(v_x_61_);
v_y_14__boxed_64_ = lean_unbox(v_y_62_);
v_res_65_ = lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqRowRef(v_x_13__boxed_63_, v_y_14__boxed_64_);
v_r_66_ = lean_box(v_res_65_);
return v_r_66_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__4(void){
_start:
{
lean_object* v___x_73_; lean_object* v___x_74_; 
v___x_73_ = lean_unsigned_to_nat(2u);
v___x_74_ = lean_nat_to_int(v___x_73_);
return v___x_74_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__5(void){
_start:
{
lean_object* v___x_75_; lean_object* v___x_76_; 
v___x_75_ = lean_unsigned_to_nat(1u);
v___x_76_ = lean_nat_to_int(v___x_75_);
return v___x_76_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr(uint8_t v_x_77_, lean_object* v_prec_78_){
_start:
{
lean_object* v___y_80_; lean_object* v___y_87_; 
if (v_x_77_ == 0)
{
lean_object* v___x_93_; uint8_t v___x_94_; 
v___x_93_ = lean_unsigned_to_nat(1024u);
v___x_94_ = lean_nat_dec_le(v___x_93_, v_prec_78_);
if (v___x_94_ == 0)
{
lean_object* v___x_95_; 
v___x_95_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__4, &lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__4_once, _init_lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__4);
v___y_80_ = v___x_95_;
goto v___jp_79_;
}
else
{
lean_object* v___x_96_; 
v___x_96_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__5, &lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__5_once, _init_lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__5);
v___y_80_ = v___x_96_;
goto v___jp_79_;
}
}
else
{
lean_object* v___x_97_; uint8_t v___x_98_; 
v___x_97_ = lean_unsigned_to_nat(1024u);
v___x_98_ = lean_nat_dec_le(v___x_97_, v_prec_78_);
if (v___x_98_ == 0)
{
lean_object* v___x_99_; 
v___x_99_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__4, &lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__4_once, _init_lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__4);
v___y_87_ = v___x_99_;
goto v___jp_86_;
}
else
{
lean_object* v___x_100_; 
v___x_100_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__5, &lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__5_once, _init_lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__5);
v___y_87_ = v___x_100_;
goto v___jp_86_;
}
}
v___jp_79_:
{
lean_object* v___x_81_; lean_object* v___x_82_; uint8_t v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; 
v___x_81_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__1));
lean_inc(v___y_80_);
v___x_82_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_82_, 0, v___y_80_);
lean_ctor_set(v___x_82_, 1, v___x_81_);
v___x_83_ = 0;
v___x_84_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_84_, 0, v___x_82_);
lean_ctor_set_uint8(v___x_84_, sizeof(void*)*1, v___x_83_);
v___x_85_ = l_Repr_addAppParen(v___x_84_, v_prec_78_);
return v___x_85_;
}
v___jp_86_:
{
lean_object* v___x_88_; lean_object* v___x_89_; uint8_t v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; 
v___x_88_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__3));
lean_inc(v___y_87_);
v___x_89_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_89_, 0, v___y_87_);
lean_ctor_set(v___x_89_, 1, v___x_88_);
v___x_90_ = 0;
v___x_91_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_91_, 0, v___x_89_);
lean_ctor_set_uint8(v___x_91_, sizeof(void*)*1, v___x_90_);
v___x_92_ = l_Repr_addAppParen(v___x_91_, v_prec_78_);
return v___x_92_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___boxed(lean_object* v_x_101_, lean_object* v_prec_102_){
_start:
{
uint8_t v_x_121__boxed_103_; lean_object* v_res_104_; 
v_x_121__boxed_103_ = lean_unbox(v_x_101_);
v_res_104_ = lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr(v_x_121__boxed_103_, v_prec_102_);
lean_dec(v_prec_102_);
return v_res_104_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_ctorIdx(uint8_t v_x_107_){
_start:
{
switch(v_x_107_)
{
case 0:
{
lean_object* v___x_108_; 
v___x_108_ = lean_unsigned_to_nat(0u);
return v___x_108_;
}
case 1:
{
lean_object* v___x_109_; 
v___x_109_ = lean_unsigned_to_nat(1u);
return v___x_109_;
}
default: 
{
lean_object* v___x_110_; 
v___x_110_ = lean_unsigned_to_nat(2u);
return v___x_110_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_ctorIdx___boxed(lean_object* v_x_111_){
_start:
{
uint8_t v_x_boxed_112_; lean_object* v_res_113_; 
v_x_boxed_112_ = lean_unbox(v_x_111_);
v_res_113_ = lp_swirl_x2dfv_Fundamentals_Air_Selector_ctorIdx(v_x_boxed_112_);
return v_res_113_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_ctorElim___redArg(lean_object* v_k_114_){
_start:
{
lean_inc(v_k_114_);
return v_k_114_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_ctorElim___redArg___boxed(lean_object* v_k_115_){
_start:
{
lean_object* v_res_116_; 
v_res_116_ = lp_swirl_x2dfv_Fundamentals_Air_Selector_ctorElim___redArg(v_k_115_);
lean_dec(v_k_115_);
return v_res_116_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_ctorElim(lean_object* v_motive_117_, lean_object* v_ctorIdx_118_, uint8_t v_t_119_, lean_object* v_h_120_, lean_object* v_k_121_){
_start:
{
lean_inc(v_k_121_);
return v_k_121_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_ctorElim___boxed(lean_object* v_motive_122_, lean_object* v_ctorIdx_123_, lean_object* v_t_124_, lean_object* v_h_125_, lean_object* v_k_126_){
_start:
{
uint8_t v_t_boxed_127_; lean_object* v_res_128_; 
v_t_boxed_127_ = lean_unbox(v_t_124_);
v_res_128_ = lp_swirl_x2dfv_Fundamentals_Air_Selector_ctorElim(v_motive_122_, v_ctorIdx_123_, v_t_boxed_127_, v_h_125_, v_k_126_);
lean_dec(v_k_126_);
lean_dec(v_ctorIdx_123_);
return v_res_128_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_isFirst_elim___redArg(lean_object* v_isFirst_129_){
_start:
{
lean_inc(v_isFirst_129_);
return v_isFirst_129_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_isFirst_elim___redArg___boxed(lean_object* v_isFirst_130_){
_start:
{
lean_object* v_res_131_; 
v_res_131_ = lp_swirl_x2dfv_Fundamentals_Air_Selector_isFirst_elim___redArg(v_isFirst_130_);
lean_dec(v_isFirst_130_);
return v_res_131_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_isFirst_elim(lean_object* v_motive_132_, uint8_t v_t_133_, lean_object* v_h_134_, lean_object* v_isFirst_135_){
_start:
{
lean_inc(v_isFirst_135_);
return v_isFirst_135_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_isFirst_elim___boxed(lean_object* v_motive_136_, lean_object* v_t_137_, lean_object* v_h_138_, lean_object* v_isFirst_139_){
_start:
{
uint8_t v_t_boxed_140_; lean_object* v_res_141_; 
v_t_boxed_140_ = lean_unbox(v_t_137_);
v_res_141_ = lp_swirl_x2dfv_Fundamentals_Air_Selector_isFirst_elim(v_motive_136_, v_t_boxed_140_, v_h_138_, v_isFirst_139_);
lean_dec(v_isFirst_139_);
return v_res_141_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_isLast_elim___redArg(lean_object* v_isLast_142_){
_start:
{
lean_inc(v_isLast_142_);
return v_isLast_142_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_isLast_elim___redArg___boxed(lean_object* v_isLast_143_){
_start:
{
lean_object* v_res_144_; 
v_res_144_ = lp_swirl_x2dfv_Fundamentals_Air_Selector_isLast_elim___redArg(v_isLast_143_);
lean_dec(v_isLast_143_);
return v_res_144_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_isLast_elim(lean_object* v_motive_145_, uint8_t v_t_146_, lean_object* v_h_147_, lean_object* v_isLast_148_){
_start:
{
lean_inc(v_isLast_148_);
return v_isLast_148_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_isLast_elim___boxed(lean_object* v_motive_149_, lean_object* v_t_150_, lean_object* v_h_151_, lean_object* v_isLast_152_){
_start:
{
uint8_t v_t_boxed_153_; lean_object* v_res_154_; 
v_t_boxed_153_ = lean_unbox(v_t_150_);
v_res_154_ = lp_swirl_x2dfv_Fundamentals_Air_Selector_isLast_elim(v_motive_149_, v_t_boxed_153_, v_h_151_, v_isLast_152_);
lean_dec(v_isLast_152_);
return v_res_154_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_isTransition_elim___redArg(lean_object* v_isTransition_155_){
_start:
{
lean_inc(v_isTransition_155_);
return v_isTransition_155_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_isTransition_elim___redArg___boxed(lean_object* v_isTransition_156_){
_start:
{
lean_object* v_res_157_; 
v_res_157_ = lp_swirl_x2dfv_Fundamentals_Air_Selector_isTransition_elim___redArg(v_isTransition_156_);
lean_dec(v_isTransition_156_);
return v_res_157_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_isTransition_elim(lean_object* v_motive_158_, uint8_t v_t_159_, lean_object* v_h_160_, lean_object* v_isTransition_161_){
_start:
{
lean_inc(v_isTransition_161_);
return v_isTransition_161_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_isTransition_elim___boxed(lean_object* v_motive_162_, lean_object* v_t_163_, lean_object* v_h_164_, lean_object* v_isTransition_165_){
_start:
{
uint8_t v_t_boxed_166_; lean_object* v_res_167_; 
v_t_boxed_166_ = lean_unbox(v_t_163_);
v_res_167_ = lp_swirl_x2dfv_Fundamentals_Air_Selector_isTransition_elim(v_motive_162_, v_t_boxed_166_, v_h_164_, v_isTransition_165_);
lean_dec(v_isTransition_165_);
return v_res_167_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Air_Selector_ofNat(lean_object* v_n_168_){
_start:
{
lean_object* v___x_169_; uint8_t v___x_170_; 
v___x_169_ = lean_unsigned_to_nat(0u);
v___x_170_ = lean_nat_dec_le(v_n_168_, v___x_169_);
if (v___x_170_ == 0)
{
lean_object* v___x_171_; uint8_t v___x_172_; 
v___x_171_ = lean_unsigned_to_nat(1u);
v___x_172_ = lean_nat_dec_le(v_n_168_, v___x_171_);
if (v___x_172_ == 0)
{
uint8_t v___x_173_; 
v___x_173_ = 2;
return v___x_173_;
}
else
{
uint8_t v___x_174_; 
v___x_174_ = 1;
return v___x_174_;
}
}
else
{
uint8_t v___x_175_; 
v___x_175_ = 0;
return v___x_175_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Selector_ofNat___boxed(lean_object* v_n_176_){
_start:
{
uint8_t v_res_177_; lean_object* v_r_178_; 
v_res_177_ = lp_swirl_x2dfv_Fundamentals_Air_Selector_ofNat(v_n_176_);
lean_dec(v_n_176_);
v_r_178_ = lean_box(v_res_177_);
return v_r_178_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqSelector(uint8_t v_x_179_, uint8_t v_y_180_){
_start:
{
lean_object* v___x_181_; lean_object* v___x_182_; uint8_t v___x_183_; 
v___x_181_ = lp_swirl_x2dfv_Fundamentals_Air_Selector_ctorIdx(v_x_179_);
v___x_182_ = lp_swirl_x2dfv_Fundamentals_Air_Selector_ctorIdx(v_y_180_);
v___x_183_ = lean_nat_dec_eq(v___x_181_, v___x_182_);
lean_dec(v___x_182_);
lean_dec(v___x_181_);
return v___x_183_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqSelector___boxed(lean_object* v_x_184_, lean_object* v_y_185_){
_start:
{
uint8_t v_x_13__boxed_186_; uint8_t v_y_14__boxed_187_; uint8_t v_res_188_; lean_object* v_r_189_; 
v_x_13__boxed_186_ = lean_unbox(v_x_184_);
v_y_14__boxed_187_ = lean_unbox(v_y_185_);
v_res_188_ = lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqSelector(v_x_13__boxed_186_, v_y_14__boxed_187_);
v_r_189_ = lean_box(v_res_188_);
return v_r_189_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprSelector_repr(uint8_t v_x_199_, lean_object* v_prec_200_){
_start:
{
lean_object* v___y_202_; lean_object* v___y_209_; lean_object* v___y_216_; 
switch(v_x_199_)
{
case 0:
{
lean_object* v___x_222_; uint8_t v___x_223_; 
v___x_222_ = lean_unsigned_to_nat(1024u);
v___x_223_ = lean_nat_dec_le(v___x_222_, v_prec_200_);
if (v___x_223_ == 0)
{
lean_object* v___x_224_; 
v___x_224_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__4, &lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__4_once, _init_lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__4);
v___y_202_ = v___x_224_;
goto v___jp_201_;
}
else
{
lean_object* v___x_225_; 
v___x_225_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__5, &lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__5_once, _init_lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__5);
v___y_202_ = v___x_225_;
goto v___jp_201_;
}
}
case 1:
{
lean_object* v___x_226_; uint8_t v___x_227_; 
v___x_226_ = lean_unsigned_to_nat(1024u);
v___x_227_ = lean_nat_dec_le(v___x_226_, v_prec_200_);
if (v___x_227_ == 0)
{
lean_object* v___x_228_; 
v___x_228_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__4, &lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__4_once, _init_lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__4);
v___y_209_ = v___x_228_;
goto v___jp_208_;
}
else
{
lean_object* v___x_229_; 
v___x_229_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__5, &lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__5_once, _init_lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__5);
v___y_209_ = v___x_229_;
goto v___jp_208_;
}
}
default: 
{
lean_object* v___x_230_; uint8_t v___x_231_; 
v___x_230_ = lean_unsigned_to_nat(1024u);
v___x_231_ = lean_nat_dec_le(v___x_230_, v_prec_200_);
if (v___x_231_ == 0)
{
lean_object* v___x_232_; 
v___x_232_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__4, &lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__4_once, _init_lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__4);
v___y_216_ = v___x_232_;
goto v___jp_215_;
}
else
{
lean_object* v___x_233_; 
v___x_233_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__5, &lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__5_once, _init_lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__5);
v___y_216_ = v___x_233_;
goto v___jp_215_;
}
}
}
v___jp_201_:
{
lean_object* v___x_203_; lean_object* v___x_204_; uint8_t v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; 
v___x_203_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_Air_instReprSelector_repr___closed__1));
lean_inc(v___y_202_);
v___x_204_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_204_, 0, v___y_202_);
lean_ctor_set(v___x_204_, 1, v___x_203_);
v___x_205_ = 0;
v___x_206_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_206_, 0, v___x_204_);
lean_ctor_set_uint8(v___x_206_, sizeof(void*)*1, v___x_205_);
v___x_207_ = l_Repr_addAppParen(v___x_206_, v_prec_200_);
return v___x_207_;
}
v___jp_208_:
{
lean_object* v___x_210_; lean_object* v___x_211_; uint8_t v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; 
v___x_210_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_Air_instReprSelector_repr___closed__3));
lean_inc(v___y_209_);
v___x_211_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_211_, 0, v___y_209_);
lean_ctor_set(v___x_211_, 1, v___x_210_);
v___x_212_ = 0;
v___x_213_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_213_, 0, v___x_211_);
lean_ctor_set_uint8(v___x_213_, sizeof(void*)*1, v___x_212_);
v___x_214_ = l_Repr_addAppParen(v___x_213_, v_prec_200_);
return v___x_214_;
}
v___jp_215_:
{
lean_object* v___x_217_; lean_object* v___x_218_; uint8_t v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; 
v___x_217_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_Air_instReprSelector_repr___closed__5));
lean_inc(v___y_216_);
v___x_218_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_218_, 0, v___y_216_);
lean_ctor_set(v___x_218_, 1, v___x_217_);
v___x_219_ = 0;
v___x_220_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_220_, 0, v___x_218_);
lean_ctor_set_uint8(v___x_220_, sizeof(void*)*1, v___x_219_);
v___x_221_ = l_Repr_addAppParen(v___x_220_, v_prec_200_);
return v___x_221_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprSelector_repr___boxed(lean_object* v_x_234_, lean_object* v_prec_235_){
_start:
{
uint8_t v_x_173__boxed_236_; lean_object* v_res_237_; 
v_x_173__boxed_236_ = lean_unbox(v_x_234_);
v_res_237_ = lp_swirl_x2dfv_Fundamentals_Air_instReprSelector_repr(v_x_173__boxed_236_, v_prec_235_);
lean_dec(v_prec_235_);
return v_res_237_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqTraceLayout_decEq(lean_object* v_x_240_, lean_object* v_x_241_){
_start:
{
lean_object* v_preprocessedWidth_242_; lean_object* v_cachedMainWidths_243_; lean_object* v_commonMainWidth_244_; lean_object* v_publicValueCount_245_; lean_object* v_preprocessedWidth_246_; lean_object* v_cachedMainWidths_247_; lean_object* v_commonMainWidth_248_; lean_object* v_publicValueCount_249_; lean_object* v___x_250_; uint8_t v___x_251_; 
v_preprocessedWidth_242_ = lean_ctor_get(v_x_240_, 0);
lean_inc(v_preprocessedWidth_242_);
v_cachedMainWidths_243_ = lean_ctor_get(v_x_240_, 1);
lean_inc(v_cachedMainWidths_243_);
v_commonMainWidth_244_ = lean_ctor_get(v_x_240_, 2);
lean_inc(v_commonMainWidth_244_);
v_publicValueCount_245_ = lean_ctor_get(v_x_240_, 3);
lean_inc(v_publicValueCount_245_);
lean_dec_ref(v_x_240_);
v_preprocessedWidth_246_ = lean_ctor_get(v_x_241_, 0);
lean_inc(v_preprocessedWidth_246_);
v_cachedMainWidths_247_ = lean_ctor_get(v_x_241_, 1);
lean_inc(v_cachedMainWidths_247_);
v_commonMainWidth_248_ = lean_ctor_get(v_x_241_, 2);
lean_inc(v_commonMainWidth_248_);
v_publicValueCount_249_ = lean_ctor_get(v_x_241_, 3);
lean_inc(v_publicValueCount_249_);
lean_dec_ref(v_x_241_);
v___x_250_ = lean_alloc_closure((void*)(l_instDecidableEqNat___boxed), 2, 0);
lean_inc_ref(v___x_250_);
v___x_251_ = l_Option_instDecidableEq___redArg(v___x_250_, v_preprocessedWidth_242_, v_preprocessedWidth_246_);
if (v___x_251_ == 0)
{
lean_dec_ref(v___x_250_);
lean_dec(v_publicValueCount_249_);
lean_dec(v_commonMainWidth_248_);
lean_dec(v_cachedMainWidths_247_);
lean_dec(v_publicValueCount_245_);
lean_dec(v_commonMainWidth_244_);
lean_dec(v_cachedMainWidths_243_);
return v___x_251_;
}
else
{
uint8_t v___x_252_; 
v___x_252_ = l_instDecidableEqList___redArg(v___x_250_, v_cachedMainWidths_243_, v_cachedMainWidths_247_);
if (v___x_252_ == 0)
{
lean_dec(v_publicValueCount_249_);
lean_dec(v_commonMainWidth_248_);
lean_dec(v_publicValueCount_245_);
lean_dec(v_commonMainWidth_244_);
return v___x_252_;
}
else
{
uint8_t v___x_253_; 
v___x_253_ = lean_nat_dec_eq(v_commonMainWidth_244_, v_commonMainWidth_248_);
lean_dec(v_commonMainWidth_248_);
lean_dec(v_commonMainWidth_244_);
if (v___x_253_ == 0)
{
lean_dec(v_publicValueCount_249_);
lean_dec(v_publicValueCount_245_);
return v___x_253_;
}
else
{
uint8_t v___x_254_; 
v___x_254_ = lean_nat_dec_eq(v_publicValueCount_245_, v_publicValueCount_249_);
lean_dec(v_publicValueCount_249_);
lean_dec(v_publicValueCount_245_);
return v___x_254_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqTraceLayout_decEq___boxed(lean_object* v_x_255_, lean_object* v_x_256_){
_start:
{
uint8_t v_res_257_; lean_object* v_r_258_; 
v_res_257_ = lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqTraceLayout_decEq(v_x_255_, v_x_256_);
v_r_258_ = lean_box(v_res_257_);
return v_r_258_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqTraceLayout(lean_object* v_x_259_, lean_object* v_x_260_){
_start:
{
uint8_t v___x_261_; 
v___x_261_ = lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqTraceLayout_decEq(v_x_259_, v_x_260_);
return v___x_261_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqTraceLayout___boxed(lean_object* v_x_262_, lean_object* v_x_263_){
_start:
{
uint8_t v_res_264_; lean_object* v_r_265_; 
v_res_264_ = lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqTraceLayout(v_x_262_, v_x_263_);
v_r_265_ = lean_box(v_res_264_);
return v_r_265_;
}
}
static lean_object* _init_lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__7(void){
_start:
{
lean_object* v___x_277_; lean_object* v___x_278_; 
v___x_277_ = ((lean_object*)(lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__2));
v___x_278_ = lean_string_length(v___x_277_);
return v___x_278_;
}
}
static lean_object* _init_lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__8(void){
_start:
{
lean_object* v___x_279_; lean_object* v___x_280_; 
v___x_279_ = lean_obj_once(&lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__7, &lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__7_once, _init_lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__7);
v___x_280_ = lean_nat_to_int(v___x_279_);
return v___x_280_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg(lean_object* v_a_285_){
_start:
{
if (lean_obj_tag(v_a_285_) == 0)
{
lean_object* v___x_286_; 
v___x_286_ = ((lean_object*)(lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__1));
return v___x_286_;
}
else
{
lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; 
v___x_287_ = ((lean_object*)(lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__5));
v___x_288_ = l_Std_Format_joinSep___at___00Array_repr___at___00Lean_Elab_Term_instReprElabElimInfo_repr_spec__0_spec__0(v_a_285_, v___x_287_);
v___x_289_ = lean_obj_once(&lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__8, &lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__8_once, _init_lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__8);
v___x_290_ = ((lean_object*)(lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__9));
v___x_291_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_291_, 0, v___x_290_);
lean_ctor_set(v___x_291_, 1, v___x_288_);
v___x_292_ = ((lean_object*)(lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__10));
v___x_293_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_293_, 0, v___x_291_);
lean_ctor_set(v___x_293_, 1, v___x_292_);
v___x_294_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_294_, 0, v___x_289_);
lean_ctor_set(v___x_294_, 1, v___x_293_);
v___x_295_ = l_Std_Format_fill(v___x_294_);
return v___x_295_;
}
}
}
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__7(void){
_start:
{
lean_object* v___x_309_; lean_object* v___x_310_; 
v___x_309_ = lean_unsigned_to_nat(21u);
v___x_310_ = lean_nat_to_int(v___x_309_);
return v___x_310_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__10(void){
_start:
{
lean_object* v___x_314_; lean_object* v___x_315_; 
v___x_314_ = lean_unsigned_to_nat(20u);
v___x_315_ = lean_nat_to_int(v___x_314_);
return v___x_315_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__13(void){
_start:
{
lean_object* v___x_319_; lean_object* v___x_320_; 
v___x_319_ = lean_unsigned_to_nat(19u);
v___x_320_ = lean_nat_to_int(v___x_319_);
return v___x_320_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__17(void){
_start:
{
lean_object* v___x_325_; lean_object* v___x_326_; 
v___x_325_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__0));
v___x_326_ = lean_string_length(v___x_325_);
return v___x_326_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__18(void){
_start:
{
lean_object* v___x_327_; lean_object* v___x_328_; 
v___x_327_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__17, &lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__17_once, _init_lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__17);
v___x_328_ = lean_nat_to_int(v___x_327_);
return v___x_328_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg(lean_object* v_x_333_){
_start:
{
lean_object* v_preprocessedWidth_334_; lean_object* v_cachedMainWidths_335_; lean_object* v_commonMainWidth_336_; lean_object* v_publicValueCount_337_; lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v___x_340_; lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; uint8_t v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___x_353_; lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; lean_object* v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; lean_object* v___x_364_; lean_object* v___x_365_; lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v___x_386_; 
v_preprocessedWidth_334_ = lean_ctor_get(v_x_333_, 0);
lean_inc(v_preprocessedWidth_334_);
v_cachedMainWidths_335_ = lean_ctor_get(v_x_333_, 1);
lean_inc(v_cachedMainWidths_335_);
v_commonMainWidth_336_ = lean_ctor_get(v_x_333_, 2);
lean_inc(v_commonMainWidth_336_);
v_publicValueCount_337_ = lean_ctor_get(v_x_333_, 3);
lean_inc(v_publicValueCount_337_);
lean_dec_ref(v_x_333_);
v___x_338_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__5));
v___x_339_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__6));
v___x_340_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__7, &lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__7_once, _init_lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__7);
v___x_341_ = lean_unsigned_to_nat(0u);
v___x_342_ = l_Option_repr___at___00Lean_Doc_Parser_instReprBlockCtxt_repr_spec__0(v_preprocessedWidth_334_, v___x_341_);
v___x_343_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_343_, 0, v___x_340_);
lean_ctor_set(v___x_343_, 1, v___x_342_);
v___x_344_ = 0;
v___x_345_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_345_, 0, v___x_343_);
lean_ctor_set_uint8(v___x_345_, sizeof(void*)*1, v___x_344_);
v___x_346_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_346_, 0, v___x_339_);
lean_ctor_set(v___x_346_, 1, v___x_345_);
v___x_347_ = ((lean_object*)(lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg___closed__4));
v___x_348_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_348_, 0, v___x_346_);
lean_ctor_set(v___x_348_, 1, v___x_347_);
v___x_349_ = lean_box(1);
v___x_350_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_350_, 0, v___x_348_);
lean_ctor_set(v___x_350_, 1, v___x_349_);
v___x_351_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__9));
v___x_352_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_352_, 0, v___x_350_);
lean_ctor_set(v___x_352_, 1, v___x_351_);
v___x_353_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_353_, 0, v___x_352_);
lean_ctor_set(v___x_353_, 1, v___x_338_);
v___x_354_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__10, &lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__10_once, _init_lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__10);
v___x_355_ = lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg(v_cachedMainWidths_335_);
v___x_356_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_356_, 0, v___x_354_);
lean_ctor_set(v___x_356_, 1, v___x_355_);
v___x_357_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_357_, 0, v___x_356_);
lean_ctor_set_uint8(v___x_357_, sizeof(void*)*1, v___x_344_);
v___x_358_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_358_, 0, v___x_353_);
lean_ctor_set(v___x_358_, 1, v___x_357_);
v___x_359_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_359_, 0, v___x_358_);
lean_ctor_set(v___x_359_, 1, v___x_347_);
v___x_360_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_360_, 0, v___x_359_);
lean_ctor_set(v___x_360_, 1, v___x_349_);
v___x_361_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__12));
v___x_362_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_362_, 0, v___x_360_);
lean_ctor_set(v___x_362_, 1, v___x_361_);
v___x_363_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_363_, 0, v___x_362_);
lean_ctor_set(v___x_363_, 1, v___x_338_);
v___x_364_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__13, &lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__13_once, _init_lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__13);
v___x_365_ = l_Nat_reprFast(v_commonMainWidth_336_);
v___x_366_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_366_, 0, v___x_365_);
v___x_367_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_367_, 0, v___x_364_);
lean_ctor_set(v___x_367_, 1, v___x_366_);
v___x_368_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_368_, 0, v___x_367_);
lean_ctor_set_uint8(v___x_368_, sizeof(void*)*1, v___x_344_);
v___x_369_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_369_, 0, v___x_363_);
lean_ctor_set(v___x_369_, 1, v___x_368_);
v___x_370_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_370_, 0, v___x_369_);
lean_ctor_set(v___x_370_, 1, v___x_347_);
v___x_371_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_371_, 0, v___x_370_);
lean_ctor_set(v___x_371_, 1, v___x_349_);
v___x_372_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__15));
v___x_373_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_373_, 0, v___x_371_);
lean_ctor_set(v___x_373_, 1, v___x_372_);
v___x_374_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_374_, 0, v___x_373_);
lean_ctor_set(v___x_374_, 1, v___x_338_);
v___x_375_ = l_Nat_reprFast(v_publicValueCount_337_);
v___x_376_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_376_, 0, v___x_375_);
v___x_377_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_377_, 0, v___x_354_);
lean_ctor_set(v___x_377_, 1, v___x_376_);
v___x_378_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_378_, 0, v___x_377_);
lean_ctor_set_uint8(v___x_378_, sizeof(void*)*1, v___x_344_);
v___x_379_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_379_, 0, v___x_374_);
lean_ctor_set(v___x_379_, 1, v___x_378_);
v___x_380_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__18, &lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__18_once, _init_lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__18);
v___x_381_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__19));
v___x_382_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_382_, 0, v___x_381_);
lean_ctor_set(v___x_382_, 1, v___x_379_);
v___x_383_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg___closed__20));
v___x_384_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_384_, 0, v___x_382_);
lean_ctor_set(v___x_384_, 1, v___x_383_);
v___x_385_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_385_, 0, v___x_380_);
lean_ctor_set(v___x_385_, 1, v___x_384_);
v___x_386_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_386_, 0, v___x_385_);
lean_ctor_set_uint8(v___x_386_, sizeof(void*)*1, v___x_344_);
return v___x_386_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr(lean_object* v_x_387_, lean_object* v_prec_388_){
_start:
{
lean_object* v___x_389_; 
v___x_389_ = lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___redArg(v_x_387_);
return v___x_389_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr___boxed(lean_object* v_x_390_, lean_object* v_prec_391_){
_start:
{
lean_object* v_res_392_; 
v_res_392_ = lp_swirl_x2dfv_Fundamentals_Air_instReprTraceLayout_repr(v_x_390_, v_prec_391_);
lean_dec(v_prec_391_);
return v_res_392_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0(lean_object* v_a_393_, lean_object* v_n_394_){
_start:
{
lean_object* v___x_395_; 
v___x_395_ = lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___redArg(v_a_393_);
return v___x_395_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0___boxed(lean_object* v_a_396_, lean_object* v_n_397_){
_start:
{
lean_object* v_res_398_; 
v_res_398_ = lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_Air_instReprTraceLayout_repr_spec__0(v_a_396_, v_n_397_);
lean_dec(v_n_397_);
return v_res_398_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TraceLayout_cachedMainWidth(lean_object* v_layout_401_, lean_object* v_group_402_){
_start:
{
lean_object* v_cachedMainWidths_403_; lean_object* v___x_404_; 
v_cachedMainWidths_403_ = lean_ctor_get(v_layout_401_, 1);
v___x_404_ = l_List_get___redArg(v_cachedMainWidths_403_, v_group_402_);
return v___x_404_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TraceLayout_cachedMainWidth___boxed(lean_object* v_layout_405_, lean_object* v_group_406_){
_start:
{
lean_object* v_res_407_; 
v_res_407_ = lp_swirl_x2dfv_Fundamentals_Air_TraceLayout_cachedMainWidth(v_layout_405_, v_group_406_);
lean_dec_ref(v_layout_405_);
return v_res_407_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_sum___at___00Fundamentals_Air_TraceLayout_totalCachedMainWidth_spec__0(lean_object* v_l_408_){
_start:
{
lean_object* v___x_409_; lean_object* v___x_410_; 
v___x_409_ = lean_unsigned_to_nat(0u);
v___x_410_ = lp_plausible_List_foldr___at___00List_sum___at___00__private_Plausible_Gen_0__Plausible_Gen_sumFst_spec__1_spec__1(v___x_409_, v_l_408_);
return v___x_410_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_sum___at___00Fundamentals_Air_TraceLayout_totalCachedMainWidth_spec__0___boxed(lean_object* v_l_411_){
_start:
{
lean_object* v_res_412_; 
v_res_412_ = lp_swirl_x2dfv_List_sum___at___00Fundamentals_Air_TraceLayout_totalCachedMainWidth_spec__0(v_l_411_);
lean_dec(v_l_411_);
return v_res_412_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TraceLayout_totalCachedMainWidth(lean_object* v_layout_413_){
_start:
{
lean_object* v_cachedMainWidths_414_; lean_object* v___x_415_; 
v_cachedMainWidths_414_ = lean_ctor_get(v_layout_413_, 1);
v___x_415_ = lp_swirl_x2dfv_List_sum___at___00Fundamentals_Air_TraceLayout_totalCachedMainWidth_spec__0(v_cachedMainWidths_414_);
return v___x_415_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TraceLayout_totalCachedMainWidth___boxed(lean_object* v_layout_416_){
_start:
{
lean_object* v_res_417_; 
v_res_417_ = lp_swirl_x2dfv_Fundamentals_Air_TraceLayout_totalCachedMainWidth(v_layout_416_);
lean_dec_ref(v_layout_416_);
return v_res_417_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TraceLayout_committedWidth(lean_object* v_layout_418_){
_start:
{
lean_object* v_preprocessedWidth_419_; lean_object* v_commonMainWidth_420_; lean_object* v___y_422_; 
v_preprocessedWidth_419_ = lean_ctor_get(v_layout_418_, 0);
v_commonMainWidth_420_ = lean_ctor_get(v_layout_418_, 2);
if (lean_obj_tag(v_preprocessedWidth_419_) == 0)
{
lean_object* v___x_426_; 
v___x_426_ = lean_unsigned_to_nat(0u);
v___y_422_ = v___x_426_;
goto v___jp_421_;
}
else
{
lean_object* v_val_427_; 
v_val_427_ = lean_ctor_get(v_preprocessedWidth_419_, 0);
v___y_422_ = v_val_427_;
goto v___jp_421_;
}
v___jp_421_:
{
lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___x_425_; 
v___x_423_ = lean_nat_add(v___y_422_, v_commonMainWidth_420_);
v___x_424_ = lp_swirl_x2dfv_Fundamentals_Air_TraceLayout_totalCachedMainWidth(v_layout_418_);
v___x_425_ = lean_nat_add(v___x_423_, v___x_424_);
lean_dec(v___x_424_);
lean_dec(v___x_423_);
return v___x_425_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TraceLayout_committedWidth___boxed(lean_object* v_layout_428_){
_start:
{
lean_object* v_res_429_; 
v_res_429_ = lp_swirl_x2dfv_Fundamentals_Air_TraceLayout_committedWidth(v_layout_428_);
lean_dec_ref(v_layout_428_);
return v_res_429_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TraceLayout_singleMain(lean_object* v_width_430_, lean_object* v_publicValueCount_431_){
_start:
{
lean_object* v___x_432_; lean_object* v___x_433_; lean_object* v___x_434_; 
v___x_432_ = lean_box(0);
v___x_433_ = lean_box(0);
v___x_434_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_434_, 0, v___x_432_);
lean_ctor_set(v___x_434_, 1, v___x_433_);
lean_ctor_set(v___x_434_, 2, v_width_430_);
lean_ctor_set(v___x_434_, 3, v_publicValueCount_431_);
return v___x_434_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TraceLayout_ofWidths(lean_object* v_preprocessedWidth_435_, lean_object* v_cachedMainWidths_436_, lean_object* v_commonMainWidth_437_, lean_object* v_publicValueCount_438_){
_start:
{
lean_object* v___x_439_; 
v___x_439_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_439_, 0, v_preprocessedWidth_435_);
lean_ctor_set(v___x_439_, 1, v_cachedMainWidths_436_);
lean_ctor_set(v___x_439_, 2, v_commonMainWidth_437_);
lean_ctor_set(v___x_439_, 3, v_publicValueCount_438_);
return v___x_439_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TracePart_ctorIdx___redArg(lean_object* v_x_440_){
_start:
{
switch(lean_obj_tag(v_x_440_))
{
case 0:
{
lean_object* v___x_441_; 
v___x_441_ = lean_unsigned_to_nat(0u);
return v___x_441_;
}
case 1:
{
lean_object* v___x_442_; 
v___x_442_ = lean_unsigned_to_nat(1u);
return v___x_442_;
}
default: 
{
lean_object* v___x_443_; 
v___x_443_ = lean_unsigned_to_nat(2u);
return v___x_443_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TracePart_ctorIdx___redArg___boxed(lean_object* v_x_444_){
_start:
{
lean_object* v_res_445_; 
v_res_445_ = lp_swirl_x2dfv_Fundamentals_Air_TracePart_ctorIdx___redArg(v_x_444_);
lean_dec(v_x_444_);
return v_res_445_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TracePart_ctorIdx(lean_object* v_layout_446_, lean_object* v_x_447_){
_start:
{
lean_object* v___x_448_; 
v___x_448_ = lp_swirl_x2dfv_Fundamentals_Air_TracePart_ctorIdx___redArg(v_x_447_);
return v___x_448_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TracePart_ctorIdx___boxed(lean_object* v_layout_449_, lean_object* v_x_450_){
_start:
{
lean_object* v_res_451_; 
v_res_451_ = lp_swirl_x2dfv_Fundamentals_Air_TracePart_ctorIdx(v_layout_449_, v_x_450_);
lean_dec(v_x_450_);
lean_dec_ref(v_layout_449_);
return v_res_451_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TracePart_ctorElim___redArg(lean_object* v_t_452_, lean_object* v_k_453_){
_start:
{
if (lean_obj_tag(v_t_452_) == 1)
{
lean_object* v_group_454_; lean_object* v___x_455_; 
v_group_454_ = lean_ctor_get(v_t_452_, 0);
lean_inc(v_group_454_);
lean_dec_ref_known(v_t_452_, 1);
v___x_455_ = lean_apply_1(v_k_453_, v_group_454_);
return v___x_455_;
}
else
{
lean_dec(v_t_452_);
return v_k_453_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TracePart_ctorElim(lean_object* v_layout_456_, lean_object* v_motive_457_, lean_object* v_ctorIdx_458_, lean_object* v_t_459_, lean_object* v_h_460_, lean_object* v_k_461_){
_start:
{
lean_object* v___x_462_; 
v___x_462_ = lp_swirl_x2dfv_Fundamentals_Air_TracePart_ctorElim___redArg(v_t_459_, v_k_461_);
return v___x_462_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TracePart_ctorElim___boxed(lean_object* v_layout_463_, lean_object* v_motive_464_, lean_object* v_ctorIdx_465_, lean_object* v_t_466_, lean_object* v_h_467_, lean_object* v_k_468_){
_start:
{
lean_object* v_res_469_; 
v_res_469_ = lp_swirl_x2dfv_Fundamentals_Air_TracePart_ctorElim(v_layout_463_, v_motive_464_, v_ctorIdx_465_, v_t_466_, v_h_467_, v_k_468_);
lean_dec(v_ctorIdx_465_);
lean_dec_ref(v_layout_463_);
return v_res_469_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TracePart_preprocessed_elim___redArg(lean_object* v_t_470_, lean_object* v_preprocessed_471_){
_start:
{
lean_object* v___x_472_; 
v___x_472_ = lp_swirl_x2dfv_Fundamentals_Air_TracePart_ctorElim___redArg(v_t_470_, v_preprocessed_471_);
return v___x_472_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TracePart_preprocessed_elim(lean_object* v_layout_473_, lean_object* v_motive_474_, lean_object* v_t_475_, lean_object* v_h_476_, lean_object* v_preprocessed_477_){
_start:
{
lean_object* v___x_478_; 
v___x_478_ = lp_swirl_x2dfv_Fundamentals_Air_TracePart_ctorElim___redArg(v_t_475_, v_preprocessed_477_);
return v___x_478_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TracePart_preprocessed_elim___boxed(lean_object* v_layout_479_, lean_object* v_motive_480_, lean_object* v_t_481_, lean_object* v_h_482_, lean_object* v_preprocessed_483_){
_start:
{
lean_object* v_res_484_; 
v_res_484_ = lp_swirl_x2dfv_Fundamentals_Air_TracePart_preprocessed_elim(v_layout_479_, v_motive_480_, v_t_481_, v_h_482_, v_preprocessed_483_);
lean_dec_ref(v_layout_479_);
return v_res_484_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TracePart_cachedMain_elim___redArg(lean_object* v_t_485_, lean_object* v_cachedMain_486_){
_start:
{
lean_object* v___x_487_; 
v___x_487_ = lp_swirl_x2dfv_Fundamentals_Air_TracePart_ctorElim___redArg(v_t_485_, v_cachedMain_486_);
return v___x_487_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TracePart_cachedMain_elim(lean_object* v_layout_488_, lean_object* v_motive_489_, lean_object* v_t_490_, lean_object* v_h_491_, lean_object* v_cachedMain_492_){
_start:
{
lean_object* v___x_493_; 
v___x_493_ = lp_swirl_x2dfv_Fundamentals_Air_TracePart_ctorElim___redArg(v_t_490_, v_cachedMain_492_);
return v___x_493_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TracePart_cachedMain_elim___boxed(lean_object* v_layout_494_, lean_object* v_motive_495_, lean_object* v_t_496_, lean_object* v_h_497_, lean_object* v_cachedMain_498_){
_start:
{
lean_object* v_res_499_; 
v_res_499_ = lp_swirl_x2dfv_Fundamentals_Air_TracePart_cachedMain_elim(v_layout_494_, v_motive_495_, v_t_496_, v_h_497_, v_cachedMain_498_);
lean_dec_ref(v_layout_494_);
return v_res_499_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TracePart_commonMain_elim___redArg(lean_object* v_t_500_, lean_object* v_commonMain_501_){
_start:
{
lean_object* v___x_502_; 
v___x_502_ = lp_swirl_x2dfv_Fundamentals_Air_TracePart_ctorElim___redArg(v_t_500_, v_commonMain_501_);
return v___x_502_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TracePart_commonMain_elim(lean_object* v_layout_503_, lean_object* v_motive_504_, lean_object* v_t_505_, lean_object* v_h_506_, lean_object* v_commonMain_507_){
_start:
{
lean_object* v___x_508_; 
v___x_508_ = lp_swirl_x2dfv_Fundamentals_Air_TracePart_ctorElim___redArg(v_t_505_, v_commonMain_507_);
return v___x_508_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TracePart_commonMain_elim___boxed(lean_object* v_layout_509_, lean_object* v_motive_510_, lean_object* v_t_511_, lean_object* v_h_512_, lean_object* v_commonMain_513_){
_start:
{
lean_object* v_res_514_; 
v_res_514_ = lp_swirl_x2dfv_Fundamentals_Air_TracePart_commonMain_elim(v_layout_509_, v_motive_510_, v_t_511_, v_h_512_, v_commonMain_513_);
lean_dec_ref(v_layout_509_);
return v_res_514_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqTracePart_decEq___redArg(lean_object* v_x_515_, lean_object* v_x_516_){
_start:
{
switch(lean_obj_tag(v_x_515_))
{
case 0:
{
if (lean_obj_tag(v_x_516_) == 0)
{
uint8_t v___x_517_; 
v___x_517_ = 1;
return v___x_517_;
}
else
{
uint8_t v___x_518_; 
v___x_518_ = 0;
return v___x_518_;
}
}
case 1:
{
lean_object* v_group_519_; uint8_t v___x_520_; 
v_group_519_ = lean_ctor_get(v_x_515_, 0);
v___x_520_ = 0;
if (lean_obj_tag(v_x_516_) == 1)
{
lean_object* v_group_521_; uint8_t v___x_522_; 
v_group_521_ = lean_ctor_get(v_x_516_, 0);
v___x_522_ = lean_nat_dec_eq(v_group_519_, v_group_521_);
if (v___x_522_ == 0)
{
return v___x_520_;
}
else
{
return v___x_522_;
}
}
else
{
return v___x_520_;
}
}
default: 
{
if (lean_obj_tag(v_x_516_) == 2)
{
uint8_t v___x_523_; 
v___x_523_ = 1;
return v___x_523_;
}
else
{
uint8_t v___x_524_; 
v___x_524_ = 0;
return v___x_524_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqTracePart_decEq___redArg___boxed(lean_object* v_x_525_, lean_object* v_x_526_){
_start:
{
uint8_t v_res_527_; lean_object* v_r_528_; 
v_res_527_ = lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqTracePart_decEq___redArg(v_x_525_, v_x_526_);
lean_dec(v_x_526_);
lean_dec(v_x_525_);
v_r_528_ = lean_box(v_res_527_);
return v_r_528_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqTracePart_decEq(lean_object* v_layout_529_, lean_object* v_x_530_, lean_object* v_x_531_){
_start:
{
uint8_t v___x_532_; 
v___x_532_ = lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqTracePart_decEq___redArg(v_x_530_, v_x_531_);
return v___x_532_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqTracePart_decEq___boxed(lean_object* v_layout_533_, lean_object* v_x_534_, lean_object* v_x_535_){
_start:
{
uint8_t v_res_536_; lean_object* v_r_537_; 
v_res_536_ = lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqTracePart_decEq(v_layout_533_, v_x_534_, v_x_535_);
lean_dec(v_x_535_);
lean_dec(v_x_534_);
lean_dec_ref(v_layout_533_);
v_r_537_ = lean_box(v_res_536_);
return v_r_537_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqTracePart___redArg(lean_object* v_x_538_, lean_object* v_x_539_){
_start:
{
uint8_t v___x_540_; 
v___x_540_ = lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqTracePart_decEq___redArg(v_x_538_, v_x_539_);
return v___x_540_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqTracePart___redArg___boxed(lean_object* v_x_541_, lean_object* v_x_542_){
_start:
{
uint8_t v_res_543_; lean_object* v_r_544_; 
v_res_543_ = lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqTracePart___redArg(v_x_541_, v_x_542_);
lean_dec(v_x_542_);
lean_dec(v_x_541_);
v_r_544_ = lean_box(v_res_543_);
return v_r_544_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqTracePart(lean_object* v_layout_545_, lean_object* v_x_546_, lean_object* v_x_547_){
_start:
{
uint8_t v___x_548_; 
v___x_548_ = lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqTracePart_decEq___redArg(v_x_546_, v_x_547_);
return v___x_548_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqTracePart___boxed(lean_object* v_layout_549_, lean_object* v_x_550_, lean_object* v_x_551_){
_start:
{
uint8_t v_res_552_; lean_object* v_r_553_; 
v_res_552_ = lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqTracePart(v_layout_549_, v_x_550_, v_x_551_);
lean_dec(v_x_551_);
lean_dec(v_x_550_);
lean_dec_ref(v_layout_549_);
v_r_553_ = lean_box(v_res_552_);
return v_r_553_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___redArg(lean_object* v_x_566_, lean_object* v_prec_567_){
_start:
{
lean_object* v___y_569_; lean_object* v___y_576_; 
switch(lean_obj_tag(v_x_566_))
{
case 0:
{
lean_object* v___x_582_; uint8_t v___x_583_; 
v___x_582_ = lean_unsigned_to_nat(1024u);
v___x_583_ = lean_nat_dec_le(v___x_582_, v_prec_567_);
if (v___x_583_ == 0)
{
lean_object* v___x_584_; 
v___x_584_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__4, &lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__4_once, _init_lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__4);
v___y_569_ = v___x_584_;
goto v___jp_568_;
}
else
{
lean_object* v___x_585_; 
v___x_585_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__5, &lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__5_once, _init_lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__5);
v___y_569_ = v___x_585_;
goto v___jp_568_;
}
}
case 1:
{
lean_object* v_group_586_; lean_object* v___x_588_; uint8_t v_isShared_589_; uint8_t v_isSharedCheck_606_; 
v_group_586_ = lean_ctor_get(v_x_566_, 0);
v_isSharedCheck_606_ = !lean_is_exclusive(v_x_566_);
if (v_isSharedCheck_606_ == 0)
{
v___x_588_ = v_x_566_;
v_isShared_589_ = v_isSharedCheck_606_;
goto v_resetjp_587_;
}
else
{
lean_inc(v_group_586_);
lean_dec(v_x_566_);
v___x_588_ = lean_box(0);
v_isShared_589_ = v_isSharedCheck_606_;
goto v_resetjp_587_;
}
v_resetjp_587_:
{
lean_object* v___y_591_; lean_object* v___x_602_; uint8_t v___x_603_; 
v___x_602_ = lean_unsigned_to_nat(1024u);
v___x_603_ = lean_nat_dec_le(v___x_602_, v_prec_567_);
if (v___x_603_ == 0)
{
lean_object* v___x_604_; 
v___x_604_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__4, &lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__4_once, _init_lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__4);
v___y_591_ = v___x_604_;
goto v___jp_590_;
}
else
{
lean_object* v___x_605_; 
v___x_605_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__5, &lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__5_once, _init_lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__5);
v___y_591_ = v___x_605_;
goto v___jp_590_;
}
v___jp_590_:
{
lean_object* v___x_592_; lean_object* v___x_593_; lean_object* v___x_595_; 
v___x_592_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___redArg___closed__6));
v___x_593_ = l_Nat_reprFast(v_group_586_);
if (v_isShared_589_ == 0)
{
lean_ctor_set_tag(v___x_588_, 3);
lean_ctor_set(v___x_588_, 0, v___x_593_);
v___x_595_ = v___x_588_;
goto v_reusejp_594_;
}
else
{
lean_object* v_reuseFailAlloc_601_; 
v_reuseFailAlloc_601_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_601_, 0, v___x_593_);
v___x_595_ = v_reuseFailAlloc_601_;
goto v_reusejp_594_;
}
v_reusejp_594_:
{
lean_object* v___x_596_; lean_object* v___x_597_; uint8_t v___x_598_; lean_object* v___x_599_; lean_object* v___x_600_; 
v___x_596_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_596_, 0, v___x_592_);
lean_ctor_set(v___x_596_, 1, v___x_595_);
lean_inc(v___y_591_);
v___x_597_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_597_, 0, v___y_591_);
lean_ctor_set(v___x_597_, 1, v___x_596_);
v___x_598_ = 0;
v___x_599_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_599_, 0, v___x_597_);
lean_ctor_set_uint8(v___x_599_, sizeof(void*)*1, v___x_598_);
v___x_600_ = l_Repr_addAppParen(v___x_599_, v_prec_567_);
return v___x_600_;
}
}
}
}
default: 
{
lean_object* v___x_607_; uint8_t v___x_608_; 
v___x_607_ = lean_unsigned_to_nat(1024u);
v___x_608_ = lean_nat_dec_le(v___x_607_, v_prec_567_);
if (v___x_608_ == 0)
{
lean_object* v___x_609_; 
v___x_609_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__4, &lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__4_once, _init_lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__4);
v___y_576_ = v___x_609_;
goto v___jp_575_;
}
else
{
lean_object* v___x_610_; 
v___x_610_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__5, &lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__5_once, _init_lp_swirl_x2dfv_Fundamentals_Air_instReprRowRef_repr___closed__5);
v___y_576_ = v___x_610_;
goto v___jp_575_;
}
}
}
v___jp_568_:
{
lean_object* v___x_570_; lean_object* v___x_571_; uint8_t v___x_572_; lean_object* v___x_573_; lean_object* v___x_574_; 
v___x_570_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___redArg___closed__1));
lean_inc(v___y_569_);
v___x_571_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_571_, 0, v___y_569_);
lean_ctor_set(v___x_571_, 1, v___x_570_);
v___x_572_ = 0;
v___x_573_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_573_, 0, v___x_571_);
lean_ctor_set_uint8(v___x_573_, sizeof(void*)*1, v___x_572_);
v___x_574_ = l_Repr_addAppParen(v___x_573_, v_prec_567_);
return v___x_574_;
}
v___jp_575_:
{
lean_object* v___x_577_; lean_object* v___x_578_; uint8_t v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; 
v___x_577_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___redArg___closed__3));
lean_inc(v___y_576_);
v___x_578_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_578_, 0, v___y_576_);
lean_ctor_set(v___x_578_, 1, v___x_577_);
v___x_579_ = 0;
v___x_580_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_580_, 0, v___x_578_);
lean_ctor_set_uint8(v___x_580_, sizeof(void*)*1, v___x_579_);
v___x_581_ = l_Repr_addAppParen(v___x_580_, v_prec_567_);
return v___x_581_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___redArg___boxed(lean_object* v_x_611_, lean_object* v_prec_612_){
_start:
{
lean_object* v_res_613_; 
v_res_613_ = lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___redArg(v_x_611_, v_prec_612_);
lean_dec(v_prec_612_);
return v_res_613_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr(lean_object* v_layout_614_, lean_object* v_x_615_, lean_object* v_prec_616_){
_start:
{
lean_object* v___x_617_; 
v___x_617_ = lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___redArg(v_x_615_, v_prec_616_);
return v___x_617_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___boxed(lean_object* v_layout_618_, lean_object* v_x_619_, lean_object* v_prec_620_){
_start:
{
lean_object* v_res_621_; 
v_res_621_ = lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr(v_layout_618_, v_x_619_, v_prec_620_);
lean_dec(v_prec_620_);
lean_dec_ref(v_layout_618_);
return v_res_621_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart(lean_object* v_layout_622_){
_start:
{
lean_object* v___x_623_; 
v___x_623_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Air_instReprTracePart_repr___boxed), 3, 1);
lean_closure_set(v___x_623_, 0, v_layout_622_);
return v___x_623_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TracePart_width(lean_object* v_layout_624_, lean_object* v_x_625_){
_start:
{
switch(lean_obj_tag(v_x_625_))
{
case 0:
{
lean_object* v_preprocessedWidth_626_; 
v_preprocessedWidth_626_ = lean_ctor_get(v_layout_624_, 0);
if (lean_obj_tag(v_preprocessedWidth_626_) == 0)
{
lean_object* v___x_627_; 
v___x_627_ = lean_unsigned_to_nat(0u);
return v___x_627_;
}
else
{
lean_object* v_val_628_; 
v_val_628_ = lean_ctor_get(v_preprocessedWidth_626_, 0);
lean_inc(v_val_628_);
return v_val_628_;
}
}
case 1:
{
lean_object* v_group_629_; lean_object* v_cachedMainWidths_630_; lean_object* v___x_631_; 
v_group_629_ = lean_ctor_get(v_x_625_, 0);
lean_inc(v_group_629_);
lean_dec_ref_known(v_x_625_, 1);
v_cachedMainWidths_630_ = lean_ctor_get(v_layout_624_, 1);
v___x_631_ = l_List_get___redArg(v_cachedMainWidths_630_, v_group_629_);
return v___x_631_;
}
default: 
{
lean_object* v_commonMainWidth_632_; 
v_commonMainWidth_632_ = lean_ctor_get(v_layout_624_, 2);
lean_inc(v_commonMainWidth_632_);
return v_commonMainWidth_632_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_TracePart_width___boxed(lean_object* v_layout_633_, lean_object* v_x_634_){
_start:
{
lean_object* v_res_635_; 
v_res_635_ = lp_swirl_x2dfv_Fundamentals_Air_TracePart_width(v_layout_633_, v_x_634_);
lean_dec_ref(v_layout_633_);
return v_res_635_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_preprocessed___redArg(lean_object* v_colIdx_636_){
_start:
{
lean_object* v___x_637_; lean_object* v___x_638_; 
v___x_637_ = lean_box(0);
v___x_638_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_638_, 0, v___x_637_);
lean_ctor_set(v___x_638_, 1, v_colIdx_636_);
return v___x_638_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_preprocessed(lean_object* v_layout_639_, lean_object* v_colIdx_640_){
_start:
{
lean_object* v___x_641_; 
v___x_641_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_preprocessed___redArg(v_colIdx_640_);
return v___x_641_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_preprocessed___boxed(lean_object* v_layout_642_, lean_object* v_colIdx_643_){
_start:
{
lean_object* v_res_644_; 
v_res_644_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_preprocessed(v_layout_642_, v_colIdx_643_);
lean_dec_ref(v_layout_642_);
return v_res_644_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_cachedMain___redArg(lean_object* v_group_645_, lean_object* v_colIdx_646_){
_start:
{
lean_object* v___x_647_; lean_object* v___x_648_; 
v___x_647_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_647_, 0, v_group_645_);
v___x_648_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_648_, 0, v___x_647_);
lean_ctor_set(v___x_648_, 1, v_colIdx_646_);
return v___x_648_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_cachedMain(lean_object* v_layout_649_, lean_object* v_group_650_, lean_object* v_colIdx_651_){
_start:
{
lean_object* v___x_652_; 
v___x_652_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_cachedMain___redArg(v_group_650_, v_colIdx_651_);
return v___x_652_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_cachedMain___boxed(lean_object* v_layout_653_, lean_object* v_group_654_, lean_object* v_colIdx_655_){
_start:
{
lean_object* v_res_656_; 
v_res_656_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_cachedMain(v_layout_653_, v_group_654_, v_colIdx_655_);
lean_dec_ref(v_layout_653_);
return v_res_656_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___redArg(lean_object* v_colIdx_657_){
_start:
{
lean_object* v___x_658_; lean_object* v___x_659_; 
v___x_658_ = lean_box(2);
v___x_659_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_659_, 0, v___x_658_);
lean_ctor_set(v___x_659_, 1, v_colIdx_657_);
return v___x_659_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain(lean_object* v_layout_660_, lean_object* v_colIdx_661_){
_start:
{
lean_object* v___x_662_; 
v___x_662_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___redArg(v_colIdx_661_);
return v___x_662_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___boxed(lean_object* v_layout_663_, lean_object* v_colIdx_664_){
_start:
{
lean_object* v_res_665_; 
v_res_665_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain(v_layout_663_, v_colIdx_664_);
lean_dec_ref(v_layout_663_);
return v_res_665_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_instDecidableEq___redArg(lean_object* v_x_666_, lean_object* v_x_667_){
_start:
{
lean_object* v_part_668_; lean_object* v_colIdx_669_; lean_object* v_part_670_; lean_object* v_colIdx_671_; uint8_t v___x_672_; 
v_part_668_ = lean_ctor_get(v_x_666_, 0);
v_colIdx_669_ = lean_ctor_get(v_x_666_, 1);
v_part_670_ = lean_ctor_get(v_x_667_, 0);
v_colIdx_671_ = lean_ctor_get(v_x_667_, 1);
v___x_672_ = lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqTracePart_decEq___redArg(v_part_668_, v_part_670_);
if (v___x_672_ == 0)
{
return v___x_672_;
}
else
{
uint8_t v___x_673_; 
v___x_673_ = lean_nat_dec_eq(v_colIdx_669_, v_colIdx_671_);
return v___x_673_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_instDecidableEq___redArg___boxed(lean_object* v_x_674_, lean_object* v_x_675_){
_start:
{
uint8_t v_res_676_; lean_object* v_r_677_; 
v_res_676_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_instDecidableEq___redArg(v_x_674_, v_x_675_);
lean_dec_ref(v_x_675_);
lean_dec_ref(v_x_674_);
v_r_677_ = lean_box(v_res_676_);
return v_r_677_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_instDecidableEq(lean_object* v_layout_678_, lean_object* v_x_679_, lean_object* v_x_680_){
_start:
{
uint8_t v___x_681_; 
v___x_681_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_instDecidableEq___redArg(v_x_679_, v_x_680_);
return v___x_681_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_instDecidableEq___boxed(lean_object* v_layout_682_, lean_object* v_x_683_, lean_object* v_x_684_){
_start:
{
uint8_t v_res_685_; lean_object* v_r_686_; 
v_res_685_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_instDecidableEq(v_layout_682_, v_x_683_, v_x_684_);
lean_dec_ref(v_x_684_);
lean_dec_ref(v_x_683_);
lean_dec_ref(v_layout_682_);
v_r_686_ = lean_box(v_res_685_);
return v_r_686_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Row_get___redArg(lean_object* v_row_687_, lean_object* v_column_688_){
_start:
{
lean_object* v_part_689_; lean_object* v_colIdx_690_; lean_object* v___x_691_; lean_object* v___x_692_; 
v_part_689_ = lean_ctor_get(v_column_688_, 0);
lean_inc(v_part_689_);
v_colIdx_690_ = lean_ctor_get(v_column_688_, 1);
lean_inc(v_colIdx_690_);
lean_dec_ref(v_column_688_);
v___x_691_ = lean_apply_1(v_row_687_, v_part_689_);
v___x_692_ = lean_array_fget(v___x_691_, v_colIdx_690_);
lean_dec(v_colIdx_690_);
lean_dec_ref(v___x_691_);
return v___x_692_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Row_get(lean_object* v_F_693_, lean_object* v_layout_694_, lean_object* v_row_695_, lean_object* v_column_696_){
_start:
{
lean_object* v___x_697_; 
v___x_697_ = lp_swirl_x2dfv_Fundamentals_Air_Row_get___redArg(v_row_695_, v_column_696_);
return v___x_697_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Row_get___boxed(lean_object* v_F_698_, lean_object* v_layout_699_, lean_object* v_row_700_, lean_object* v_column_701_){
_start:
{
lean_object* v_res_702_; 
v_res_702_ = lp_swirl_x2dfv_Fundamentals_Air_Row_get(v_F_698_, v_layout_699_, v_row_700_, v_column_701_);
lean_dec_ref(v_layout_699_);
return v_res_702_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_ctorIdx___redArg(lean_object* v_x_703_){
_start:
{
switch(lean_obj_tag(v_x_703_))
{
case 0:
{
lean_object* v___x_704_; 
v___x_704_ = lean_unsigned_to_nat(0u);
return v___x_704_;
}
case 1:
{
lean_object* v___x_705_; 
v___x_705_ = lean_unsigned_to_nat(1u);
return v___x_705_;
}
default: 
{
lean_object* v___x_706_; 
v___x_706_ = lean_unsigned_to_nat(2u);
return v___x_706_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_ctorIdx___redArg___boxed(lean_object* v_x_707_){
_start:
{
lean_object* v_res_708_; 
v_res_708_ = lp_swirl_x2dfv_Fundamentals_Air_Var_ctorIdx___redArg(v_x_707_);
lean_dec_ref(v_x_707_);
return v_res_708_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_ctorIdx(lean_object* v_layout_709_, lean_object* v_x_710_){
_start:
{
lean_object* v___x_711_; 
v___x_711_ = lp_swirl_x2dfv_Fundamentals_Air_Var_ctorIdx___redArg(v_x_710_);
return v___x_711_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_ctorIdx___boxed(lean_object* v_layout_712_, lean_object* v_x_713_){
_start:
{
lean_object* v_res_714_; 
v_res_714_ = lp_swirl_x2dfv_Fundamentals_Air_Var_ctorIdx(v_layout_712_, v_x_713_);
lean_dec_ref(v_x_713_);
lean_dec_ref(v_layout_712_);
return v_res_714_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_ctorElim___redArg(lean_object* v_t_715_, lean_object* v_k_716_){
_start:
{
switch(lean_obj_tag(v_t_715_))
{
case 0:
{
uint8_t v_a_717_; lean_object* v___x_718_; lean_object* v___x_719_; 
v_a_717_ = lean_ctor_get_uint8(v_t_715_, 0);
lean_dec_ref_known(v_t_715_, 0);
v___x_718_ = lean_box(v_a_717_);
v___x_719_ = lean_apply_1(v_k_716_, v___x_718_);
return v___x_719_;
}
case 1:
{
uint8_t v_a_720_; lean_object* v_a_721_; lean_object* v___x_722_; lean_object* v___x_723_; 
v_a_720_ = lean_ctor_get_uint8(v_t_715_, sizeof(void*)*1);
v_a_721_ = lean_ctor_get(v_t_715_, 0);
lean_inc_ref(v_a_721_);
lean_dec_ref_known(v_t_715_, 1);
v___x_722_ = lean_box(v_a_720_);
v___x_723_ = lean_apply_2(v_k_716_, v___x_722_, v_a_721_);
return v___x_723_;
}
default: 
{
lean_object* v_a_724_; lean_object* v___x_725_; 
v_a_724_ = lean_ctor_get(v_t_715_, 0);
lean_inc(v_a_724_);
lean_dec_ref_known(v_t_715_, 1);
v___x_725_ = lean_apply_1(v_k_716_, v_a_724_);
return v___x_725_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_ctorElim(lean_object* v_layout_726_, lean_object* v_motive_727_, lean_object* v_ctorIdx_728_, lean_object* v_t_729_, lean_object* v_h_730_, lean_object* v_k_731_){
_start:
{
lean_object* v___x_732_; 
v___x_732_ = lp_swirl_x2dfv_Fundamentals_Air_Var_ctorElim___redArg(v_t_729_, v_k_731_);
return v___x_732_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_ctorElim___boxed(lean_object* v_layout_733_, lean_object* v_motive_734_, lean_object* v_ctorIdx_735_, lean_object* v_t_736_, lean_object* v_h_737_, lean_object* v_k_738_){
_start:
{
lean_object* v_res_739_; 
v_res_739_ = lp_swirl_x2dfv_Fundamentals_Air_Var_ctorElim(v_layout_733_, v_motive_734_, v_ctorIdx_735_, v_t_736_, v_h_737_, v_k_738_);
lean_dec(v_ctorIdx_735_);
lean_dec_ref(v_layout_733_);
return v_res_739_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_selector_elim___redArg(lean_object* v_t_740_, lean_object* v_selector_741_){
_start:
{
lean_object* v___x_742_; 
v___x_742_ = lp_swirl_x2dfv_Fundamentals_Air_Var_ctorElim___redArg(v_t_740_, v_selector_741_);
return v___x_742_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_selector_elim(lean_object* v_layout_743_, lean_object* v_motive_744_, lean_object* v_t_745_, lean_object* v_h_746_, lean_object* v_selector_747_){
_start:
{
lean_object* v___x_748_; 
v___x_748_ = lp_swirl_x2dfv_Fundamentals_Air_Var_ctorElim___redArg(v_t_745_, v_selector_747_);
return v___x_748_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_selector_elim___boxed(lean_object* v_layout_749_, lean_object* v_motive_750_, lean_object* v_t_751_, lean_object* v_h_752_, lean_object* v_selector_753_){
_start:
{
lean_object* v_res_754_; 
v_res_754_ = lp_swirl_x2dfv_Fundamentals_Air_Var_selector_elim(v_layout_749_, v_motive_750_, v_t_751_, v_h_752_, v_selector_753_);
lean_dec_ref(v_layout_749_);
return v_res_754_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_cell_elim___redArg(lean_object* v_t_755_, lean_object* v_cell_756_){
_start:
{
lean_object* v___x_757_; 
v___x_757_ = lp_swirl_x2dfv_Fundamentals_Air_Var_ctorElim___redArg(v_t_755_, v_cell_756_);
return v___x_757_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_cell_elim(lean_object* v_layout_758_, lean_object* v_motive_759_, lean_object* v_t_760_, lean_object* v_h_761_, lean_object* v_cell_762_){
_start:
{
lean_object* v___x_763_; 
v___x_763_ = lp_swirl_x2dfv_Fundamentals_Air_Var_ctorElim___redArg(v_t_760_, v_cell_762_);
return v___x_763_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_cell_elim___boxed(lean_object* v_layout_764_, lean_object* v_motive_765_, lean_object* v_t_766_, lean_object* v_h_767_, lean_object* v_cell_768_){
_start:
{
lean_object* v_res_769_; 
v_res_769_ = lp_swirl_x2dfv_Fundamentals_Air_Var_cell_elim(v_layout_764_, v_motive_765_, v_t_766_, v_h_767_, v_cell_768_);
lean_dec_ref(v_layout_764_);
return v_res_769_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_publicValue_elim___redArg(lean_object* v_t_770_, lean_object* v_publicValue_771_){
_start:
{
lean_object* v___x_772_; 
v___x_772_ = lp_swirl_x2dfv_Fundamentals_Air_Var_ctorElim___redArg(v_t_770_, v_publicValue_771_);
return v___x_772_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_publicValue_elim(lean_object* v_layout_773_, lean_object* v_motive_774_, lean_object* v_t_775_, lean_object* v_h_776_, lean_object* v_publicValue_777_){
_start:
{
lean_object* v___x_778_; 
v___x_778_ = lp_swirl_x2dfv_Fundamentals_Air_Var_ctorElim___redArg(v_t_775_, v_publicValue_777_);
return v___x_778_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_publicValue_elim___boxed(lean_object* v_layout_779_, lean_object* v_motive_780_, lean_object* v_t_781_, lean_object* v_h_782_, lean_object* v_publicValue_783_){
_start:
{
lean_object* v_res_784_; 
v_res_784_ = lp_swirl_x2dfv_Fundamentals_Air_Var_publicValue_elim(v_layout_779_, v_motive_780_, v_t_781_, v_h_782_, v_publicValue_783_);
lean_dec_ref(v_layout_779_);
return v_res_784_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqVar_decEq___redArg(lean_object* v_x_785_, lean_object* v_x_786_){
_start:
{
switch(lean_obj_tag(v_x_785_))
{
case 0:
{
if (lean_obj_tag(v_x_786_) == 0)
{
uint8_t v_a_787_; uint8_t v_a_788_; uint8_t v___x_789_; 
v_a_787_ = lean_ctor_get_uint8(v_x_785_, 0);
v_a_788_ = lean_ctor_get_uint8(v_x_786_, 0);
v___x_789_ = lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqSelector(v_a_787_, v_a_788_);
return v___x_789_;
}
else
{
uint8_t v___x_790_; 
v___x_790_ = 0;
return v___x_790_;
}
}
case 1:
{
if (lean_obj_tag(v_x_786_) == 1)
{
uint8_t v_a_791_; lean_object* v_a_792_; uint8_t v_a_793_; lean_object* v_a_794_; uint8_t v___x_795_; 
v_a_791_ = lean_ctor_get_uint8(v_x_785_, sizeof(void*)*1);
v_a_792_ = lean_ctor_get(v_x_785_, 0);
v_a_793_ = lean_ctor_get_uint8(v_x_786_, sizeof(void*)*1);
v_a_794_ = lean_ctor_get(v_x_786_, 0);
v___x_795_ = lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqRowRef(v_a_791_, v_a_793_);
if (v___x_795_ == 0)
{
return v___x_795_;
}
else
{
uint8_t v___x_796_; 
v___x_796_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_instDecidableEq___redArg(v_a_792_, v_a_794_);
return v___x_796_;
}
}
else
{
uint8_t v___x_797_; 
v___x_797_ = 0;
return v___x_797_;
}
}
default: 
{
if (lean_obj_tag(v_x_786_) == 2)
{
lean_object* v_a_798_; lean_object* v_a_799_; uint8_t v___x_800_; 
v_a_798_ = lean_ctor_get(v_x_785_, 0);
v_a_799_ = lean_ctor_get(v_x_786_, 0);
v___x_800_ = lean_nat_dec_eq(v_a_798_, v_a_799_);
return v___x_800_;
}
else
{
uint8_t v___x_801_; 
v___x_801_ = 0;
return v___x_801_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqVar_decEq___redArg___boxed(lean_object* v_x_802_, lean_object* v_x_803_){
_start:
{
uint8_t v_res_804_; lean_object* v_r_805_; 
v_res_804_ = lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqVar_decEq___redArg(v_x_802_, v_x_803_);
lean_dec_ref(v_x_803_);
lean_dec_ref(v_x_802_);
v_r_805_ = lean_box(v_res_804_);
return v_r_805_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqVar_decEq(lean_object* v_layout_806_, lean_object* v_x_807_, lean_object* v_x_808_){
_start:
{
uint8_t v___x_809_; 
v___x_809_ = lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqVar_decEq___redArg(v_x_807_, v_x_808_);
return v___x_809_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqVar_decEq___boxed(lean_object* v_layout_810_, lean_object* v_x_811_, lean_object* v_x_812_){
_start:
{
uint8_t v_res_813_; lean_object* v_r_814_; 
v_res_813_ = lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqVar_decEq(v_layout_810_, v_x_811_, v_x_812_);
lean_dec_ref(v_x_812_);
lean_dec_ref(v_x_811_);
lean_dec_ref(v_layout_810_);
v_r_814_ = lean_box(v_res_813_);
return v_r_814_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqVar___redArg(lean_object* v_x_815_, lean_object* v_x_816_){
_start:
{
uint8_t v___x_817_; 
v___x_817_ = lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqVar_decEq___redArg(v_x_815_, v_x_816_);
return v___x_817_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqVar___redArg___boxed(lean_object* v_x_818_, lean_object* v_x_819_){
_start:
{
uint8_t v_res_820_; lean_object* v_r_821_; 
v_res_820_ = lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqVar___redArg(v_x_818_, v_x_819_);
lean_dec_ref(v_x_819_);
lean_dec_ref(v_x_818_);
v_r_821_ = lean_box(v_res_820_);
return v_r_821_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqVar(lean_object* v_layout_822_, lean_object* v_x_823_, lean_object* v_x_824_){
_start:
{
uint8_t v___x_825_; 
v___x_825_ = lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqVar_decEq___redArg(v_x_823_, v_x_824_);
return v___x_825_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqVar___boxed(lean_object* v_layout_826_, lean_object* v_x_827_, lean_object* v_x_828_){
_start:
{
uint8_t v_res_829_; lean_object* v_r_830_; 
v_res_829_ = lp_swirl_x2dfv_Fundamentals_Air_instDecidableEqVar(v_layout_826_, v_x_827_, v_x_828_);
lean_dec_ref(v_x_828_);
lean_dec_ref(v_x_827_);
lean_dec_ref(v_layout_826_);
v_r_830_ = lean_box(v_res_829_);
return v_r_830_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Expr_ofPolynomial___redArg(lean_object* v_polynomial_831_){
_start:
{
lean_inc_ref(v_polynomial_831_);
return v_polynomial_831_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Expr_ofPolynomial___redArg___boxed(lean_object* v_polynomial_832_){
_start:
{
lean_object* v_res_833_; 
v_res_833_ = lp_swirl_x2dfv_Fundamentals_Air_Expr_ofPolynomial___redArg(v_polynomial_832_);
lean_dec_ref(v_polynomial_832_);
return v_res_833_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Expr_ofPolynomial(lean_object* v_F_834_, lean_object* v_inst_835_, lean_object* v_layout_836_, lean_object* v_polynomial_837_){
_start:
{
lean_inc_ref(v_polynomial_837_);
return v_polynomial_837_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Expr_ofPolynomial___boxed(lean_object* v_F_838_, lean_object* v_inst_839_, lean_object* v_layout_840_, lean_object* v_polynomial_841_){
_start:
{
lean_object* v_res_842_; 
v_res_842_ = lp_swirl_x2dfv_Fundamentals_Air_Expr_ofPolynomial(v_F_838_, v_inst_839_, v_layout_840_, v_polynomial_841_);
lean_dec_ref(v_polynomial_841_);
lean_dec_ref(v_layout_840_);
lean_dec_ref(v_inst_839_);
return v_res_842_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Expr_eval___redArg___lam__0(lean_object* v_assignment_843_, lean_object* v_toNPow_844_, lean_object* v_symbol_845_, lean_object* v_exponent_846_){
_start:
{
lean_object* v___x_847_; lean_object* v___x_848_; 
v___x_847_ = lean_apply_1(v_assignment_843_, v_symbol_845_);
v___x_848_ = lean_apply_2(v_toNPow_844_, v_exponent_846_, v___x_847_);
return v___x_848_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Expr_eval___redArg___lam__1(lean_object* v_toMonoid_849_, lean_object* v___f_850_, lean_object* v_toMul_851_, lean_object* v_exponents_852_, lean_object* v_coefficient_853_){
_start:
{
lean_object* v___x_854_; lean_object* v___x_855_; 
v___x_854_ = lp_mathlib_Finsupp_prod___redArg(v_toMonoid_849_, v_exponents_852_, v___f_850_);
v___x_855_ = lean_apply_2(v_toMul_851_, v_coefficient_853_, v___x_854_);
return v___x_855_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Expr_eval___redArg___lam__1___boxed(lean_object* v_toMonoid_856_, lean_object* v___f_857_, lean_object* v_toMul_858_, lean_object* v_exponents_859_, lean_object* v_coefficient_860_){
_start:
{
lean_object* v_res_861_; 
v_res_861_ = lp_swirl_x2dfv_Fundamentals_Air_Expr_eval___redArg___lam__1(v_toMonoid_856_, v___f_857_, v_toMul_858_, v_exponents_859_, v_coefficient_860_);
lean_dec_ref(v_toMonoid_856_);
return v_res_861_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Expr_eval___redArg(lean_object* v_inst_862_, lean_object* v_e_863_, lean_object* v_assignment_864_){
_start:
{
lean_object* v_toAddCommMonoid_865_; lean_object* v_toMonoid_866_; lean_object* v___x_867_; lean_object* v_toMul_868_; lean_object* v_toNPow_869_; lean_object* v___f_870_; lean_object* v___f_871_; lean_object* v___x_872_; 
v_toAddCommMonoid_865_ = lean_ctor_get(v_inst_862_, 0);
lean_inc_ref(v_toAddCommMonoid_865_);
v_toMonoid_866_ = lean_ctor_get(v_inst_862_, 1);
lean_inc_ref(v_toMonoid_866_);
v___x_867_ = lp_mathlib_instDistribOfSemiring___redArg(v_inst_862_);
v_toMul_868_ = lean_ctor_get(v___x_867_, 0);
lean_inc(v_toMul_868_);
lean_dec_ref(v___x_867_);
v_toNPow_869_ = lean_ctor_get(v_toMonoid_866_, 2);
lean_inc(v_toNPow_869_);
v___f_870_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Air_Expr_eval___redArg___lam__0), 4, 2);
lean_closure_set(v___f_870_, 0, v_assignment_864_);
lean_closure_set(v___f_870_, 1, v_toNPow_869_);
v___f_871_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Air_Expr_eval___redArg___lam__1___boxed), 5, 3);
lean_closure_set(v___f_871_, 0, v_toMonoid_866_);
lean_closure_set(v___f_871_, 1, v___f_870_);
lean_closure_set(v___f_871_, 2, v_toMul_868_);
v___x_872_ = lp_mathlib_Finsupp_sum___redArg(v_toAddCommMonoid_865_, v_e_863_, v___f_871_);
lean_dec_ref(v_toAddCommMonoid_865_);
return v___x_872_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Expr_eval(lean_object* v_F_873_, lean_object* v_inst_874_, lean_object* v_layout_875_, lean_object* v_e_876_, lean_object* v_assignment_877_){
_start:
{
lean_object* v___x_878_; 
v___x_878_ = lp_swirl_x2dfv_Fundamentals_Air_Expr_eval___redArg(v_inst_874_, v_e_876_, v_assignment_877_);
return v___x_878_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Expr_eval___boxed(lean_object* v_F_879_, lean_object* v_inst_880_, lean_object* v_layout_881_, lean_object* v_e_882_, lean_object* v_assignment_883_){
_start:
{
lean_object* v_res_884_; 
v_res_884_ = lp_swirl_x2dfv_Fundamentals_Air_Expr_eval(v_F_879_, v_inst_880_, v_layout_881_, v_e_882_, v_assignment_883_);
lean_dec_ref(v_layout_881_);
return v_res_884_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_finsuppSingle___redArg___lam__0(lean_object* v_inst_885_, lean_object* v_index_886_, lean_object* v_inst_887_, lean_object* v_value_888_, lean_object* v_candidate_889_){
_start:
{
lean_object* v___x_890_; uint8_t v___x_891_; 
v___x_890_ = lean_apply_2(v_inst_885_, v_index_886_, v_candidate_889_);
v___x_891_ = lean_unbox(v___x_890_);
if (v___x_891_ == 0)
{
lean_inc(v_inst_887_);
return v_inst_887_;
}
else
{
lean_inc(v_value_888_);
return v_value_888_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_finsuppSingle___redArg___lam__0___boxed(lean_object* v_inst_892_, lean_object* v_index_893_, lean_object* v_inst_894_, lean_object* v_value_895_, lean_object* v_candidate_896_){
_start:
{
lean_object* v_res_897_; 
v_res_897_ = lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_finsuppSingle___redArg___lam__0(v_inst_892_, v_index_893_, v_inst_894_, v_value_895_, v_candidate_896_);
lean_dec(v_value_895_);
lean_dec(v_inst_894_);
return v_res_897_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_finsuppSingle___redArg(lean_object* v_inst_898_, lean_object* v_inst_899_, lean_object* v_inst_900_, lean_object* v_index_901_, lean_object* v_value_902_){
_start:
{
lean_object* v___f_903_; lean_object* v___x_904_; uint8_t v___x_905_; 
lean_inc(v_value_902_);
lean_inc(v_inst_899_);
lean_inc(v_index_901_);
v___f_903_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_finsuppSingle___redArg___lam__0___boxed), 5, 4);
lean_closure_set(v___f_903_, 0, v_inst_898_);
lean_closure_set(v___f_903_, 1, v_index_901_);
lean_closure_set(v___f_903_, 2, v_inst_899_);
lean_closure_set(v___f_903_, 3, v_value_902_);
v___x_904_ = lean_apply_2(v_inst_900_, v_value_902_, v_inst_899_);
v___x_905_ = lean_unbox(v___x_904_);
if (v___x_905_ == 0)
{
lean_object* v___x_906_; lean_object* v___x_907_; lean_object* v___x_908_; 
v___x_906_ = lean_box(0);
v___x_907_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_907_, 0, v_index_901_);
lean_ctor_set(v___x_907_, 1, v___x_906_);
v___x_908_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_908_, 0, v___x_907_);
lean_ctor_set(v___x_908_, 1, v___f_903_);
return v___x_908_;
}
else
{
lean_object* v___x_909_; lean_object* v___x_910_; 
lean_dec(v_index_901_);
v___x_909_ = lean_box(0);
v___x_910_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_910_, 0, v___x_909_);
lean_ctor_set(v___x_910_, 1, v___f_903_);
return v___x_910_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_finsuppSingle(lean_object* v_Index_911_, lean_object* v_Value_912_, lean_object* v_inst_913_, lean_object* v_inst_914_, lean_object* v_inst_915_, lean_object* v_index_916_, lean_object* v_value_917_){
_start:
{
lean_object* v___x_918_; 
v___x_918_ = lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_finsuppSingle___redArg(v_inst_913_, v_inst_914_, v_inst_915_, v_index_916_, v_value_917_);
return v___x_918_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppAdd___redArg___lam__0(lean_object* v_self_919_, lean_object* v___y_920_){
_start:
{
lean_object* v_toFun_921_; lean_object* v___x_922_; 
v_toFun_921_ = lean_ctor_get(v_self_919_, 1);
lean_inc(v_toFun_921_);
lean_dec_ref(v_self_919_);
v___x_922_ = lean_apply_1(v_toFun_921_, v___y_920_);
return v___x_922_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppAdd___redArg___lam__2(lean_object* v___f_923_, lean_object* v_left_924_, lean_object* v_right_925_, lean_object* v_toAdd_926_, lean_object* v_index_927_){
_start:
{
lean_object* v___x_928_; lean_object* v___x_929_; lean_object* v___x_930_; 
lean_inc(v___f_923_);
lean_inc(v_index_927_);
v___x_928_ = lean_apply_2(v___f_923_, v_left_924_, v_index_927_);
v___x_929_ = lean_apply_2(v___f_923_, v_right_925_, v_index_927_);
v___x_930_ = lean_apply_2(v_toAdd_926_, v___x_928_, v___x_929_);
return v___x_930_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppAdd___redArg___lam__1(lean_object* v___f_931_, lean_object* v_left_932_, lean_object* v_right_933_, lean_object* v_toAdd_934_, lean_object* v_inst_935_, lean_object* v_toZero_936_, lean_object* v_a_937_){
_start:
{
lean_object* v___x_938_; lean_object* v___x_939_; lean_object* v___x_940_; lean_object* v___x_941_; uint8_t v___x_942_; 
lean_inc(v___f_931_);
lean_inc(v_a_937_);
v___x_938_ = lean_apply_2(v___f_931_, v_left_932_, v_a_937_);
v___x_939_ = lean_apply_2(v___f_931_, v_right_933_, v_a_937_);
v___x_940_ = lean_apply_2(v_toAdd_934_, v___x_938_, v___x_939_);
v___x_941_ = lean_apply_2(v_inst_935_, v___x_940_, v_toZero_936_);
v___x_942_ = lean_unbox(v___x_941_);
if (v___x_942_ == 0)
{
uint8_t v___x_943_; 
v___x_943_ = 1;
return v___x_943_;
}
else
{
uint8_t v___x_944_; 
v___x_944_ = 0;
return v___x_944_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppAdd___redArg___lam__1___boxed(lean_object* v___f_945_, lean_object* v_left_946_, lean_object* v_right_947_, lean_object* v_toAdd_948_, lean_object* v_inst_949_, lean_object* v_toZero_950_, lean_object* v_a_951_){
_start:
{
uint8_t v_res_952_; lean_object* v_r_953_; 
v_res_952_ = lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppAdd___redArg___lam__1(v___f_945_, v_left_946_, v_right_947_, v_toAdd_948_, v_inst_949_, v_toZero_950_, v_a_951_);
v_r_953_ = lean_box(v_res_952_);
return v_r_953_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppAdd___redArg(lean_object* v_inst_955_, lean_object* v_inst_956_, lean_object* v_inst_957_, lean_object* v_left_958_, lean_object* v_right_959_){
_start:
{
lean_object* v___x_960_; lean_object* v_toZero_961_; lean_object* v_toAdd_962_; lean_object* v___x_964_; uint8_t v_isShared_965_; uint8_t v_isSharedCheck_976_; 
v___x_960_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_956_);
v_toZero_961_ = lean_ctor_get(v___x_960_, 0);
v_toAdd_962_ = lean_ctor_get(v___x_960_, 1);
v_isSharedCheck_976_ = !lean_is_exclusive(v___x_960_);
if (v_isSharedCheck_976_ == 0)
{
v___x_964_ = v___x_960_;
v_isShared_965_ = v_isSharedCheck_976_;
goto v_resetjp_963_;
}
else
{
lean_inc(v_toAdd_962_);
lean_inc(v_toZero_961_);
lean_dec(v___x_960_);
v___x_964_ = lean_box(0);
v_isShared_965_ = v_isSharedCheck_976_;
goto v_resetjp_963_;
}
v_resetjp_963_:
{
lean_object* v_support_966_; lean_object* v_support_967_; lean_object* v___f_968_; lean_object* v___f_969_; lean_object* v___f_970_; lean_object* v___x_971_; lean_object* v___x_972_; lean_object* v___x_974_; 
v_support_966_ = lean_ctor_get(v_left_958_, 0);
lean_inc(v_support_966_);
v_support_967_ = lean_ctor_get(v_right_959_, 0);
lean_inc(v_support_967_);
v___f_968_ = ((lean_object*)(lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppAdd___redArg___closed__0));
lean_inc(v_toAdd_962_);
lean_inc_ref(v_right_959_);
lean_inc_ref(v_left_958_);
v___f_969_ = lean_alloc_closure((void*)(lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppAdd___redArg___lam__2), 5, 4);
lean_closure_set(v___f_969_, 0, v___f_968_);
lean_closure_set(v___f_969_, 1, v_left_958_);
lean_closure_set(v___f_969_, 2, v_right_959_);
lean_closure_set(v___f_969_, 3, v_toAdd_962_);
v___f_970_ = lean_alloc_closure((void*)(lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppAdd___redArg___lam__1___boxed), 7, 6);
lean_closure_set(v___f_970_, 0, v___f_968_);
lean_closure_set(v___f_970_, 1, v_left_958_);
lean_closure_set(v___f_970_, 2, v_right_959_);
lean_closure_set(v___f_970_, 3, v_toAdd_962_);
lean_closure_set(v___f_970_, 4, v_inst_957_);
lean_closure_set(v___f_970_, 5, v_toZero_961_);
v___x_971_ = lp_mathlib_Multiset_ndunion___redArg(v_inst_955_, v_support_966_, v_support_967_);
v___x_972_ = lp_mathlib_Multiset_filter___redArg(v___f_970_, v___x_971_);
if (v_isShared_965_ == 0)
{
lean_ctor_set(v___x_964_, 1, v___f_969_);
lean_ctor_set(v___x_964_, 0, v___x_972_);
v___x_974_ = v___x_964_;
goto v_reusejp_973_;
}
else
{
lean_object* v_reuseFailAlloc_975_; 
v_reuseFailAlloc_975_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_975_, 0, v___x_972_);
lean_ctor_set(v_reuseFailAlloc_975_, 1, v___f_969_);
v___x_974_ = v_reuseFailAlloc_975_;
goto v_reusejp_973_;
}
v_reusejp_973_:
{
return v___x_974_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppAdd(lean_object* v_Index_977_, lean_object* v_Value_978_, lean_object* v_inst_979_, lean_object* v_inst_980_, lean_object* v_inst_981_, lean_object* v_left_982_, lean_object* v_right_983_){
_start:
{
lean_object* v___x_984_; 
v___x_984_ = lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppAdd___redArg(v_inst_979_, v_inst_980_, v_inst_981_, v_left_982_, v_right_983_);
return v___x_984_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppNeg___redArg___lam__0(lean_object* v_toFun_985_, lean_object* v_toNeg_986_, lean_object* v_index_987_){
_start:
{
lean_object* v___x_988_; lean_object* v___x_989_; 
v___x_988_ = lean_apply_1(v_toFun_985_, v_index_987_);
v___x_989_ = lean_apply_1(v_toNeg_986_, v___x_988_);
return v___x_989_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppNeg___redArg___lam__1(lean_object* v_toFun_990_, lean_object* v_toNeg_991_, lean_object* v_inst_992_, lean_object* v_toZero_993_, lean_object* v_a_994_){
_start:
{
lean_object* v___x_995_; lean_object* v___x_996_; lean_object* v___x_997_; uint8_t v___x_998_; 
v___x_995_ = lean_apply_1(v_toFun_990_, v_a_994_);
v___x_996_ = lean_apply_1(v_toNeg_991_, v___x_995_);
v___x_997_ = lean_apply_2(v_inst_992_, v___x_996_, v_toZero_993_);
v___x_998_ = lean_unbox(v___x_997_);
if (v___x_998_ == 0)
{
uint8_t v___x_999_; 
v___x_999_ = 1;
return v___x_999_;
}
else
{
uint8_t v___x_1000_; 
v___x_1000_ = 0;
return v___x_1000_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppNeg___redArg___lam__1___boxed(lean_object* v_toFun_1001_, lean_object* v_toNeg_1002_, lean_object* v_inst_1003_, lean_object* v_toZero_1004_, lean_object* v_a_1005_){
_start:
{
uint8_t v_res_1006_; lean_object* v_r_1007_; 
v_res_1006_ = lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppNeg___redArg___lam__1(v_toFun_1001_, v_toNeg_1002_, v_inst_1003_, v_toZero_1004_, v_a_1005_);
v_r_1007_ = lean_box(v_res_1006_);
return v_r_1007_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppNeg___redArg(lean_object* v_inst_1008_, lean_object* v_inst_1009_, lean_object* v_value_1010_){
_start:
{
lean_object* v___x_1011_; lean_object* v_toZero_1012_; lean_object* v_toNeg_1013_; lean_object* v_support_1014_; lean_object* v_toFun_1015_; lean_object* v___x_1017_; uint8_t v_isShared_1018_; uint8_t v_isSharedCheck_1025_; 
v___x_1011_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_1008_);
v_toZero_1012_ = lean_ctor_get(v___x_1011_, 0);
lean_inc(v_toZero_1012_);
v_toNeg_1013_ = lean_ctor_get(v___x_1011_, 1);
lean_inc(v_toNeg_1013_);
lean_dec_ref(v___x_1011_);
v_support_1014_ = lean_ctor_get(v_value_1010_, 0);
v_toFun_1015_ = lean_ctor_get(v_value_1010_, 1);
v_isSharedCheck_1025_ = !lean_is_exclusive(v_value_1010_);
if (v_isSharedCheck_1025_ == 0)
{
v___x_1017_ = v_value_1010_;
v_isShared_1018_ = v_isSharedCheck_1025_;
goto v_resetjp_1016_;
}
else
{
lean_inc(v_toFun_1015_);
lean_inc(v_support_1014_);
lean_dec(v_value_1010_);
v___x_1017_ = lean_box(0);
v_isShared_1018_ = v_isSharedCheck_1025_;
goto v_resetjp_1016_;
}
v_resetjp_1016_:
{
lean_object* v___f_1019_; lean_object* v___f_1020_; lean_object* v___x_1021_; lean_object* v___x_1023_; 
lean_inc(v_toNeg_1013_);
lean_inc(v_toFun_1015_);
v___f_1019_ = lean_alloc_closure((void*)(lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppNeg___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1019_, 0, v_toFun_1015_);
lean_closure_set(v___f_1019_, 1, v_toNeg_1013_);
v___f_1020_ = lean_alloc_closure((void*)(lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppNeg___redArg___lam__1___boxed), 5, 4);
lean_closure_set(v___f_1020_, 0, v_toFun_1015_);
lean_closure_set(v___f_1020_, 1, v_toNeg_1013_);
lean_closure_set(v___f_1020_, 2, v_inst_1009_);
lean_closure_set(v___f_1020_, 3, v_toZero_1012_);
v___x_1021_ = lp_mathlib_Multiset_filter___redArg(v___f_1020_, v_support_1014_);
if (v_isShared_1018_ == 0)
{
lean_ctor_set(v___x_1017_, 1, v___f_1019_);
lean_ctor_set(v___x_1017_, 0, v___x_1021_);
v___x_1023_ = v___x_1017_;
goto v_reusejp_1022_;
}
else
{
lean_object* v_reuseFailAlloc_1024_; 
v_reuseFailAlloc_1024_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1024_, 0, v___x_1021_);
lean_ctor_set(v_reuseFailAlloc_1024_, 1, v___f_1019_);
v___x_1023_ = v_reuseFailAlloc_1024_;
goto v_reusejp_1022_;
}
v_reusejp_1022_:
{
return v___x_1023_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppNeg___redArg___boxed(lean_object* v_inst_1026_, lean_object* v_inst_1027_, lean_object* v_value_1028_){
_start:
{
lean_object* v_res_1029_; 
v_res_1029_ = lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppNeg___redArg(v_inst_1026_, v_inst_1027_, v_value_1028_);
lean_dec_ref(v_inst_1026_);
return v_res_1029_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppNeg(lean_object* v_Index_1030_, lean_object* v_Value_1031_, lean_object* v_inst_1032_, lean_object* v_inst_1033_, lean_object* v_inst_1034_, lean_object* v_value_1035_){
_start:
{
lean_object* v___x_1036_; 
v___x_1036_ = lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppNeg___redArg(v_inst_1033_, v_inst_1034_, v_value_1035_);
return v___x_1036_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppNeg___boxed(lean_object* v_Index_1037_, lean_object* v_Value_1038_, lean_object* v_inst_1039_, lean_object* v_inst_1040_, lean_object* v_inst_1041_, lean_object* v_value_1042_){
_start:
{
lean_object* v_res_1043_; 
v_res_1043_ = lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppNeg(v_Index_1037_, v_Value_1038_, v_inst_1039_, v_inst_1040_, v_inst_1041_, v_value_1042_);
lean_dec_ref(v_inst_1040_);
lean_dec_ref(v_inst_1039_);
return v_res_1043_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_add___redArg___lam__0(lean_object* v_inst_1044_, lean_object* v_a_1045_, lean_object* v_b_1046_){
_start:
{
lean_object* v___x_1047_; uint8_t v___x_1048_; 
v___x_1047_ = lean_alloc_closure((void*)(l_instDecidableEqNat___boxed), 2, 0);
v___x_1048_ = lp_mathlib_Finsupp_instDecidableEq___redArg(v_inst_1044_, v___x_1047_, v_a_1045_, v_b_1046_);
return v___x_1048_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_add___redArg___lam__0___boxed(lean_object* v_inst_1049_, lean_object* v_a_1050_, lean_object* v_b_1051_){
_start:
{
uint8_t v_res_1052_; lean_object* v_r_1053_; 
v_res_1052_ = lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_add___redArg___lam__0(v_inst_1049_, v_a_1050_, v_b_1051_);
v_r_1053_ = lean_box(v_res_1052_);
return v_r_1053_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_add___redArg(lean_object* v_inst_1054_, lean_object* v_inst_1055_, lean_object* v_inst_1056_, lean_object* v_left_1057_, lean_object* v_right_1058_){
_start:
{
lean_object* v___x_1059_; lean_object* v_toAddMonoidWithOne_1060_; lean_object* v_toAddMonoid_1061_; lean_object* v___f_1062_; lean_object* v___x_1063_; lean_object* v___x_1064_; 
v___x_1059_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_inst_1054_);
v_toAddMonoidWithOne_1060_ = lean_ctor_get(v___x_1059_, 1);
lean_inc_ref(v_toAddMonoidWithOne_1060_);
lean_dec_ref(v___x_1059_);
v_toAddMonoid_1061_ = lean_ctor_get(v_toAddMonoidWithOne_1060_, 1);
lean_inc_ref(v_toAddMonoid_1061_);
lean_dec_ref(v_toAddMonoidWithOne_1060_);
v___f_1062_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_add___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_1062_, 0, v_inst_1056_);
v___x_1063_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_1061_);
lean_dec_ref(v_toAddMonoid_1061_);
v___x_1064_ = lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppAdd___redArg(v___f_1062_, v___x_1063_, v_inst_1055_, v_left_1057_, v_right_1058_);
return v___x_1064_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_add(lean_object* v_F_1065_, lean_object* v_Variable_1066_, lean_object* v_inst_1067_, lean_object* v_inst_1068_, lean_object* v_inst_1069_, lean_object* v_left_1070_, lean_object* v_right_1071_){
_start:
{
lean_object* v___x_1072_; 
v___x_1072_ = lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_add___redArg(v_inst_1067_, v_inst_1068_, v_inst_1069_, v_left_1070_, v_right_1071_);
return v___x_1072_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_neg___redArg(lean_object* v_inst_1073_, lean_object* v_inst_1074_, lean_object* v_value_1075_){
_start:
{
lean_object* v___x_1076_; lean_object* v___x_1077_; lean_object* v___x_1078_; 
v___x_1076_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_inst_1073_);
v___x_1077_ = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(v___x_1076_);
lean_dec_ref(v___x_1076_);
v___x_1078_ = lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppNeg___redArg(v___x_1077_, v_inst_1074_, v_value_1075_);
lean_dec_ref(v___x_1077_);
return v___x_1078_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_neg(lean_object* v_F_1079_, lean_object* v_Variable_1080_, lean_object* v_inst_1081_, lean_object* v_inst_1082_, lean_object* v_inst_1083_, lean_object* v_value_1084_){
_start:
{
lean_object* v___x_1085_; 
v___x_1085_ = lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_neg___redArg(v_inst_1081_, v_inst_1082_, v_value_1084_);
return v___x_1085_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_neg___boxed(lean_object* v_F_1086_, lean_object* v_Variable_1087_, lean_object* v_inst_1088_, lean_object* v_inst_1089_, lean_object* v_inst_1090_, lean_object* v_value_1091_){
_start:
{
lean_object* v_res_1092_; 
v_res_1092_ = lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_neg(v_F_1086_, v_Variable_1087_, v_inst_1088_, v_inst_1089_, v_inst_1090_, v_value_1091_);
lean_dec_ref(v_inst_1090_);
return v_res_1092_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_monomial___redArg(lean_object* v_inst_1093_, lean_object* v_inst_1094_, lean_object* v_inst_1095_, lean_object* v_exponents_1096_, lean_object* v_coefficient_1097_){
_start:
{
lean_object* v_toSemiring_1098_; lean_object* v___x_1099_; lean_object* v_toZero_1100_; lean_object* v___f_1101_; lean_object* v___x_1102_; 
v_toSemiring_1098_ = lean_ctor_get(v_inst_1093_, 0);
lean_inc_ref(v_toSemiring_1098_);
lean_dec_ref(v_inst_1093_);
v___x_1099_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_toSemiring_1098_);
v_toZero_1100_ = lean_ctor_get(v___x_1099_, 1);
lean_inc(v_toZero_1100_);
lean_dec_ref(v___x_1099_);
v___f_1101_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_add___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_1101_, 0, v_inst_1095_);
v___x_1102_ = lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_finsuppSingle___redArg(v___f_1101_, v_toZero_1100_, v_inst_1094_, v_exponents_1096_, v_coefficient_1097_);
return v___x_1102_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_monomial(lean_object* v_F_1103_, lean_object* v_Variable_1104_, lean_object* v_inst_1105_, lean_object* v_inst_1106_, lean_object* v_inst_1107_, lean_object* v_exponents_1108_, lean_object* v_coefficient_1109_){
_start:
{
lean_object* v___x_1110_; 
v___x_1110_ = lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_monomial___redArg(v_inst_1105_, v_inst_1106_, v_inst_1107_, v_exponents_1108_, v_coefficient_1109_);
return v___x_1110_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instZeroFinsuppAccumulator___redArg___lam__0(lean_object* v_toZero_1111_, lean_object* v_x_1112_){
_start:
{
lean_inc(v_toZero_1111_);
return v_toZero_1111_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instZeroFinsuppAccumulator___redArg___lam__0___boxed(lean_object* v_toZero_1113_, lean_object* v_x_1114_){
_start:
{
lean_object* v_res_1115_; 
v_res_1115_ = lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instZeroFinsuppAccumulator___redArg___lam__0(v_toZero_1113_, v_x_1114_);
lean_dec(v_x_1114_);
lean_dec(v_toZero_1113_);
return v_res_1115_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instZeroFinsuppAccumulator___redArg(lean_object* v_inst_1116_){
_start:
{
lean_object* v___x_1117_; lean_object* v___x_1118_; lean_object* v_toZero_1119_; lean_object* v___x_1121_; uint8_t v_isShared_1122_; uint8_t v_isSharedCheck_1128_; 
v___x_1117_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_1116_);
v___x_1118_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_1117_);
v_toZero_1119_ = lean_ctor_get(v___x_1118_, 0);
v_isSharedCheck_1128_ = !lean_is_exclusive(v___x_1118_);
if (v_isSharedCheck_1128_ == 0)
{
lean_object* v_unused_1129_; 
v_unused_1129_ = lean_ctor_get(v___x_1118_, 1);
lean_dec(v_unused_1129_);
v___x_1121_ = v___x_1118_;
v_isShared_1122_ = v_isSharedCheck_1128_;
goto v_resetjp_1120_;
}
else
{
lean_inc(v_toZero_1119_);
lean_dec(v___x_1118_);
v___x_1121_ = lean_box(0);
v_isShared_1122_ = v_isSharedCheck_1128_;
goto v_resetjp_1120_;
}
v_resetjp_1120_:
{
lean_object* v___f_1123_; lean_object* v___x_1124_; lean_object* v___x_1126_; 
v___f_1123_ = lean_alloc_closure((void*)(lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instZeroFinsuppAccumulator___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1123_, 0, v_toZero_1119_);
v___x_1124_ = lean_box(0);
if (v_isShared_1122_ == 0)
{
lean_ctor_set(v___x_1121_, 1, v___f_1123_);
lean_ctor_set(v___x_1121_, 0, v___x_1124_);
v___x_1126_ = v___x_1121_;
goto v_reusejp_1125_;
}
else
{
lean_object* v_reuseFailAlloc_1127_; 
v_reuseFailAlloc_1127_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1127_, 0, v___x_1124_);
lean_ctor_set(v_reuseFailAlloc_1127_, 1, v___f_1123_);
v___x_1126_ = v_reuseFailAlloc_1127_;
goto v_reusejp_1125_;
}
v_reusejp_1125_:
{
return v___x_1126_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instZeroFinsuppAccumulator___redArg___boxed(lean_object* v_inst_1130_){
_start:
{
lean_object* v_res_1131_; 
v_res_1131_ = lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instZeroFinsuppAccumulator___redArg(v_inst_1130_);
lean_dec_ref(v_inst_1130_);
return v_res_1131_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instZeroFinsuppAccumulator(lean_object* v_Index_1132_, lean_object* v_Value_1133_, lean_object* v_inst_1134_, lean_object* v_inst_1135_, lean_object* v_inst_1136_){
_start:
{
lean_object* v___x_1137_; 
v___x_1137_ = lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instZeroFinsuppAccumulator___redArg(v_inst_1135_);
return v___x_1137_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instZeroFinsuppAccumulator___boxed(lean_object* v_Index_1138_, lean_object* v_Value_1139_, lean_object* v_inst_1140_, lean_object* v_inst_1141_, lean_object* v_inst_1142_){
_start:
{
lean_object* v_res_1143_; 
v_res_1143_ = lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instZeroFinsuppAccumulator(v_Index_1138_, v_Value_1139_, v_inst_1140_, v_inst_1141_, v_inst_1142_);
lean_dec_ref(v_inst_1142_);
lean_dec_ref(v_inst_1141_);
lean_dec_ref(v_inst_1140_);
return v_res_1143_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instAddFinsuppAccumulator___redArg___lam__0(lean_object* v_inst_1144_, lean_object* v___x_1145_, lean_object* v_inst_1146_, lean_object* v_left_1147_, lean_object* v_right_1148_){
_start:
{
lean_object* v___x_1149_; 
v___x_1149_ = lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppAdd___redArg(v_inst_1144_, v___x_1145_, v_inst_1146_, v_left_1147_, v_right_1148_);
return v___x_1149_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instAddFinsuppAccumulator___redArg(lean_object* v_inst_1150_, lean_object* v_inst_1151_, lean_object* v_inst_1152_){
_start:
{
lean_object* v___x_1153_; lean_object* v___f_1154_; 
v___x_1153_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_1151_);
v___f_1154_ = lean_alloc_closure((void*)(lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instAddFinsuppAccumulator___redArg___lam__0), 5, 3);
lean_closure_set(v___f_1154_, 0, v_inst_1150_);
lean_closure_set(v___f_1154_, 1, v___x_1153_);
lean_closure_set(v___f_1154_, 2, v_inst_1152_);
return v___f_1154_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instAddFinsuppAccumulator___redArg___boxed(lean_object* v_inst_1155_, lean_object* v_inst_1156_, lean_object* v_inst_1157_){
_start:
{
lean_object* v_res_1158_; 
v_res_1158_ = lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instAddFinsuppAccumulator___redArg(v_inst_1155_, v_inst_1156_, v_inst_1157_);
lean_dec_ref(v_inst_1156_);
return v_res_1158_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instAddFinsuppAccumulator(lean_object* v_Index_1159_, lean_object* v_Value_1160_, lean_object* v_inst_1161_, lean_object* v_inst_1162_, lean_object* v_inst_1163_){
_start:
{
lean_object* v___x_1164_; 
v___x_1164_ = lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instAddFinsuppAccumulator___redArg(v_inst_1161_, v_inst_1162_, v_inst_1163_);
return v___x_1164_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instAddFinsuppAccumulator___boxed(lean_object* v_Index_1165_, lean_object* v_Value_1166_, lean_object* v_inst_1167_, lean_object* v_inst_1168_, lean_object* v_inst_1169_){
_start:
{
lean_object* v_res_1170_; 
v_res_1170_ = lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instAddFinsuppAccumulator(v_Index_1165_, v_Value_1166_, v_inst_1167_, v_inst_1168_, v_inst_1169_);
lean_dec_ref(v_inst_1168_);
return v_res_1170_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instAddCommMonoidFinsuppAccumulator___redArg(lean_object* v_inst_1171_, lean_object* v_inst_1172_, lean_object* v_inst_1173_){
_start:
{
lean_object* v___x_1174_; lean_object* v___x_1175_; lean_object* v___x_1176_; lean_object* v___x_1177_; 
v___x_1174_ = lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instZeroFinsuppAccumulator___redArg(v_inst_1172_);
v___x_1175_ = lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instAddFinsuppAccumulator___redArg(v_inst_1171_, v_inst_1172_, v_inst_1173_);
lean_inc_ref(v___x_1175_);
lean_inc_ref(v___x_1174_);
v___x_1176_ = lean_alloc_closure((void*)(l_nsmulRec___boxed), 5, 3);
lean_closure_set(v___x_1176_, 0, lean_box(0));
lean_closure_set(v___x_1176_, 1, v___x_1174_);
lean_closure_set(v___x_1176_, 2, v___x_1175_);
v___x_1177_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1177_, 0, v___x_1174_);
lean_ctor_set(v___x_1177_, 1, v___x_1175_);
lean_ctor_set(v___x_1177_, 2, v___x_1176_);
return v___x_1177_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instAddCommMonoidFinsuppAccumulator___redArg___boxed(lean_object* v_inst_1178_, lean_object* v_inst_1179_, lean_object* v_inst_1180_){
_start:
{
lean_object* v_res_1181_; 
v_res_1181_ = lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instAddCommMonoidFinsuppAccumulator___redArg(v_inst_1178_, v_inst_1179_, v_inst_1180_);
lean_dec_ref(v_inst_1179_);
return v_res_1181_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instAddCommMonoidFinsuppAccumulator(lean_object* v_Index_1182_, lean_object* v_Value_1183_, lean_object* v_inst_1184_, lean_object* v_inst_1185_, lean_object* v_inst_1186_){
_start:
{
lean_object* v___x_1187_; 
v___x_1187_ = lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instAddCommMonoidFinsuppAccumulator___redArg(v_inst_1184_, v_inst_1185_, v_inst_1186_);
return v___x_1187_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instAddCommMonoidFinsuppAccumulator___boxed(lean_object* v_Index_1188_, lean_object* v_Value_1189_, lean_object* v_inst_1190_, lean_object* v_inst_1191_, lean_object* v_inst_1192_){
_start:
{
lean_object* v_res_1193_; 
v_res_1193_ = lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instAddCommMonoidFinsuppAccumulator(v_Index_1188_, v_Value_1189_, v_inst_1190_, v_inst_1191_, v_inst_1192_);
lean_dec_ref(v_inst_1191_);
return v_res_1193_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_mul___redArg___lam__1(lean_object* v_inst_1194_, lean_object* v___x_1195_, lean_object* v_leftExponent_1196_, lean_object* v_toMul_1197_, lean_object* v_leftCoefficient_1198_, lean_object* v___f_1199_, lean_object* v_toZero_1200_, lean_object* v_inst_1201_, lean_object* v_rightExponent_1202_, lean_object* v_rightCoefficient_1203_){
_start:
{
lean_object* v___x_1204_; lean_object* v___x_1205_; lean_object* v___x_1206_; lean_object* v___x_1207_; 
v___x_1204_ = lean_alloc_closure((void*)(l_instDecidableEqNat___boxed), 2, 0);
v___x_1205_ = lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_finsuppAdd___redArg(v_inst_1194_, v___x_1195_, v___x_1204_, v_leftExponent_1196_, v_rightExponent_1202_);
v___x_1206_ = lean_apply_2(v_toMul_1197_, v_leftCoefficient_1198_, v_rightCoefficient_1203_);
v___x_1207_ = lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_finsuppSingle___redArg(v___f_1199_, v_toZero_1200_, v_inst_1201_, v___x_1205_, v___x_1206_);
return v___x_1207_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_mul___redArg___lam__0(lean_object* v_inst_1208_, lean_object* v___x_1209_, lean_object* v_toMul_1210_, lean_object* v___f_1211_, lean_object* v_toZero_1212_, lean_object* v_inst_1213_, lean_object* v___x_1214_, lean_object* v_right_1215_, lean_object* v_leftExponent_1216_, lean_object* v_leftCoefficient_1217_){
_start:
{
lean_object* v___f_1218_; lean_object* v___x_1219_; 
v___f_1218_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_mul___redArg___lam__1), 10, 8);
lean_closure_set(v___f_1218_, 0, v_inst_1208_);
lean_closure_set(v___f_1218_, 1, v___x_1209_);
lean_closure_set(v___f_1218_, 2, v_leftExponent_1216_);
lean_closure_set(v___f_1218_, 3, v_toMul_1210_);
lean_closure_set(v___f_1218_, 4, v_leftCoefficient_1217_);
lean_closure_set(v___f_1218_, 5, v___f_1211_);
lean_closure_set(v___f_1218_, 6, v_toZero_1212_);
lean_closure_set(v___f_1218_, 7, v_inst_1213_);
v___x_1219_ = lp_mathlib_Finsupp_sum___redArg(v___x_1214_, v_right_1215_, v___f_1218_);
return v___x_1219_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_mul___redArg___lam__0___boxed(lean_object* v_inst_1220_, lean_object* v___x_1221_, lean_object* v_toMul_1222_, lean_object* v___f_1223_, lean_object* v_toZero_1224_, lean_object* v_inst_1225_, lean_object* v___x_1226_, lean_object* v_right_1227_, lean_object* v_leftExponent_1228_, lean_object* v_leftCoefficient_1229_){
_start:
{
lean_object* v_res_1230_; 
v_res_1230_ = lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_mul___redArg___lam__0(v_inst_1220_, v___x_1221_, v_toMul_1222_, v___f_1223_, v_toZero_1224_, v_inst_1225_, v___x_1226_, v_right_1227_, v_leftExponent_1228_, v_leftCoefficient_1229_);
lean_dec_ref(v___x_1226_);
return v_res_1230_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_mul___redArg___closed__0(void){
_start:
{
lean_object* v___x_1231_; lean_object* v___x_1232_; 
v___x_1231_ = lp_mathlib_Nat_instAddCancelCommMonoid;
v___x_1232_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v___x_1231_);
return v___x_1232_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_mul___redArg(lean_object* v_inst_1233_, lean_object* v_inst_1234_, lean_object* v_inst_1235_, lean_object* v_left_1236_, lean_object* v_right_1237_){
_start:
{
lean_object* v___x_1238_; lean_object* v_toSemiring_1239_; lean_object* v_toAddCommMonoid_1240_; lean_object* v___x_1241_; lean_object* v___x_1242_; lean_object* v_toZero_1243_; lean_object* v___x_1244_; lean_object* v_toMul_1245_; lean_object* v___f_1246_; lean_object* v___x_1247_; lean_object* v___f_1248_; lean_object* v___x_1249_; 
v___x_1238_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_mul___redArg___closed__0, &lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_mul___redArg___closed__0_once, _init_lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_mul___redArg___closed__0);
v_toSemiring_1239_ = lean_ctor_get(v_inst_1233_, 0);
lean_inc_ref(v_toSemiring_1239_);
lean_dec_ref(v_inst_1233_);
v_toAddCommMonoid_1240_ = lean_ctor_get(v_toSemiring_1239_, 0);
lean_inc_ref(v_toAddCommMonoid_1240_);
v___x_1241_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddCommMonoid_1240_);
v___x_1242_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_1241_);
v_toZero_1243_ = lean_ctor_get(v___x_1242_, 0);
lean_inc(v_toZero_1243_);
lean_dec_ref(v___x_1242_);
v___x_1244_ = lp_mathlib_instDistribOfSemiring___redArg(v_toSemiring_1239_);
v_toMul_1245_ = lean_ctor_get(v___x_1244_, 0);
lean_inc(v_toMul_1245_);
lean_dec_ref(v___x_1244_);
lean_inc_ref(v_inst_1235_);
v___f_1246_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_add___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_1246_, 0, v_inst_1235_);
lean_inc_ref(v_inst_1234_);
lean_inc_ref(v___f_1246_);
v___x_1247_ = lp_swirl_x2dfv___private_Fundamentals_Spec_Air_0__Fundamentals_Air_ExecutablePolynomial_FinsuppAccumulator_instAddCommMonoidFinsuppAccumulator___redArg(v___f_1246_, v_toAddCommMonoid_1240_, v_inst_1234_);
lean_dec_ref(v_toAddCommMonoid_1240_);
lean_inc_ref(v___x_1247_);
v___f_1248_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_mul___redArg___lam__0___boxed), 10, 8);
lean_closure_set(v___f_1248_, 0, v_inst_1235_);
lean_closure_set(v___f_1248_, 1, v___x_1238_);
lean_closure_set(v___f_1248_, 2, v_toMul_1245_);
lean_closure_set(v___f_1248_, 3, v___f_1246_);
lean_closure_set(v___f_1248_, 4, v_toZero_1243_);
lean_closure_set(v___f_1248_, 5, v_inst_1234_);
lean_closure_set(v___f_1248_, 6, v___x_1247_);
lean_closure_set(v___f_1248_, 7, v_right_1237_);
v___x_1249_ = lp_mathlib_Finsupp_sum___redArg(v___x_1247_, v_left_1236_, v___f_1248_);
lean_dec_ref(v___x_1247_);
return v___x_1249_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_mul(lean_object* v_F_1250_, lean_object* v_Variable_1251_, lean_object* v_inst_1252_, lean_object* v_inst_1253_, lean_object* v_inst_1254_, lean_object* v_left_1255_, lean_object* v_right_1256_){
_start:
{
lean_object* v___x_1257_; 
v___x_1257_ = lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_mul___redArg(v_inst_1252_, v_inst_1253_, v_inst_1254_, v_left_1255_, v_right_1256_);
return v___x_1257_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_ofVariable___redArg(lean_object* v_inst_1258_, lean_object* v_inst_1259_, lean_object* v_inst_1260_, lean_object* v_symbol_1261_){
_start:
{
lean_object* v___x_1262_; lean_object* v_toZero_1263_; lean_object* v___x_1264_; lean_object* v_toAddMonoidWithOne_1265_; lean_object* v_toOne_1266_; lean_object* v___x_1267_; lean_object* v___x_1268_; lean_object* v___x_1269_; lean_object* v___x_1270_; 
v___x_1262_ = lp_mathlib_Nat_instMulZeroClass;
v_toZero_1263_ = lean_ctor_get(v___x_1262_, 1);
lean_inc_ref(v_inst_1258_);
v___x_1264_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_inst_1258_);
v_toAddMonoidWithOne_1265_ = lean_ctor_get(v___x_1264_, 1);
lean_inc_ref(v_toAddMonoidWithOne_1265_);
lean_dec_ref(v___x_1264_);
v_toOne_1266_ = lean_ctor_get(v_toAddMonoidWithOne_1265_, 2);
lean_inc(v_toOne_1266_);
lean_dec_ref(v_toAddMonoidWithOne_1265_);
v___x_1267_ = lean_alloc_closure((void*)(l_instDecidableEqNat___boxed), 2, 0);
v___x_1268_ = lean_unsigned_to_nat(1u);
lean_inc(v_toZero_1263_);
lean_inc_ref(v_inst_1260_);
v___x_1269_ = lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_finsuppSingle___redArg(v_inst_1260_, v_toZero_1263_, v___x_1267_, v_symbol_1261_, v___x_1268_);
v___x_1270_ = lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_monomial___redArg(v_inst_1258_, v_inst_1259_, v_inst_1260_, v___x_1269_, v_toOne_1266_);
return v___x_1270_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_ofVariable(lean_object* v_F_1271_, lean_object* v_Variable_1272_, lean_object* v_inst_1273_, lean_object* v_inst_1274_, lean_object* v_inst_1275_, lean_object* v_symbol_1276_){
_start:
{
lean_object* v___x_1277_; 
v___x_1277_ = lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_ofVariable___redArg(v_inst_1273_, v_inst_1274_, v_inst_1275_, v_symbol_1276_);
return v___x_1277_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_constant___redArg___lam__0(lean_object* v_x_1278_){
_start:
{
lean_object* v___x_1279_; 
v___x_1279_ = lean_unsigned_to_nat(0u);
return v___x_1279_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_constant___redArg___lam__0___boxed(lean_object* v_x_1280_){
_start:
{
lean_object* v_res_1281_; 
v_res_1281_ = lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_constant___redArg___lam__0(v_x_1280_);
lean_dec(v_x_1280_);
return v_res_1281_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_constant___redArg(lean_object* v_inst_1286_, lean_object* v_inst_1287_, lean_object* v_inst_1288_, lean_object* v_value_1289_){
_start:
{
lean_object* v___x_1290_; lean_object* v___x_1291_; 
v___x_1290_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_constant___redArg___closed__1));
v___x_1291_ = lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_monomial___redArg(v_inst_1286_, v_inst_1287_, v_inst_1288_, v___x_1290_, v_value_1289_);
return v___x_1291_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_constant(lean_object* v_F_1292_, lean_object* v_Variable_1293_, lean_object* v_inst_1294_, lean_object* v_inst_1295_, lean_object* v_inst_1296_, lean_object* v_value_1297_){
_start:
{
lean_object* v___x_1298_; 
v___x_1298_ = lp_swirl_x2dfv_Fundamentals_Air_ExecutablePolynomial_constant___redArg(v_inst_1294_, v_inst_1295_, v_inst_1296_, v_value_1297_);
return v___x_1298_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_traceWeight___redArg(lean_object* v_x_1299_){
_start:
{
if (lean_obj_tag(v_x_1299_) == 2)
{
lean_object* v___x_1300_; 
v___x_1300_ = lean_unsigned_to_nat(0u);
return v___x_1300_;
}
else
{
lean_object* v___x_1301_; 
v___x_1301_ = lean_unsigned_to_nat(1u);
return v___x_1301_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_traceWeight___redArg___boxed(lean_object* v_x_1302_){
_start:
{
lean_object* v_res_1303_; 
v_res_1303_ = lp_swirl_x2dfv_Fundamentals_Air_Var_traceWeight___redArg(v_x_1302_);
lean_dec_ref(v_x_1302_);
return v_res_1303_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_traceWeight(lean_object* v_layout_1304_, lean_object* v_x_1305_){
_start:
{
lean_object* v___x_1306_; 
v___x_1306_ = lp_swirl_x2dfv_Fundamentals_Air_Var_traceWeight___redArg(v_x_1305_);
return v___x_1306_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Var_traceWeight___boxed(lean_object* v_layout_1307_, lean_object* v_x_1308_){
_start:
{
lean_object* v_res_1309_; 
v_res_1309_ = lp_swirl_x2dfv_Fundamentals_Air_Var_traceWeight(v_layout_1307_, v_x_1308_);
lean_dec_ref(v_x_1308_);
lean_dec_ref(v_layout_1307_);
return v_res_1309_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Trace_get___redArg___lam__0(lean_object* v_trace_1310_, lean_object* v_row_1311_, lean_object* v_part_1312_){
_start:
{
lean_object* v_rows_1313_; lean_object* v___x_1314_; lean_object* v___x_1315_; 
v_rows_1313_ = lean_ctor_get(v_trace_1310_, 1);
lean_inc_ref(v_rows_1313_);
lean_dec_ref(v_trace_1310_);
v___x_1314_ = lean_apply_1(v_rows_1313_, v_part_1312_);
v___x_1315_ = lean_array_fget(v___x_1314_, v_row_1311_);
lean_dec_ref(v___x_1314_);
return v___x_1315_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Trace_get___redArg___lam__0___boxed(lean_object* v_trace_1316_, lean_object* v_row_1317_, lean_object* v_part_1318_){
_start:
{
lean_object* v_res_1319_; 
v_res_1319_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_get___redArg___lam__0(v_trace_1316_, v_row_1317_, v_part_1318_);
lean_dec(v_row_1317_);
return v_res_1319_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Trace_get___redArg(lean_object* v_trace_1320_, lean_object* v_row_1321_){
_start:
{
lean_object* v___f_1322_; 
v___f_1322_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Air_Trace_get___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1322_, 0, v_trace_1320_);
lean_closure_set(v___f_1322_, 1, v_row_1321_);
return v___f_1322_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Trace_get(lean_object* v_F_1323_, lean_object* v_layout_1324_, lean_object* v_trace_1325_, lean_object* v_row_1326_){
_start:
{
lean_object* v___f_1327_; 
v___f_1327_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Air_Trace_get___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1327_, 0, v_trace_1325_);
lean_closure_set(v___f_1327_, 1, v_row_1326_);
return v___f_1327_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Trace_get___boxed(lean_object* v_F_1328_, lean_object* v_layout_1329_, lean_object* v_trace_1330_, lean_object* v_row_1331_){
_start:
{
lean_object* v_res_1332_; 
v_res_1332_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_get(v_F_1328_, v_layout_1329_, v_trace_1330_, v_row_1331_);
lean_dec_ref(v_layout_1329_);
return v_res_1332_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Trace_nextRowIndex___redArg(lean_object* v_trace_1333_, lean_object* v_i_1334_){
_start:
{
lean_object* v_height_1335_; lean_object* v___x_1336_; lean_object* v___x_1337_; lean_object* v___x_1338_; 
v_height_1335_ = lean_ctor_get(v_trace_1333_, 0);
v___x_1336_ = lean_unsigned_to_nat(1u);
v___x_1337_ = lean_nat_add(v_i_1334_, v___x_1336_);
v___x_1338_ = lean_nat_mod(v___x_1337_, v_height_1335_);
lean_dec(v___x_1337_);
return v___x_1338_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Trace_nextRowIndex___redArg___boxed(lean_object* v_trace_1339_, lean_object* v_i_1340_){
_start:
{
lean_object* v_res_1341_; 
v_res_1341_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_nextRowIndex___redArg(v_trace_1339_, v_i_1340_);
lean_dec(v_i_1340_);
lean_dec_ref(v_trace_1339_);
return v_res_1341_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Trace_nextRowIndex(lean_object* v_F_1342_, lean_object* v_layout_1343_, lean_object* v_trace_1344_, lean_object* v_i_1345_){
_start:
{
lean_object* v___x_1346_; 
v___x_1346_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_nextRowIndex___redArg(v_trace_1344_, v_i_1345_);
return v___x_1346_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Trace_nextRowIndex___boxed(lean_object* v_F_1347_, lean_object* v_layout_1348_, lean_object* v_trace_1349_, lean_object* v_i_1350_){
_start:
{
lean_object* v_res_1351_; 
v_res_1351_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_nextRowIndex(v_F_1347_, v_layout_1348_, v_trace_1349_, v_i_1350_);
lean_dec(v_i_1350_);
lean_dec_ref(v_trace_1349_);
lean_dec_ref(v_layout_1348_);
return v_res_1351_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Trace_col___redArg(lean_object* v_trace_1352_, lean_object* v_column_1353_, lean_object* v_row_1354_){
_start:
{
lean_object* v___f_1355_; lean_object* v___x_1356_; 
v___f_1355_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Air_Trace_get___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1355_, 0, v_trace_1352_);
lean_closure_set(v___f_1355_, 1, v_row_1354_);
v___x_1356_ = lp_swirl_x2dfv_Fundamentals_Air_Row_get___redArg(v___f_1355_, v_column_1353_);
return v___x_1356_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Trace_col(lean_object* v_F_1357_, lean_object* v_layout_1358_, lean_object* v_trace_1359_, lean_object* v_column_1360_, lean_object* v_row_1361_){
_start:
{
lean_object* v___x_1362_; 
v___x_1362_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_col___redArg(v_trace_1359_, v_column_1360_, v_row_1361_);
return v___x_1362_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Trace_col___boxed(lean_object* v_F_1363_, lean_object* v_layout_1364_, lean_object* v_trace_1365_, lean_object* v_column_1366_, lean_object* v_row_1367_){
_start:
{
lean_object* v_res_1368_; 
v_res_1368_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_col(v_F_1363_, v_layout_1364_, v_trace_1365_, v_column_1366_, v_row_1367_);
lean_dec_ref(v_layout_1364_);
return v_res_1368_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Trace_colNext___redArg(lean_object* v_trace_1369_, lean_object* v_column_1370_, lean_object* v_row_1371_){
_start:
{
lean_object* v___x_1372_; lean_object* v___f_1373_; lean_object* v___x_1374_; 
v___x_1372_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_nextRowIndex___redArg(v_trace_1369_, v_row_1371_);
v___f_1373_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Air_Trace_get___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1373_, 0, v_trace_1369_);
lean_closure_set(v___f_1373_, 1, v___x_1372_);
v___x_1374_ = lp_swirl_x2dfv_Fundamentals_Air_Row_get___redArg(v___f_1373_, v_column_1370_);
return v___x_1374_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Trace_colNext___redArg___boxed(lean_object* v_trace_1375_, lean_object* v_column_1376_, lean_object* v_row_1377_){
_start:
{
lean_object* v_res_1378_; 
v_res_1378_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_colNext___redArg(v_trace_1375_, v_column_1376_, v_row_1377_);
lean_dec(v_row_1377_);
return v_res_1378_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Trace_colNext(lean_object* v_F_1379_, lean_object* v_layout_1380_, lean_object* v_trace_1381_, lean_object* v_column_1382_, lean_object* v_row_1383_){
_start:
{
lean_object* v___x_1384_; 
v___x_1384_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_colNext___redArg(v_trace_1381_, v_column_1382_, v_row_1383_);
return v___x_1384_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Trace_colNext___boxed(lean_object* v_F_1385_, lean_object* v_layout_1386_, lean_object* v_trace_1387_, lean_object* v_column_1388_, lean_object* v_row_1389_){
_start:
{
lean_object* v_res_1390_; 
v_res_1390_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_colNext(v_F_1385_, v_layout_1386_, v_trace_1387_, v_column_1388_, v_row_1389_);
lean_dec(v_row_1389_);
lean_dec_ref(v_layout_1386_);
return v_res_1390_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_publicValueCount___redArg(lean_object* v_A_1391_){
_start:
{
lean_object* v_layout_1392_; lean_object* v_publicValueCount_1393_; 
v_layout_1392_ = lean_ctor_get(v_A_1391_, 0);
v_publicValueCount_1393_ = lean_ctor_get(v_layout_1392_, 3);
lean_inc(v_publicValueCount_1393_);
return v_publicValueCount_1393_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_publicValueCount___redArg___boxed(lean_object* v_A_1394_){
_start:
{
lean_object* v_res_1395_; 
v_res_1395_ = lp_swirl_x2dfv_Fundamentals_Air_AIR_publicValueCount___redArg(v_A_1394_);
lean_dec_ref(v_A_1394_);
return v_res_1395_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_publicValueCount(lean_object* v_F_1396_, lean_object* v_inst_1397_, lean_object* v_A_1398_){
_start:
{
lean_object* v_layout_1399_; lean_object* v_publicValueCount_1400_; 
v_layout_1399_ = lean_ctor_get(v_A_1398_, 0);
v_publicValueCount_1400_ = lean_ctor_get(v_layout_1399_, 3);
lean_inc(v_publicValueCount_1400_);
return v_publicValueCount_1400_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_publicValueCount___boxed(lean_object* v_F_1401_, lean_object* v_inst_1402_, lean_object* v_A_1403_){
_start:
{
lean_object* v_res_1404_; 
v_res_1404_ = lp_swirl_x2dfv_Fundamentals_Air_AIR_publicValueCount(v_F_1401_, v_inst_1402_, v_A_1403_);
lean_dec_ref(v_A_1403_);
lean_dec_ref(v_inst_1402_);
return v_res_1404_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_get___redArg(lean_object* v_trace_1405_, lean_object* v_row_1406_){
_start:
{
lean_object* v___f_1407_; 
v___f_1407_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Air_Trace_get___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1407_, 0, v_trace_1405_);
lean_closure_set(v___f_1407_, 1, v_row_1406_);
return v___f_1407_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_get(lean_object* v_F_1408_, lean_object* v_inst_1409_, lean_object* v_A_1410_, lean_object* v_trace_1411_, lean_object* v_row_1412_){
_start:
{
lean_object* v___f_1413_; 
v___f_1413_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Air_Trace_get___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1413_, 0, v_trace_1411_);
lean_closure_set(v___f_1413_, 1, v_row_1412_);
return v___f_1413_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_get___boxed(lean_object* v_F_1414_, lean_object* v_inst_1415_, lean_object* v_A_1416_, lean_object* v_trace_1417_, lean_object* v_row_1418_){
_start:
{
lean_object* v_res_1419_; 
v_res_1419_ = lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_get(v_F_1414_, v_inst_1415_, v_A_1416_, v_trace_1417_, v_row_1418_);
lean_dec_ref(v_A_1416_);
lean_dec_ref(v_inst_1415_);
return v_res_1419_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_EvalCtx_localRow___redArg(lean_object* v_ctx_1420_){
_start:
{
lean_object* v_trace_1421_; lean_object* v_row_1422_; lean_object* v___f_1423_; 
v_trace_1421_ = lean_ctor_get(v_ctx_1420_, 0);
lean_inc_ref(v_trace_1421_);
v_row_1422_ = lean_ctor_get(v_ctx_1420_, 1);
lean_inc(v_row_1422_);
lean_dec_ref(v_ctx_1420_);
v___f_1423_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Air_Trace_get___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1423_, 0, v_trace_1421_);
lean_closure_set(v___f_1423_, 1, v_row_1422_);
return v___f_1423_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_EvalCtx_localRow(lean_object* v_F_1424_, lean_object* v_inst_1425_, lean_object* v_A_1426_, lean_object* v_ctx_1427_){
_start:
{
lean_object* v___x_1428_; 
v___x_1428_ = lp_swirl_x2dfv_Fundamentals_Air_AIR_EvalCtx_localRow___redArg(v_ctx_1427_);
return v___x_1428_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_EvalCtx_localRow___boxed(lean_object* v_F_1429_, lean_object* v_inst_1430_, lean_object* v_A_1431_, lean_object* v_ctx_1432_){
_start:
{
lean_object* v_res_1433_; 
v_res_1433_ = lp_swirl_x2dfv_Fundamentals_Air_AIR_EvalCtx_localRow(v_F_1429_, v_inst_1430_, v_A_1431_, v_ctx_1432_);
lean_dec_ref(v_A_1431_);
lean_dec_ref(v_inst_1430_);
return v_res_1433_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_nextRowIndex___redArg(lean_object* v_trace_1434_, lean_object* v_i_1435_){
_start:
{
lean_object* v_height_1436_; lean_object* v___x_1437_; lean_object* v___x_1438_; lean_object* v___x_1439_; 
v_height_1436_ = lean_ctor_get(v_trace_1434_, 0);
v___x_1437_ = lean_unsigned_to_nat(1u);
v___x_1438_ = lean_nat_add(v_i_1435_, v___x_1437_);
v___x_1439_ = lean_nat_mod(v___x_1438_, v_height_1436_);
lean_dec(v___x_1438_);
return v___x_1439_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_nextRowIndex___redArg___boxed(lean_object* v_trace_1440_, lean_object* v_i_1441_){
_start:
{
lean_object* v_res_1442_; 
v_res_1442_ = lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_nextRowIndex___redArg(v_trace_1440_, v_i_1441_);
lean_dec(v_i_1441_);
lean_dec_ref(v_trace_1440_);
return v_res_1442_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_nextRowIndex(lean_object* v_F_1443_, lean_object* v_inst_1444_, lean_object* v_A_1445_, lean_object* v_trace_1446_, lean_object* v_i_1447_){
_start:
{
lean_object* v___x_1448_; 
v___x_1448_ = lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_nextRowIndex___redArg(v_trace_1446_, v_i_1447_);
return v___x_1448_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_nextRowIndex___boxed(lean_object* v_F_1449_, lean_object* v_inst_1450_, lean_object* v_A_1451_, lean_object* v_trace_1452_, lean_object* v_i_1453_){
_start:
{
lean_object* v_res_1454_; 
v_res_1454_ = lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_nextRowIndex(v_F_1449_, v_inst_1450_, v_A_1451_, v_trace_1452_, v_i_1453_);
lean_dec(v_i_1453_);
lean_dec_ref(v_trace_1452_);
lean_dec_ref(v_A_1451_);
lean_dec_ref(v_inst_1450_);
return v_res_1454_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_col___redArg(lean_object* v_trace_1455_, lean_object* v_column_1456_, lean_object* v_row_1457_){
_start:
{
lean_object* v___f_1458_; lean_object* v___x_1459_; 
v___f_1458_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Air_Trace_get___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1458_, 0, v_trace_1455_);
lean_closure_set(v___f_1458_, 1, v_row_1457_);
v___x_1459_ = lp_swirl_x2dfv_Fundamentals_Air_Row_get___redArg(v___f_1458_, v_column_1456_);
return v___x_1459_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_col(lean_object* v_F_1460_, lean_object* v_inst_1461_, lean_object* v_A_1462_, lean_object* v_trace_1463_, lean_object* v_column_1464_, lean_object* v_row_1465_){
_start:
{
lean_object* v___x_1466_; 
v___x_1466_ = lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_col___redArg(v_trace_1463_, v_column_1464_, v_row_1465_);
return v___x_1466_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_col___boxed(lean_object* v_F_1467_, lean_object* v_inst_1468_, lean_object* v_A_1469_, lean_object* v_trace_1470_, lean_object* v_column_1471_, lean_object* v_row_1472_){
_start:
{
lean_object* v_res_1473_; 
v_res_1473_ = lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_col(v_F_1467_, v_inst_1468_, v_A_1469_, v_trace_1470_, v_column_1471_, v_row_1472_);
lean_dec_ref(v_A_1469_);
lean_dec_ref(v_inst_1468_);
return v_res_1473_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_colNext___redArg(lean_object* v_trace_1474_, lean_object* v_column_1475_, lean_object* v_row_1476_){
_start:
{
lean_object* v___x_1477_; lean_object* v___f_1478_; lean_object* v___x_1479_; 
v___x_1477_ = lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_nextRowIndex___redArg(v_trace_1474_, v_row_1476_);
v___f_1478_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Air_Trace_get___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1478_, 0, v_trace_1474_);
lean_closure_set(v___f_1478_, 1, v___x_1477_);
v___x_1479_ = lp_swirl_x2dfv_Fundamentals_Air_Row_get___redArg(v___f_1478_, v_column_1475_);
return v___x_1479_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_colNext___redArg___boxed(lean_object* v_trace_1480_, lean_object* v_column_1481_, lean_object* v_row_1482_){
_start:
{
lean_object* v_res_1483_; 
v_res_1483_ = lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_colNext___redArg(v_trace_1480_, v_column_1481_, v_row_1482_);
lean_dec(v_row_1482_);
return v_res_1483_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_colNext(lean_object* v_F_1484_, lean_object* v_inst_1485_, lean_object* v_A_1486_, lean_object* v_trace_1487_, lean_object* v_column_1488_, lean_object* v_row_1489_){
_start:
{
lean_object* v___x_1490_; 
v___x_1490_ = lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_colNext___redArg(v_trace_1487_, v_column_1488_, v_row_1489_);
return v___x_1490_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_colNext___boxed(lean_object* v_F_1491_, lean_object* v_inst_1492_, lean_object* v_A_1493_, lean_object* v_trace_1494_, lean_object* v_column_1495_, lean_object* v_row_1496_){
_start:
{
lean_object* v_res_1497_; 
v_res_1497_ = lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_colNext(v_F_1491_, v_inst_1492_, v_A_1493_, v_trace_1494_, v_column_1495_, v_row_1496_);
lean_dec(v_row_1496_);
lean_dec_ref(v_A_1493_);
lean_dec_ref(v_inst_1492_);
return v_res_1497_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_EvalCtx_nextRow___redArg(lean_object* v_ctx_1498_){
_start:
{
lean_object* v_trace_1499_; lean_object* v_row_1500_; lean_object* v___x_1501_; lean_object* v___f_1502_; 
v_trace_1499_ = lean_ctor_get(v_ctx_1498_, 0);
lean_inc_ref(v_trace_1499_);
v_row_1500_ = lean_ctor_get(v_ctx_1498_, 1);
lean_inc(v_row_1500_);
lean_dec_ref(v_ctx_1498_);
v___x_1501_ = lp_swirl_x2dfv_Fundamentals_Air_AIR_Trace_nextRowIndex___redArg(v_trace_1499_, v_row_1500_);
lean_dec(v_row_1500_);
v___f_1502_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Air_Trace_get___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1502_, 0, v_trace_1499_);
lean_closure_set(v___f_1502_, 1, v___x_1501_);
return v___f_1502_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_EvalCtx_nextRow(lean_object* v_F_1503_, lean_object* v_inst_1504_, lean_object* v_A_1505_, lean_object* v_ctx_1506_){
_start:
{
lean_object* v___x_1507_; 
v___x_1507_ = lp_swirl_x2dfv_Fundamentals_Air_AIR_EvalCtx_nextRow___redArg(v_ctx_1506_);
return v___x_1507_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_EvalCtx_nextRow___boxed(lean_object* v_F_1508_, lean_object* v_inst_1509_, lean_object* v_A_1510_, lean_object* v_ctx_1511_){
_start:
{
lean_object* v_res_1512_; 
v_res_1512_ = lp_swirl_x2dfv_Fundamentals_Air_AIR_EvalCtx_nextRow(v_F_1508_, v_inst_1509_, v_A_1510_, v_ctx_1511_);
lean_dec_ref(v_A_1510_);
lean_dec_ref(v_inst_1509_);
return v_res_1512_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_evalSelector___redArg(lean_object* v_inst_1513_, lean_object* v_ctx_1514_, uint8_t v_x_1515_){
_start:
{
switch(v_x_1515_)
{
case 0:
{
lean_object* v_row_1516_; lean_object* v___x_1517_; uint8_t v___x_1518_; 
v_row_1516_ = lean_ctor_get(v_ctx_1514_, 1);
v___x_1517_ = lean_unsigned_to_nat(0u);
v___x_1518_ = lean_nat_dec_eq(v_row_1516_, v___x_1517_);
if (v___x_1518_ == 0)
{
lean_object* v___x_1519_; lean_object* v_toZero_1520_; 
v___x_1519_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_inst_1513_);
v_toZero_1520_ = lean_ctor_get(v___x_1519_, 1);
lean_inc(v_toZero_1520_);
lean_dec_ref(v___x_1519_);
return v_toZero_1520_;
}
else
{
lean_object* v___x_1521_; lean_object* v___x_1522_; lean_object* v_toOne_1523_; 
v___x_1521_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_1513_);
v___x_1522_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v___x_1521_);
v_toOne_1523_ = lean_ctor_get(v___x_1522_, 2);
lean_inc(v_toOne_1523_);
lean_dec_ref(v___x_1522_);
return v_toOne_1523_;
}
}
case 1:
{
lean_object* v_trace_1524_; lean_object* v_row_1525_; lean_object* v_height_1526_; lean_object* v___x_1527_; lean_object* v___x_1528_; uint8_t v___x_1529_; 
v_trace_1524_ = lean_ctor_get(v_ctx_1514_, 0);
v_row_1525_ = lean_ctor_get(v_ctx_1514_, 1);
v_height_1526_ = lean_ctor_get(v_trace_1524_, 0);
v___x_1527_ = lean_unsigned_to_nat(1u);
v___x_1528_ = lean_nat_add(v_row_1525_, v___x_1527_);
v___x_1529_ = lean_nat_dec_eq(v___x_1528_, v_height_1526_);
lean_dec(v___x_1528_);
if (v___x_1529_ == 0)
{
lean_object* v___x_1530_; lean_object* v_toZero_1531_; 
v___x_1530_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_inst_1513_);
v_toZero_1531_ = lean_ctor_get(v___x_1530_, 1);
lean_inc(v_toZero_1531_);
lean_dec_ref(v___x_1530_);
return v_toZero_1531_;
}
else
{
lean_object* v___x_1532_; lean_object* v___x_1533_; lean_object* v_toOne_1534_; 
v___x_1532_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_1513_);
v___x_1533_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v___x_1532_);
v_toOne_1534_ = lean_ctor_get(v___x_1533_, 2);
lean_inc(v_toOne_1534_);
lean_dec_ref(v___x_1533_);
return v_toOne_1534_;
}
}
default: 
{
lean_object* v_trace_1535_; lean_object* v_row_1536_; lean_object* v_height_1537_; lean_object* v___x_1538_; lean_object* v___x_1539_; uint8_t v___x_1540_; 
v_trace_1535_ = lean_ctor_get(v_ctx_1514_, 0);
v_row_1536_ = lean_ctor_get(v_ctx_1514_, 1);
v_height_1537_ = lean_ctor_get(v_trace_1535_, 0);
v___x_1538_ = lean_unsigned_to_nat(1u);
v___x_1539_ = lean_nat_add(v_row_1536_, v___x_1538_);
v___x_1540_ = lean_nat_dec_eq(v___x_1539_, v_height_1537_);
lean_dec(v___x_1539_);
if (v___x_1540_ == 0)
{
lean_object* v___x_1541_; lean_object* v___x_1542_; lean_object* v_toOne_1543_; 
v___x_1541_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_1513_);
v___x_1542_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v___x_1541_);
v_toOne_1543_ = lean_ctor_get(v___x_1542_, 2);
lean_inc(v_toOne_1543_);
lean_dec_ref(v___x_1542_);
return v_toOne_1543_;
}
else
{
lean_object* v___x_1544_; lean_object* v_toZero_1545_; 
v___x_1544_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_inst_1513_);
v_toZero_1545_ = lean_ctor_get(v___x_1544_, 1);
lean_inc(v_toZero_1545_);
lean_dec_ref(v___x_1544_);
return v_toZero_1545_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_evalSelector___redArg___boxed(lean_object* v_inst_1546_, lean_object* v_ctx_1547_, lean_object* v_x_1548_){
_start:
{
uint8_t v_x_559__boxed_1549_; lean_object* v_res_1550_; 
v_x_559__boxed_1549_ = lean_unbox(v_x_1548_);
v_res_1550_ = lp_swirl_x2dfv_Fundamentals_Air_AIR_evalSelector___redArg(v_inst_1546_, v_ctx_1547_, v_x_559__boxed_1549_);
lean_dec_ref(v_ctx_1547_);
return v_res_1550_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_evalSelector(lean_object* v_F_1551_, lean_object* v_inst_1552_, lean_object* v_A_1553_, lean_object* v_ctx_1554_, uint8_t v_x_1555_){
_start:
{
lean_object* v___x_1556_; 
v___x_1556_ = lp_swirl_x2dfv_Fundamentals_Air_AIR_evalSelector___redArg(v_inst_1552_, v_ctx_1554_, v_x_1555_);
return v___x_1556_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_evalSelector___boxed(lean_object* v_F_1557_, lean_object* v_inst_1558_, lean_object* v_A_1559_, lean_object* v_ctx_1560_, lean_object* v_x_1561_){
_start:
{
uint8_t v_x_599__boxed_1562_; lean_object* v_res_1563_; 
v_x_599__boxed_1562_ = lean_unbox(v_x_1561_);
v_res_1563_ = lp_swirl_x2dfv_Fundamentals_Air_AIR_evalSelector(v_F_1557_, v_inst_1558_, v_A_1559_, v_ctx_1560_, v_x_599__boxed_1562_);
lean_dec_ref(v_ctx_1560_);
lean_dec_ref(v_A_1559_);
return v_res_1563_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_evalVar___redArg(lean_object* v_inst_1564_, lean_object* v_ctx_1565_, lean_object* v_x_1566_){
_start:
{
lean_object* v___x_1567_; 
lean_inc_ref(v_inst_1564_);
v___x_1567_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_inst_1564_);
switch(lean_obj_tag(v_x_1566_))
{
case 0:
{
uint8_t v_a_1568_; lean_object* v___x_1569_; 
lean_dec_ref(v___x_1567_);
v_a_1568_ = lean_ctor_get_uint8(v_x_1566_, 0);
lean_dec_ref_known(v_x_1566_, 0);
v___x_1569_ = lp_swirl_x2dfv_Fundamentals_Air_AIR_evalSelector___redArg(v_inst_1564_, v_ctx_1565_, v_a_1568_);
lean_dec_ref(v_ctx_1565_);
return v___x_1569_;
}
case 1:
{
uint8_t v_a_1570_; 
lean_dec_ref(v___x_1567_);
lean_dec_ref(v_inst_1564_);
v_a_1570_ = lean_ctor_get_uint8(v_x_1566_, sizeof(void*)*1);
if (v_a_1570_ == 0)
{
lean_object* v_a_1571_; lean_object* v___x_1572_; lean_object* v___x_1573_; 
v_a_1571_ = lean_ctor_get(v_x_1566_, 0);
lean_inc_ref(v_a_1571_);
lean_dec_ref_known(v_x_1566_, 1);
v___x_1572_ = lp_swirl_x2dfv_Fundamentals_Air_AIR_EvalCtx_localRow___redArg(v_ctx_1565_);
v___x_1573_ = lp_swirl_x2dfv_Fundamentals_Air_Row_get___redArg(v___x_1572_, v_a_1571_);
return v___x_1573_;
}
else
{
lean_object* v_a_1574_; lean_object* v___x_1575_; lean_object* v___x_1576_; 
v_a_1574_ = lean_ctor_get(v_x_1566_, 0);
lean_inc_ref(v_a_1574_);
lean_dec_ref_known(v_x_1566_, 1);
v___x_1575_ = lp_swirl_x2dfv_Fundamentals_Air_AIR_EvalCtx_nextRow___redArg(v_ctx_1565_);
v___x_1576_ = lp_swirl_x2dfv_Fundamentals_Air_Row_get___redArg(v___x_1575_, v_a_1574_);
return v___x_1576_;
}
}
default: 
{
lean_object* v_toZero_1577_; lean_object* v_a_1578_; lean_object* v_publicValues_1579_; lean_object* v___x_1580_; 
lean_dec_ref(v_inst_1564_);
v_toZero_1577_ = lean_ctor_get(v___x_1567_, 1);
lean_inc(v_toZero_1577_);
lean_dec_ref(v___x_1567_);
v_a_1578_ = lean_ctor_get(v_x_1566_, 0);
lean_inc(v_a_1578_);
lean_dec_ref_known(v_x_1566_, 1);
v_publicValues_1579_ = lean_ctor_get(v_ctx_1565_, 2);
lean_inc(v_publicValues_1579_);
lean_dec_ref(v_ctx_1565_);
v___x_1580_ = l_List_getD___redArg(v_publicValues_1579_, v_a_1578_, v_toZero_1577_);
lean_dec(v_toZero_1577_);
lean_dec(v_publicValues_1579_);
return v___x_1580_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_evalVar(lean_object* v_F_1581_, lean_object* v_inst_1582_, lean_object* v_A_1583_, lean_object* v_ctx_1584_, lean_object* v_x_1585_){
_start:
{
lean_object* v___x_1586_; 
v___x_1586_ = lp_swirl_x2dfv_Fundamentals_Air_AIR_evalVar___redArg(v_inst_1582_, v_ctx_1584_, v_x_1585_);
return v___x_1586_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_evalVar___boxed(lean_object* v_F_1587_, lean_object* v_inst_1588_, lean_object* v_A_1589_, lean_object* v_ctx_1590_, lean_object* v_x_1591_){
_start:
{
lean_object* v_res_1592_; 
v_res_1592_ = lp_swirl_x2dfv_Fundamentals_Air_AIR_evalVar(v_F_1587_, v_inst_1588_, v_A_1589_, v_ctx_1590_, v_x_1591_);
lean_dec_ref(v_A_1589_);
return v_res_1592_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_Expr_evalAt___redArg(lean_object* v_inst_1593_, lean_object* v_A_1594_, lean_object* v_e_1595_, lean_object* v_ctx_1596_){
_start:
{
lean_object* v___x_1597_; lean_object* v___x_1598_; 
lean_inc_ref(v_inst_1593_);
v___x_1597_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Air_AIR_evalVar___boxed), 5, 4);
lean_closure_set(v___x_1597_, 0, lean_box(0));
lean_closure_set(v___x_1597_, 1, v_inst_1593_);
lean_closure_set(v___x_1597_, 2, v_A_1594_);
lean_closure_set(v___x_1597_, 3, v_ctx_1596_);
v___x_1598_ = lp_swirl_x2dfv_Fundamentals_Air_Expr_eval___redArg(v_inst_1593_, v_e_1595_, v___x_1597_);
return v___x_1598_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AIR_Expr_evalAt(lean_object* v_F_1599_, lean_object* v_inst_1600_, lean_object* v_A_1601_, lean_object* v_e_1602_, lean_object* v_ctx_1603_){
_start:
{
lean_object* v___x_1604_; 
v___x_1604_ = lp_swirl_x2dfv_Fundamentals_Air_AIR_Expr_evalAt___redArg(v_inst_1600_, v_A_1601_, v_e_1602_, v_ctx_1603_);
return v___x_1604_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMessage___redArg___lam__0(lean_object* v_toCommSemiring_1605_, lean_object* v_assignment_1606_, lean_object* v_e_1607_){
_start:
{
lean_object* v___x_1608_; 
v___x_1608_ = lp_swirl_x2dfv_Fundamentals_Air_Expr_eval___redArg(v_toCommSemiring_1605_, v_e_1607_, v_assignment_1606_);
return v___x_1608_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMessage___redArg(lean_object* v_inst_1609_, lean_object* v_I_1610_, lean_object* v_assignment_1611_){
_start:
{
lean_object* v___x_1612_; lean_object* v_toCommSemiring_1613_; lean_object* v_msgExprs_1614_; lean_object* v___f_1615_; lean_object* v___x_1616_; lean_object* v___x_1617_; 
v___x_1612_ = lp_mathlib_Field_toSemifield___redArg(v_inst_1609_);
v_toCommSemiring_1613_ = lean_ctor_get(v___x_1612_, 0);
lean_inc_ref(v_toCommSemiring_1613_);
lean_dec_ref(v___x_1612_);
v_msgExprs_1614_ = lean_ctor_get(v_I_1610_, 1);
lean_inc(v_msgExprs_1614_);
lean_dec_ref(v_I_1610_);
v___f_1615_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMessage___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1615_, 0, v_toCommSemiring_1613_);
lean_closure_set(v___f_1615_, 1, v_assignment_1611_);
v___x_1616_ = lean_box(0);
v___x_1617_ = l_List_mapTR_loop___redArg(v___f_1615_, v_msgExprs_1614_, v___x_1616_);
return v___x_1617_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMessage___redArg___boxed(lean_object* v_inst_1618_, lean_object* v_I_1619_, lean_object* v_assignment_1620_){
_start:
{
lean_object* v_res_1621_; 
v_res_1621_ = lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMessage___redArg(v_inst_1618_, v_I_1619_, v_assignment_1620_);
lean_dec_ref(v_inst_1618_);
return v_res_1621_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMessage(lean_object* v_F_1622_, lean_object* v_inst_1623_, lean_object* v_layout_1624_, lean_object* v_I_1625_, lean_object* v_assignment_1626_){
_start:
{
lean_object* v___x_1627_; 
v___x_1627_ = lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMessage___redArg(v_inst_1623_, v_I_1625_, v_assignment_1626_);
return v___x_1627_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMessage___boxed(lean_object* v_F_1628_, lean_object* v_inst_1629_, lean_object* v_layout_1630_, lean_object* v_I_1631_, lean_object* v_assignment_1632_){
_start:
{
lean_object* v_res_1633_; 
v_res_1633_ = lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMessage(v_F_1628_, v_inst_1629_, v_layout_1630_, v_I_1631_, v_assignment_1632_);
lean_dec_ref(v_layout_1630_);
lean_dec_ref(v_inst_1629_);
return v_res_1633_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMultiplicity___redArg(lean_object* v_inst_1634_, lean_object* v_I_1635_, lean_object* v_assignment_1636_){
_start:
{
lean_object* v___x_1637_; lean_object* v_toCommSemiring_1638_; lean_object* v_multExpr_1639_; lean_object* v___x_1640_; 
v___x_1637_ = lp_mathlib_Field_toSemifield___redArg(v_inst_1634_);
v_toCommSemiring_1638_ = lean_ctor_get(v___x_1637_, 0);
lean_inc_ref(v_toCommSemiring_1638_);
lean_dec_ref(v___x_1637_);
v_multExpr_1639_ = lean_ctor_get(v_I_1635_, 2);
lean_inc_ref(v_multExpr_1639_);
lean_dec_ref(v_I_1635_);
v___x_1640_ = lp_swirl_x2dfv_Fundamentals_Air_Expr_eval___redArg(v_toCommSemiring_1638_, v_multExpr_1639_, v_assignment_1636_);
return v___x_1640_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMultiplicity___redArg___boxed(lean_object* v_inst_1641_, lean_object* v_I_1642_, lean_object* v_assignment_1643_){
_start:
{
lean_object* v_res_1644_; 
v_res_1644_ = lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMultiplicity___redArg(v_inst_1641_, v_I_1642_, v_assignment_1643_);
lean_dec_ref(v_inst_1641_);
return v_res_1644_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMultiplicity(lean_object* v_F_1645_, lean_object* v_inst_1646_, lean_object* v_layout_1647_, lean_object* v_I_1648_, lean_object* v_assignment_1649_){
_start:
{
lean_object* v___x_1650_; 
v___x_1650_ = lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMultiplicity___redArg(v_inst_1646_, v_I_1648_, v_assignment_1649_);
return v___x_1650_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMultiplicity___boxed(lean_object* v_F_1651_, lean_object* v_inst_1652_, lean_object* v_layout_1653_, lean_object* v_I_1654_, lean_object* v_assignment_1655_){
_start:
{
lean_object* v_res_1656_; 
v_res_1656_ = lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMultiplicity(v_F_1651_, v_inst_1652_, v_layout_1653_, v_I_1654_, v_assignment_1655_);
lean_dec_ref(v_layout_1653_);
lean_dec_ref(v_inst_1652_);
return v_res_1656_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMessageAt___redArg(lean_object* v_inst_1657_, lean_object* v_A_1658_, lean_object* v_I_1659_, lean_object* v_ctx_1660_){
_start:
{
lean_object* v___x_1661_; lean_object* v_toCommSemiring_1662_; lean_object* v___x_1663_; lean_object* v___x_1664_; 
v___x_1661_ = lp_mathlib_Field_toSemifield___redArg(v_inst_1657_);
v_toCommSemiring_1662_ = lean_ctor_get(v___x_1661_, 0);
lean_inc_ref(v_toCommSemiring_1662_);
lean_dec_ref(v___x_1661_);
v___x_1663_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Air_AIR_evalVar___boxed), 5, 4);
lean_closure_set(v___x_1663_, 0, lean_box(0));
lean_closure_set(v___x_1663_, 1, v_toCommSemiring_1662_);
lean_closure_set(v___x_1663_, 2, v_A_1658_);
lean_closure_set(v___x_1663_, 3, v_ctx_1660_);
v___x_1664_ = lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMessage___redArg(v_inst_1657_, v_I_1659_, v___x_1663_);
return v___x_1664_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMessageAt___redArg___boxed(lean_object* v_inst_1665_, lean_object* v_A_1666_, lean_object* v_I_1667_, lean_object* v_ctx_1668_){
_start:
{
lean_object* v_res_1669_; 
v_res_1669_ = lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMessageAt___redArg(v_inst_1665_, v_A_1666_, v_I_1667_, v_ctx_1668_);
lean_dec_ref(v_inst_1665_);
return v_res_1669_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMessageAt(lean_object* v_F_1670_, lean_object* v_inst_1671_, lean_object* v_A_1672_, lean_object* v_I_1673_, lean_object* v_ctx_1674_){
_start:
{
lean_object* v___x_1675_; 
v___x_1675_ = lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMessageAt___redArg(v_inst_1671_, v_A_1672_, v_I_1673_, v_ctx_1674_);
return v___x_1675_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMessageAt___boxed(lean_object* v_F_1676_, lean_object* v_inst_1677_, lean_object* v_A_1678_, lean_object* v_I_1679_, lean_object* v_ctx_1680_){
_start:
{
lean_object* v_res_1681_; 
v_res_1681_ = lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMessageAt(v_F_1676_, v_inst_1677_, v_A_1678_, v_I_1679_, v_ctx_1680_);
lean_dec_ref(v_inst_1677_);
return v_res_1681_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMultiplicityAt___redArg(lean_object* v_inst_1682_, lean_object* v_A_1683_, lean_object* v_I_1684_, lean_object* v_ctx_1685_){
_start:
{
lean_object* v___x_1686_; lean_object* v_toCommSemiring_1687_; lean_object* v___x_1688_; lean_object* v___x_1689_; 
v___x_1686_ = lp_mathlib_Field_toSemifield___redArg(v_inst_1682_);
v_toCommSemiring_1687_ = lean_ctor_get(v___x_1686_, 0);
lean_inc_ref(v_toCommSemiring_1687_);
lean_dec_ref(v___x_1686_);
v___x_1688_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Air_AIR_evalVar___boxed), 5, 4);
lean_closure_set(v___x_1688_, 0, lean_box(0));
lean_closure_set(v___x_1688_, 1, v_toCommSemiring_1687_);
lean_closure_set(v___x_1688_, 2, v_A_1683_);
lean_closure_set(v___x_1688_, 3, v_ctx_1685_);
v___x_1689_ = lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMultiplicity___redArg(v_inst_1682_, v_I_1684_, v___x_1688_);
return v___x_1689_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMultiplicityAt___redArg___boxed(lean_object* v_inst_1690_, lean_object* v_A_1691_, lean_object* v_I_1692_, lean_object* v_ctx_1693_){
_start:
{
lean_object* v_res_1694_; 
v_res_1694_ = lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMultiplicityAt___redArg(v_inst_1690_, v_A_1691_, v_I_1692_, v_ctx_1693_);
lean_dec_ref(v_inst_1690_);
return v_res_1694_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMultiplicityAt(lean_object* v_F_1695_, lean_object* v_inst_1696_, lean_object* v_A_1697_, lean_object* v_I_1698_, lean_object* v_ctx_1699_){
_start:
{
lean_object* v___x_1700_; 
v___x_1700_ = lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMultiplicityAt___redArg(v_inst_1696_, v_A_1697_, v_I_1698_, v_ctx_1699_);
return v___x_1700_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMultiplicityAt___boxed(lean_object* v_F_1701_, lean_object* v_inst_1702_, lean_object* v_A_1703_, lean_object* v_I_1704_, lean_object* v_ctx_1705_){
_start:
{
lean_object* v_res_1706_; 
v_res_1706_ = lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMultiplicityAt(v_F_1701_, v_inst_1702_, v_A_1703_, v_I_1704_, v_ctx_1705_);
lean_dec_ref(v_inst_1702_);
return v_res_1706_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_toBusEventAt___redArg(lean_object* v_inst_1707_, lean_object* v_A_1708_, lean_object* v_I_1709_, lean_object* v_ctx_1710_){
_start:
{
lean_object* v___x_1711_; lean_object* v___x_1712_; lean_object* v___x_1713_; 
lean_inc_ref(v_ctx_1710_);
lean_inc_ref(v_I_1709_);
lean_inc_ref(v_A_1708_);
v___x_1711_ = lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMultiplicityAt___redArg(v_inst_1707_, v_A_1708_, v_I_1709_, v_ctx_1710_);
v___x_1712_ = lp_swirl_x2dfv_Fundamentals_Air_Interaction_evalMessageAt___redArg(v_inst_1707_, v_A_1708_, v_I_1709_, v_ctx_1710_);
v___x_1713_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1713_, 0, v___x_1711_);
lean_ctor_set(v___x_1713_, 1, v___x_1712_);
return v___x_1713_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_toBusEventAt___redArg___boxed(lean_object* v_inst_1714_, lean_object* v_A_1715_, lean_object* v_I_1716_, lean_object* v_ctx_1717_){
_start:
{
lean_object* v_res_1718_; 
v_res_1718_ = lp_swirl_x2dfv_Fundamentals_Air_Interaction_toBusEventAt___redArg(v_inst_1714_, v_A_1715_, v_I_1716_, v_ctx_1717_);
lean_dec_ref(v_inst_1714_);
return v_res_1718_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_toBusEventAt(lean_object* v_F_1719_, lean_object* v_inst_1720_, lean_object* v_A_1721_, lean_object* v_I_1722_, lean_object* v_ctx_1723_){
_start:
{
lean_object* v___x_1724_; 
v___x_1724_ = lp_swirl_x2dfv_Fundamentals_Air_Interaction_toBusEventAt___redArg(v_inst_1720_, v_A_1721_, v_I_1722_, v_ctx_1723_);
return v___x_1724_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_toBusEventAt___boxed(lean_object* v_F_1725_, lean_object* v_inst_1726_, lean_object* v_A_1727_, lean_object* v_I_1728_, lean_object* v_ctx_1729_){
_start:
{
lean_object* v_res_1730_; 
v_res_1730_ = lp_swirl_x2dfv_Fundamentals_Air_Interaction_toBusEventAt(v_F_1725_, v_inst_1726_, v_A_1727_, v_I_1728_, v_ctx_1729_);
lean_dec_ref(v_inst_1726_);
return v_res_1730_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_toIndexedBusEventAt___redArg(lean_object* v_inst_1731_, lean_object* v_A_1732_, lean_object* v_I_1733_, lean_object* v_ctx_1734_){
_start:
{
lean_object* v_bus_1735_; lean_object* v___x_1736_; lean_object* v___x_1737_; 
v_bus_1735_ = lean_ctor_get(v_I_1733_, 0);
lean_inc(v_bus_1735_);
v___x_1736_ = lp_swirl_x2dfv_Fundamentals_Air_Interaction_toBusEventAt___redArg(v_inst_1731_, v_A_1732_, v_I_1733_, v_ctx_1734_);
v___x_1737_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1737_, 0, v_bus_1735_);
lean_ctor_set(v___x_1737_, 1, v___x_1736_);
return v___x_1737_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_toIndexedBusEventAt___redArg___boxed(lean_object* v_inst_1738_, lean_object* v_A_1739_, lean_object* v_I_1740_, lean_object* v_ctx_1741_){
_start:
{
lean_object* v_res_1742_; 
v_res_1742_ = lp_swirl_x2dfv_Fundamentals_Air_Interaction_toIndexedBusEventAt___redArg(v_inst_1738_, v_A_1739_, v_I_1740_, v_ctx_1741_);
lean_dec_ref(v_inst_1738_);
return v_res_1742_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_toIndexedBusEventAt(lean_object* v_F_1743_, lean_object* v_inst_1744_, lean_object* v_A_1745_, lean_object* v_I_1746_, lean_object* v_ctx_1747_){
_start:
{
lean_object* v___x_1748_; 
v___x_1748_ = lp_swirl_x2dfv_Fundamentals_Air_Interaction_toIndexedBusEventAt___redArg(v_inst_1744_, v_A_1745_, v_I_1746_, v_ctx_1747_);
return v___x_1748_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_Interaction_toIndexedBusEventAt___boxed(lean_object* v_F_1749_, lean_object* v_inst_1750_, lean_object* v_A_1751_, lean_object* v_I_1752_, lean_object* v_ctx_1753_){
_start:
{
lean_object* v_res_1754_; 
v_res_1754_ = lp_swirl_x2dfv_Fundamentals_Air_Interaction_toIndexedBusEventAt(v_F_1749_, v_inst_1750_, v_A_1751_, v_I_1752_, v_ctx_1753_);
lean_dec_ref(v_inst_1750_);
return v_res_1754_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_eventsAt___redArg___lam__0(lean_object* v_trace_1755_, lean_object* v_row_1756_, lean_object* v_publicValues_1757_, lean_object* v_inst_1758_, lean_object* v_air_1759_, lean_object* v_I_1760_){
_start:
{
lean_object* v___x_1761_; lean_object* v___x_1762_; 
v___x_1761_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1761_, 0, v_trace_1755_);
lean_ctor_set(v___x_1761_, 1, v_row_1756_);
lean_ctor_set(v___x_1761_, 2, v_publicValues_1757_);
v___x_1762_ = lp_swirl_x2dfv_Fundamentals_Air_Interaction_toIndexedBusEventAt___redArg(v_inst_1758_, v_air_1759_, v_I_1760_, v___x_1761_);
return v___x_1762_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_eventsAt___redArg___lam__0___boxed(lean_object* v_trace_1763_, lean_object* v_row_1764_, lean_object* v_publicValues_1765_, lean_object* v_inst_1766_, lean_object* v_air_1767_, lean_object* v_I_1768_){
_start:
{
lean_object* v_res_1769_; 
v_res_1769_ = lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_eventsAt___redArg___lam__0(v_trace_1763_, v_row_1764_, v_publicValues_1765_, v_inst_1766_, v_air_1767_, v_I_1768_);
lean_dec_ref(v_inst_1766_);
return v_res_1769_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_eventsAt___redArg(lean_object* v_inst_1770_, lean_object* v_A_1771_, lean_object* v_trace_1772_, lean_object* v_row_1773_, lean_object* v_publicValues_1774_){
_start:
{
lean_object* v_air_1775_; lean_object* v_interactions_1776_; lean_object* v___f_1777_; lean_object* v___x_1778_; lean_object* v___x_1779_; 
v_air_1775_ = lean_ctor_get(v_A_1771_, 0);
lean_inc_ref(v_air_1775_);
v_interactions_1776_ = lean_ctor_get(v_A_1771_, 1);
lean_inc(v_interactions_1776_);
lean_dec_ref(v_A_1771_);
v___f_1777_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_eventsAt___redArg___lam__0___boxed), 6, 5);
lean_closure_set(v___f_1777_, 0, v_trace_1772_);
lean_closure_set(v___f_1777_, 1, v_row_1773_);
lean_closure_set(v___f_1777_, 2, v_publicValues_1774_);
lean_closure_set(v___f_1777_, 3, v_inst_1770_);
lean_closure_set(v___f_1777_, 4, v_air_1775_);
v___x_1778_ = lean_box(0);
v___x_1779_ = l_List_mapTR_loop___redArg(v___f_1777_, v_interactions_1776_, v___x_1778_);
return v___x_1779_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_eventsAt(lean_object* v_F_1780_, lean_object* v_inst_1781_, lean_object* v_A_1782_, lean_object* v_trace_1783_, lean_object* v_row_1784_, lean_object* v_publicValues_1785_){
_start:
{
lean_object* v___x_1786_; 
v___x_1786_ = lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_eventsAt___redArg(v_inst_1781_, v_A_1782_, v_trace_1783_, v_row_1784_, v_publicValues_1785_);
return v___x_1786_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_events___redArg___lam__0(lean_object* v_inst_1787_, lean_object* v_A_1788_, lean_object* v_trace_1789_, lean_object* v_publicValues_1790_, lean_object* v_row_1791_){
_start:
{
lean_object* v___x_1792_; 
v___x_1792_ = lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_eventsAt___redArg(v_inst_1787_, v_A_1788_, v_trace_1789_, v_row_1791_, v_publicValues_1790_);
return v___x_1792_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_events___redArg(lean_object* v_inst_1795_, lean_object* v_A_1796_, lean_object* v_trace_1797_, lean_object* v_publicValues_1798_){
_start:
{
lean_object* v_height_1799_; lean_object* v___f_1800_; lean_object* v___x_1801_; lean_object* v___x_1802_; lean_object* v___x_1803_; 
v_height_1799_ = lean_ctor_get(v_trace_1797_, 0);
lean_inc(v_height_1799_);
v___f_1800_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_events___redArg___lam__0), 5, 4);
lean_closure_set(v___f_1800_, 0, v_inst_1795_);
lean_closure_set(v___f_1800_, 1, v_A_1796_);
lean_closure_set(v___f_1800_, 2, v_trace_1797_);
lean_closure_set(v___f_1800_, 3, v_publicValues_1798_);
v___x_1801_ = l_List_finRange(v_height_1799_);
v___x_1802_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_events___redArg___closed__0));
v___x_1803_ = l___private_Init_Data_List_Impl_0__List_flatMapTR_go___redArg(v___f_1800_, v___x_1801_, v___x_1802_);
return v___x_1803_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_events(lean_object* v_F_1804_, lean_object* v_inst_1805_, lean_object* v_A_1806_, lean_object* v_trace_1807_, lean_object* v_publicValues_1808_){
_start:
{
lean_object* v___x_1809_; 
v___x_1809_ = lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_events___redArg(v_inst_1805_, v_A_1806_, v_trace_1807_, v_publicValues_1808_);
return v___x_1809_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_eventsAtForBus___redArg___lam__0(lean_object* v_b_1810_, lean_object* v_event_1811_){
_start:
{
lean_object* v_fst_1812_; lean_object* v_snd_1813_; uint8_t v___x_1814_; 
v_fst_1812_ = lean_ctor_get(v_event_1811_, 0);
v_snd_1813_ = lean_ctor_get(v_event_1811_, 1);
v___x_1814_ = lean_nat_dec_eq(v_fst_1812_, v_b_1810_);
if (v___x_1814_ == 0)
{
lean_object* v___x_1815_; 
v___x_1815_ = lean_box(0);
return v___x_1815_;
}
else
{
lean_object* v___x_1816_; 
lean_inc(v_snd_1813_);
v___x_1816_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1816_, 0, v_snd_1813_);
return v___x_1816_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_eventsAtForBus___redArg___lam__0___boxed(lean_object* v_b_1817_, lean_object* v_event_1818_){
_start:
{
lean_object* v_res_1819_; 
v_res_1819_ = lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_eventsAtForBus___redArg___lam__0(v_b_1817_, v_event_1818_);
lean_dec_ref(v_event_1818_);
lean_dec(v_b_1817_);
return v_res_1819_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_eventsAtForBus___redArg(lean_object* v_inst_1822_, lean_object* v_A_1823_, lean_object* v_b_1824_, lean_object* v_trace_1825_, lean_object* v_row_1826_, lean_object* v_publicValues_1827_){
_start:
{
lean_object* v___f_1828_; lean_object* v___x_1829_; lean_object* v___x_1830_; lean_object* v___x_1831_; 
v___f_1828_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_eventsAtForBus___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1828_, 0, v_b_1824_);
v___x_1829_ = lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_eventsAt___redArg(v_inst_1822_, v_A_1823_, v_trace_1825_, v_row_1826_, v_publicValues_1827_);
v___x_1830_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_eventsAtForBus___redArg___closed__0));
v___x_1831_ = l_List_filterMapTR_go___redArg(v___f_1828_, v___x_1829_, v___x_1830_);
return v___x_1831_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_eventsAtForBus(lean_object* v_F_1832_, lean_object* v_inst_1833_, lean_object* v_A_1834_, lean_object* v_b_1835_, lean_object* v_trace_1836_, lean_object* v_row_1837_, lean_object* v_publicValues_1838_){
_start:
{
lean_object* v___x_1839_; 
v___x_1839_ = lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_eventsAtForBus___redArg(v_inst_1833_, v_A_1834_, v_b_1835_, v_trace_1836_, v_row_1837_, v_publicValues_1838_);
return v___x_1839_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_eventsForBus___redArg(lean_object* v_inst_1840_, lean_object* v_A_1841_, lean_object* v_b_1842_, lean_object* v_trace_1843_, lean_object* v_publicValues_1844_){
_start:
{
lean_object* v___f_1845_; lean_object* v___x_1846_; lean_object* v___x_1847_; lean_object* v___x_1848_; 
v___f_1845_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_eventsAtForBus___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1845_, 0, v_b_1842_);
v___x_1846_ = lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_events___redArg(v_inst_1840_, v_A_1841_, v_trace_1843_, v_publicValues_1844_);
v___x_1847_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_eventsAtForBus___redArg___closed__0));
v___x_1848_ = l_List_filterMapTR_go___redArg(v___f_1845_, v___x_1846_, v___x_1847_);
return v___x_1848_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_eventsForBus(lean_object* v_F_1849_, lean_object* v_inst_1850_, lean_object* v_A_1851_, lean_object* v_b_1852_, lean_object* v_trace_1853_, lean_object* v_publicValues_1854_){
_start:
{
lean_object* v___x_1855_; 
v___x_1855_ = lp_swirl_x2dfv_Fundamentals_Air_AirWithInteractions_eventsForBus___redArg(v_inst_1850_, v_A_1851_, v_b_1852_, v_trace_1853_, v_publicValues_1854_);
return v___x_1855_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_MvPolynomial_CommRing(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Vector_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_MvPolynomial_WeightedHomogeneous(uint8_t builtin);
lean_object* initialize_swirl_x2dfv_Fundamentals_Spec_Bus_Core(uint8_t builtin);
void lean_initialize();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_swirl_x2dfv_Fundamentals_Spec_Air(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
lean_initialize();
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_MvPolynomial_CommRing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Vector_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_MvPolynomial_WeightedHomogeneous(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2dfv_Fundamentals_Spec_Bus_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
