// Lean compiler output
// Module: Mathlib.Order.ConditionallyCompleteLattice.Basic
// Imports: public import Init public meta import Init public import Mathlib.Data.Set.Lattice.Indexed public import Mathlib.Order.ConditionallyCompleteLattice.Defs public import Mathlib.Order.ConditionallyCompletePartialOrder.Basic
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
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lattice_toSemilatticeInf___redArg(lean_object*);
lean_object* lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(lean_object*);
lean_object* lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(lean_object*);
lean_object* lp_mathlib_SemilatticeInf_toMin___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SemilatticeSup_toMax___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_ConditionallyCompletePartialOrder_toConditionallyCompletePartialOrderInf___redArg(lean_object*);
lean_object* lp_mathlib_Pi_instLattice___redArg(lean_object*);
lean_object* lp_mathlib_Pi_supSet___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_OrderDual_instLattice___redArg(lean_object*);
lean_object* lp_mathlib_OrderDual_supSet___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_OrderDual_instLinearOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConditionallyCompleteLinearOrder_toLinearOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConditionallyCompleteLinearOrder_toLinearOrder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLattice_toConditionallyCompleteLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLinearOrder_toConditionallyCompleteLinearOrderBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLinearOrder_toConditionallyCompleteLinearOrderBot(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompleteLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompleteLattice(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_isLUB__csSup___auto__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_isLUB__csSup___auto__1___closed__0 = (const lean_object*)&lp_mathlib_isLUB__csSup___auto__1___closed__0_value;
static const lean_string_object lp_mathlib_isLUB__csSup___auto__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_isLUB__csSup___auto__1___closed__1 = (const lean_object*)&lp_mathlib_isLUB__csSup___auto__1___closed__1_value;
static const lean_string_object lp_mathlib_isLUB__csSup___auto__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_isLUB__csSup___auto__1___closed__2 = (const lean_object*)&lp_mathlib_isLUB__csSup___auto__1___closed__2_value;
static const lean_string_object lp_mathlib_isLUB__csSup___auto__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_isLUB__csSup___auto__1___closed__3 = (const lean_object*)&lp_mathlib_isLUB__csSup___auto__1___closed__3_value;
static const lean_ctor_object lp_mathlib_isLUB__csSup___auto__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_isLUB__csSup___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_isLUB__csSup___auto__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_isLUB__csSup___auto__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_isLUB__csSup___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_isLUB__csSup___auto__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_isLUB__csSup___auto__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_isLUB__csSup___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_isLUB__csSup___auto__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_isLUB__csSup___auto__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_isLUB__csSup___auto__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_isLUB__csSup___auto__1___closed__4 = (const lean_object*)&lp_mathlib_isLUB__csSup___auto__1___closed__4_value;
static const lean_array_object lp_mathlib_isLUB__csSup___auto__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_isLUB__csSup___auto__1___closed__5 = (const lean_object*)&lp_mathlib_isLUB__csSup___auto__1___closed__5_value;
static const lean_string_object lp_mathlib_isLUB__csSup___auto__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_isLUB__csSup___auto__1___closed__6 = (const lean_object*)&lp_mathlib_isLUB__csSup___auto__1___closed__6_value;
static const lean_ctor_object lp_mathlib_isLUB__csSup___auto__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_isLUB__csSup___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_isLUB__csSup___auto__1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_isLUB__csSup___auto__1___closed__7_value_aux_0),((lean_object*)&lp_mathlib_isLUB__csSup___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_isLUB__csSup___auto__1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_isLUB__csSup___auto__1___closed__7_value_aux_1),((lean_object*)&lp_mathlib_isLUB__csSup___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_isLUB__csSup___auto__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_isLUB__csSup___auto__1___closed__7_value_aux_2),((lean_object*)&lp_mathlib_isLUB__csSup___auto__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_isLUB__csSup___auto__1___closed__7 = (const lean_object*)&lp_mathlib_isLUB__csSup___auto__1___closed__7_value;
static const lean_string_object lp_mathlib_isLUB__csSup___auto__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_isLUB__csSup___auto__1___closed__8 = (const lean_object*)&lp_mathlib_isLUB__csSup___auto__1___closed__8_value;
static const lean_ctor_object lp_mathlib_isLUB__csSup___auto__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_isLUB__csSup___auto__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_isLUB__csSup___auto__1___closed__9 = (const lean_object*)&lp_mathlib_isLUB__csSup___auto__1___closed__9_value;
static const lean_string_object lp_mathlib_isLUB__csSup___auto__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "tacticBddDefault"};
static const lean_object* lp_mathlib_isLUB__csSup___auto__1___closed__10 = (const lean_object*)&lp_mathlib_isLUB__csSup___auto__1___closed__10_value;
static const lean_ctor_object lp_mathlib_isLUB__csSup___auto__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_isLUB__csSup___auto__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(253, 17, 67, 45, 28, 85, 122, 106)}};
static const lean_object* lp_mathlib_isLUB__csSup___auto__1___closed__11 = (const lean_object*)&lp_mathlib_isLUB__csSup___auto__1___closed__11_value;
static const lean_string_object lp_mathlib_isLUB__csSup___auto__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "bddDefault"};
static const lean_object* lp_mathlib_isLUB__csSup___auto__1___closed__12 = (const lean_object*)&lp_mathlib_isLUB__csSup___auto__1___closed__12_value;
static lean_once_cell_t lp_mathlib_isLUB__csSup___auto__1___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_isLUB__csSup___auto__1___closed__13;
static lean_once_cell_t lp_mathlib_isLUB__csSup___auto__1___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_isLUB__csSup___auto__1___closed__14;
static lean_once_cell_t lp_mathlib_isLUB__csSup___auto__1___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_isLUB__csSup___auto__1___closed__15;
static lean_once_cell_t lp_mathlib_isLUB__csSup___auto__1___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_isLUB__csSup___auto__1___closed__16;
static lean_once_cell_t lp_mathlib_isLUB__csSup___auto__1___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_isLUB__csSup___auto__1___closed__17;
static lean_once_cell_t lp_mathlib_isLUB__csSup___auto__1___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_isLUB__csSup___auto__1___closed__18;
static lean_once_cell_t lp_mathlib_isLUB__csSup___auto__1___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_isLUB__csSup___auto__1___closed__19;
static lean_once_cell_t lp_mathlib_isLUB__csSup___auto__1___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_isLUB__csSup___auto__1___closed__20;
static lean_once_cell_t lp_mathlib_isLUB__csSup___auto__1___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_isLUB__csSup___auto__1___closed__21;
LEAN_EXPORT lean_object* lp_mathlib_isLUB__csSup___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_isGLB__csInf___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_ConditionallyCompleteLattice_toConditionallyCompletePartialOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConditionallyCompleteLattice_toConditionallyCompletePartialOrder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_csInf__le__csSup___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_csInf__le__csSup___auto__3;
LEAN_EXPORT lean_object* lp_mathlib_csInf__le__csSup__of__nonempty__inter___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_csInf__le__csSup__of__nonempty__inter___auto__3;
LEAN_EXPORT lean_object* lp_mathlib_Pi_conditionallyCompleteLattice___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_conditionallyCompleteLattice___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_conditionallyCompleteLattice___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_conditionallyCompleteLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_conditionallyCompleteLattice(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_csSup__union_x27___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_csSup__union_x27___auto__3;
LEAN_EXPORT lean_object* lp_mathlib_csSup__inter__le_x27___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_csSup__inter__le_x27___auto__3;
LEAN_EXPORT lean_object* lp_mathlib_csSup__insert_x27___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Monotone_csSup__image__le__map__csSup___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Monotone_map__csInf__le__csInf__image___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_ConditionallyCompleteLinearOrder_toLinearOrder___redArg(lean_object* v_h_1_){
_start:
{
lean_object* v_toConditionallyCompleteLattice_2_; lean_object* v_toLattice_3_; lean_object* v_toSemilatticeSup_4_; lean_object* v_toOrd_5_; lean_object* v_toDecidableLE_6_; lean_object* v_toDecidableEq_7_; lean_object* v_toDecidableLT_8_; lean_object* v_toPartialOrder_9_; lean_object* v___x_10_; lean_object* v___f_11_; lean_object* v___f_12_; lean_object* v___x_13_; 
v_toConditionallyCompleteLattice_2_ = lean_ctor_get(v_h_1_, 0);
v_toLattice_3_ = lean_ctor_get(v_toConditionallyCompleteLattice_2_, 0);
lean_inc_ref(v_toLattice_3_);
v_toSemilatticeSup_4_ = lean_ctor_get(v_toLattice_3_, 0);
lean_inc_ref(v_toSemilatticeSup_4_);
v_toOrd_5_ = lean_ctor_get(v_h_1_, 1);
lean_inc_ref(v_toOrd_5_);
v_toDecidableLE_6_ = lean_ctor_get(v_h_1_, 2);
lean_inc_ref(v_toDecidableLE_6_);
v_toDecidableEq_7_ = lean_ctor_get(v_h_1_, 3);
lean_inc_ref(v_toDecidableEq_7_);
v_toDecidableLT_8_ = lean_ctor_get(v_h_1_, 4);
lean_inc_ref(v_toDecidableLT_8_);
lean_dec_ref(v_h_1_);
v_toPartialOrder_9_ = lean_ctor_get(v_toSemilatticeSup_4_, 0);
lean_inc_ref(v_toPartialOrder_9_);
v___x_10_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_toLattice_3_);
v___f_11_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_11_, 0, v___x_10_);
v___f_12_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toMax___redArg___lam__0), 3, 1);
lean_closure_set(v___f_12_, 0, v_toSemilatticeSup_4_);
v___x_13_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_13_, 0, v_toPartialOrder_9_);
lean_ctor_set(v___x_13_, 1, v___f_11_);
lean_ctor_set(v___x_13_, 2, v___f_12_);
lean_ctor_set(v___x_13_, 3, v_toOrd_5_);
lean_ctor_set(v___x_13_, 4, v_toDecidableLE_6_);
lean_ctor_set(v___x_13_, 5, v_toDecidableEq_7_);
lean_ctor_set(v___x_13_, 6, v_toDecidableLT_8_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConditionallyCompleteLinearOrder_toLinearOrder(lean_object* v_00_u03b1_14_, lean_object* v_h_15_){
_start:
{
lean_object* v___x_16_; 
v___x_16_ = lp_mathlib_ConditionallyCompleteLinearOrder_toLinearOrder___redArg(v_h_15_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(lean_object* v_inst_17_){
_start:
{
lean_object* v_toLattice_18_; lean_object* v___x_19_; lean_object* v_toSupSet_20_; lean_object* v___x_21_; lean_object* v_toInfSet_22_; lean_object* v___x_23_; 
v_toLattice_18_ = lean_ctor_get(v_inst_17_, 0);
lean_inc_ref(v_toLattice_18_);
lean_inc_ref(v_inst_17_);
v___x_19_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_inst_17_);
v_toSupSet_20_ = lean_ctor_get(v___x_19_, 1);
lean_inc(v_toSupSet_20_);
lean_dec_ref(v___x_19_);
v___x_21_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_inst_17_);
v_toInfSet_22_ = lean_ctor_get(v___x_21_, 1);
lean_inc(v_toInfSet_22_);
lean_dec_ref(v___x_21_);
v___x_23_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_23_, 0, v_toLattice_18_);
lean_ctor_set(v___x_23_, 1, v_toSupSet_20_);
lean_ctor_set(v___x_23_, 2, v_toInfSet_22_);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLattice_toConditionallyCompleteLattice(lean_object* v_00_u03b1_24_, lean_object* v_inst_25_){
_start:
{
lean_object* v___x_26_; 
v___x_26_ = lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(v_inst_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLinearOrder_toConditionallyCompleteLinearOrderBot___redArg(lean_object* v_h_27_){
_start:
{
lean_object* v_toCompleteLattice_28_; lean_object* v_toOrd_29_; lean_object* v_toDecidableLE_30_; lean_object* v_toDecidableEq_31_; lean_object* v_toDecidableLT_32_; lean_object* v___x_33_; lean_object* v___x_34_; lean_object* v_toBoundedOrder_35_; lean_object* v_toOrderBot_36_; lean_object* v___x_38_; uint8_t v_isShared_39_; uint8_t v_isSharedCheck_43_; 
v_toCompleteLattice_28_ = lean_ctor_get(v_h_27_, 0);
lean_inc_ref_n(v_toCompleteLattice_28_, 2);
v_toOrd_29_ = lean_ctor_get(v_h_27_, 5);
lean_inc_ref(v_toOrd_29_);
v_toDecidableLE_30_ = lean_ctor_get(v_h_27_, 6);
lean_inc_ref(v_toDecidableLE_30_);
v_toDecidableEq_31_ = lean_ctor_get(v_h_27_, 7);
lean_inc_ref(v_toDecidableEq_31_);
v_toDecidableLT_32_ = lean_ctor_get(v_h_27_, 8);
lean_inc_ref(v_toDecidableLT_32_);
lean_dec_ref(v_h_27_);
v___x_33_ = lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(v_toCompleteLattice_28_);
v___x_34_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_34_, 0, v___x_33_);
lean_ctor_set(v___x_34_, 1, v_toOrd_29_);
lean_ctor_set(v___x_34_, 2, v_toDecidableLE_30_);
lean_ctor_set(v___x_34_, 3, v_toDecidableEq_31_);
lean_ctor_set(v___x_34_, 4, v_toDecidableLT_32_);
v_toBoundedOrder_35_ = lean_ctor_get(v_toCompleteLattice_28_, 3);
lean_inc_ref(v_toBoundedOrder_35_);
lean_dec_ref(v_toCompleteLattice_28_);
v_toOrderBot_36_ = lean_ctor_get(v_toBoundedOrder_35_, 1);
v_isSharedCheck_43_ = !lean_is_exclusive(v_toBoundedOrder_35_);
if (v_isSharedCheck_43_ == 0)
{
lean_object* v_unused_44_; 
v_unused_44_ = lean_ctor_get(v_toBoundedOrder_35_, 0);
lean_dec(v_unused_44_);
v___x_38_ = v_toBoundedOrder_35_;
v_isShared_39_ = v_isSharedCheck_43_;
goto v_resetjp_37_;
}
else
{
lean_inc(v_toOrderBot_36_);
lean_dec(v_toBoundedOrder_35_);
v___x_38_ = lean_box(0);
v_isShared_39_ = v_isSharedCheck_43_;
goto v_resetjp_37_;
}
v_resetjp_37_:
{
lean_object* v___x_41_; 
if (v_isShared_39_ == 0)
{
lean_ctor_set(v___x_38_, 0, v___x_34_);
v___x_41_ = v___x_38_;
goto v_reusejp_40_;
}
else
{
lean_object* v_reuseFailAlloc_42_; 
v_reuseFailAlloc_42_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_42_, 0, v___x_34_);
lean_ctor_set(v_reuseFailAlloc_42_, 1, v_toOrderBot_36_);
v___x_41_ = v_reuseFailAlloc_42_;
goto v_reusejp_40_;
}
v_reusejp_40_:
{
return v___x_41_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLinearOrder_toConditionallyCompleteLinearOrderBot(lean_object* v_00_u03b1_45_, lean_object* v_h_46_){
_start:
{
lean_object* v___x_47_; 
v___x_47_ = lp_mathlib_CompleteLinearOrder_toConditionallyCompleteLinearOrderBot___redArg(v_h_46_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompleteLattice___redArg(lean_object* v_inst_48_){
_start:
{
lean_object* v_toLattice_49_; lean_object* v_toSupSet_50_; lean_object* v_toInfSet_51_; lean_object* v___x_53_; uint8_t v_isShared_54_; uint8_t v_isSharedCheck_61_; 
v_toLattice_49_ = lean_ctor_get(v_inst_48_, 0);
v_toSupSet_50_ = lean_ctor_get(v_inst_48_, 1);
v_toInfSet_51_ = lean_ctor_get(v_inst_48_, 2);
v_isSharedCheck_61_ = !lean_is_exclusive(v_inst_48_);
if (v_isSharedCheck_61_ == 0)
{
v___x_53_ = v_inst_48_;
v_isShared_54_ = v_isSharedCheck_61_;
goto v_resetjp_52_;
}
else
{
lean_inc(v_toInfSet_51_);
lean_inc(v_toSupSet_50_);
lean_inc(v_toLattice_49_);
lean_dec(v_inst_48_);
v___x_53_ = lean_box(0);
v_isShared_54_ = v_isSharedCheck_61_;
goto v_resetjp_52_;
}
v_resetjp_52_:
{
lean_object* v___x_55_; lean_object* v___f_56_; lean_object* v___f_57_; lean_object* v___x_59_; 
v___x_55_ = lp_mathlib_OrderDual_instLattice___redArg(v_toLattice_49_);
v___f_56_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_supSet___redArg___lam__0), 2, 1);
lean_closure_set(v___f_56_, 0, v_toInfSet_51_);
v___f_57_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_supSet___redArg___lam__0), 2, 1);
lean_closure_set(v___f_57_, 0, v_toSupSet_50_);
if (v_isShared_54_ == 0)
{
lean_ctor_set(v___x_53_, 2, v___f_57_);
lean_ctor_set(v___x_53_, 1, v___f_56_);
lean_ctor_set(v___x_53_, 0, v___x_55_);
v___x_59_ = v___x_53_;
goto v_reusejp_58_;
}
else
{
lean_object* v_reuseFailAlloc_60_; 
v_reuseFailAlloc_60_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_60_, 0, v___x_55_);
lean_ctor_set(v_reuseFailAlloc_60_, 1, v___f_56_);
lean_ctor_set(v_reuseFailAlloc_60_, 2, v___f_57_);
v___x_59_ = v_reuseFailAlloc_60_;
goto v_reusejp_58_;
}
v_reusejp_58_:
{
return v___x_59_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompleteLattice(lean_object* v_00_u03b1_62_, lean_object* v_inst_63_){
_start:
{
lean_object* v___x_64_; 
v___x_64_ = lp_mathlib_OrderDual_instConditionallyCompleteLattice___redArg(v_inst_63_);
return v___x_64_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg___lam__0(lean_object* v_toDecidableEq_65_, lean_object* v_a_66_, lean_object* v_b_67_){
_start:
{
lean_object* v___x_68_; uint8_t v___x_69_; 
v___x_68_ = lean_apply_2(v_toDecidableEq_65_, v_a_66_, v_b_67_);
v___x_69_ = lean_unbox(v___x_68_);
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg___lam__0___boxed(lean_object* v_toDecidableEq_70_, lean_object* v_a_71_, lean_object* v_b_72_){
_start:
{
uint8_t v_res_73_; lean_object* v_r_74_; 
v_res_73_ = lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg___lam__0(v_toDecidableEq_70_, v_a_71_, v_b_72_);
v_r_74_ = lean_box(v_res_73_);
return v_r_74_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg___lam__1(lean_object* v_toDecidableLT_75_, lean_object* v_a_76_, lean_object* v_b_77_){
_start:
{
lean_object* v___x_78_; uint8_t v___x_79_; 
v___x_78_ = lean_apply_2(v_toDecidableLT_75_, v_b_77_, v_a_76_);
v___x_79_ = lean_unbox(v___x_78_);
return v___x_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg___lam__1___boxed(lean_object* v_toDecidableLT_80_, lean_object* v_a_81_, lean_object* v_b_82_){
_start:
{
uint8_t v_res_83_; lean_object* v_r_84_; 
v_res_83_ = lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg___lam__1(v_toDecidableLT_80_, v_a_81_, v_b_82_);
v_r_84_ = lean_box(v_res_83_);
return v_r_84_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg___lam__2(lean_object* v_toDecidableLE_85_, lean_object* v_a_86_, lean_object* v_b_87_){
_start:
{
lean_object* v___x_88_; uint8_t v___x_89_; 
v___x_88_ = lean_apply_2(v_toDecidableLE_85_, v_b_87_, v_a_86_);
v___x_89_ = lean_unbox(v___x_88_);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg___lam__2___boxed(lean_object* v_toDecidableLE_90_, lean_object* v_a_91_, lean_object* v_b_92_){
_start:
{
uint8_t v_res_93_; lean_object* v_r_94_; 
v_res_93_ = lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg___lam__2(v_toDecidableLE_90_, v_a_91_, v_b_92_);
v_r_94_ = lean_box(v_res_93_);
return v_r_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg(lean_object* v_inst_95_){
_start:
{
lean_object* v_toConditionallyCompleteLattice_96_; lean_object* v_toDecidableLE_97_; lean_object* v_toDecidableEq_98_; lean_object* v_toDecidableLT_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v_toOrd_103_; lean_object* v___f_104_; lean_object* v___f_105_; lean_object* v___f_106_; lean_object* v___x_107_; 
v_toConditionallyCompleteLattice_96_ = lean_ctor_get(v_inst_95_, 0);
v_toDecidableLE_97_ = lean_ctor_get(v_inst_95_, 2);
lean_inc_ref(v_toDecidableLE_97_);
v_toDecidableEq_98_ = lean_ctor_get(v_inst_95_, 3);
lean_inc_ref(v_toDecidableEq_98_);
v_toDecidableLT_99_ = lean_ctor_get(v_inst_95_, 4);
lean_inc_ref(v_toDecidableLT_99_);
lean_inc_ref(v_toConditionallyCompleteLattice_96_);
v___x_100_ = lp_mathlib_OrderDual_instConditionallyCompleteLattice___redArg(v_toConditionallyCompleteLattice_96_);
v___x_101_ = lp_mathlib_ConditionallyCompleteLinearOrder_toLinearOrder___redArg(v_inst_95_);
v___x_102_ = lp_mathlib_OrderDual_instLinearOrder___redArg(v___x_101_);
v_toOrd_103_ = lean_ctor_get(v___x_102_, 3);
lean_inc_ref(v_toOrd_103_);
lean_dec_ref(v___x_102_);
v___f_104_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_104_, 0, v_toDecidableEq_98_);
v___f_105_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg___lam__1___boxed), 3, 1);
lean_closure_set(v___f_105_, 0, v_toDecidableLT_99_);
v___f_106_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg___lam__2___boxed), 3, 1);
lean_closure_set(v___f_106_, 0, v_toDecidableLE_97_);
v___x_107_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_107_, 0, v___x_100_);
lean_ctor_set(v___x_107_, 1, v_toOrd_103_);
lean_ctor_set(v___x_107_, 2, v___f_106_);
lean_ctor_set(v___x_107_, 3, v___f_104_);
lean_ctor_set(v___x_107_, 4, v___f_105_);
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder(lean_object* v_00_u03b1_108_, lean_object* v_inst_109_){
_start:
{
lean_object* v___x_110_; 
v___x_110_ = lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg(v_inst_109_);
return v___x_110_;
}
}
static lean_object* _init_lp_mathlib_isLUB__csSup___auto__1___closed__13(void){
_start:
{
lean_object* v___x_135_; lean_object* v___x_136_; 
v___x_135_ = ((lean_object*)(lp_mathlib_isLUB__csSup___auto__1___closed__12));
v___x_136_ = l_Lean_mkAtom(v___x_135_);
return v___x_136_;
}
}
static lean_object* _init_lp_mathlib_isLUB__csSup___auto__1___closed__14(void){
_start:
{
lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; 
v___x_137_ = lean_obj_once(&lp_mathlib_isLUB__csSup___auto__1___closed__13, &lp_mathlib_isLUB__csSup___auto__1___closed__13_once, _init_lp_mathlib_isLUB__csSup___auto__1___closed__13);
v___x_138_ = ((lean_object*)(lp_mathlib_isLUB__csSup___auto__1___closed__5));
v___x_139_ = lean_array_push(v___x_138_, v___x_137_);
return v___x_139_;
}
}
static lean_object* _init_lp_mathlib_isLUB__csSup___auto__1___closed__15(void){
_start:
{
lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; 
v___x_140_ = lean_obj_once(&lp_mathlib_isLUB__csSup___auto__1___closed__14, &lp_mathlib_isLUB__csSup___auto__1___closed__14_once, _init_lp_mathlib_isLUB__csSup___auto__1___closed__14);
v___x_141_ = ((lean_object*)(lp_mathlib_isLUB__csSup___auto__1___closed__11));
v___x_142_ = lean_box(2);
v___x_143_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_143_, 0, v___x_142_);
lean_ctor_set(v___x_143_, 1, v___x_141_);
lean_ctor_set(v___x_143_, 2, v___x_140_);
return v___x_143_;
}
}
static lean_object* _init_lp_mathlib_isLUB__csSup___auto__1___closed__16(void){
_start:
{
lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; 
v___x_144_ = lean_obj_once(&lp_mathlib_isLUB__csSup___auto__1___closed__15, &lp_mathlib_isLUB__csSup___auto__1___closed__15_once, _init_lp_mathlib_isLUB__csSup___auto__1___closed__15);
v___x_145_ = ((lean_object*)(lp_mathlib_isLUB__csSup___auto__1___closed__5));
v___x_146_ = lean_array_push(v___x_145_, v___x_144_);
return v___x_146_;
}
}
static lean_object* _init_lp_mathlib_isLUB__csSup___auto__1___closed__17(void){
_start:
{
lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; 
v___x_147_ = lean_obj_once(&lp_mathlib_isLUB__csSup___auto__1___closed__16, &lp_mathlib_isLUB__csSup___auto__1___closed__16_once, _init_lp_mathlib_isLUB__csSup___auto__1___closed__16);
v___x_148_ = ((lean_object*)(lp_mathlib_isLUB__csSup___auto__1___closed__9));
v___x_149_ = lean_box(2);
v___x_150_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_150_, 0, v___x_149_);
lean_ctor_set(v___x_150_, 1, v___x_148_);
lean_ctor_set(v___x_150_, 2, v___x_147_);
return v___x_150_;
}
}
static lean_object* _init_lp_mathlib_isLUB__csSup___auto__1___closed__18(void){
_start:
{
lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; 
v___x_151_ = lean_obj_once(&lp_mathlib_isLUB__csSup___auto__1___closed__17, &lp_mathlib_isLUB__csSup___auto__1___closed__17_once, _init_lp_mathlib_isLUB__csSup___auto__1___closed__17);
v___x_152_ = ((lean_object*)(lp_mathlib_isLUB__csSup___auto__1___closed__5));
v___x_153_ = lean_array_push(v___x_152_, v___x_151_);
return v___x_153_;
}
}
static lean_object* _init_lp_mathlib_isLUB__csSup___auto__1___closed__19(void){
_start:
{
lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; 
v___x_154_ = lean_obj_once(&lp_mathlib_isLUB__csSup___auto__1___closed__18, &lp_mathlib_isLUB__csSup___auto__1___closed__18_once, _init_lp_mathlib_isLUB__csSup___auto__1___closed__18);
v___x_155_ = ((lean_object*)(lp_mathlib_isLUB__csSup___auto__1___closed__7));
v___x_156_ = lean_box(2);
v___x_157_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_157_, 0, v___x_156_);
lean_ctor_set(v___x_157_, 1, v___x_155_);
lean_ctor_set(v___x_157_, 2, v___x_154_);
return v___x_157_;
}
}
static lean_object* _init_lp_mathlib_isLUB__csSup___auto__1___closed__20(void){
_start:
{
lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; 
v___x_158_ = lean_obj_once(&lp_mathlib_isLUB__csSup___auto__1___closed__19, &lp_mathlib_isLUB__csSup___auto__1___closed__19_once, _init_lp_mathlib_isLUB__csSup___auto__1___closed__19);
v___x_159_ = ((lean_object*)(lp_mathlib_isLUB__csSup___auto__1___closed__5));
v___x_160_ = lean_array_push(v___x_159_, v___x_158_);
return v___x_160_;
}
}
static lean_object* _init_lp_mathlib_isLUB__csSup___auto__1___closed__21(void){
_start:
{
lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; 
v___x_161_ = lean_obj_once(&lp_mathlib_isLUB__csSup___auto__1___closed__20, &lp_mathlib_isLUB__csSup___auto__1___closed__20_once, _init_lp_mathlib_isLUB__csSup___auto__1___closed__20);
v___x_162_ = ((lean_object*)(lp_mathlib_isLUB__csSup___auto__1___closed__4));
v___x_163_ = lean_box(2);
v___x_164_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_164_, 0, v___x_163_);
lean_ctor_set(v___x_164_, 1, v___x_162_);
lean_ctor_set(v___x_164_, 2, v___x_161_);
return v___x_164_;
}
}
static lean_object* _init_lp_mathlib_isLUB__csSup___auto__1(void){
_start:
{
lean_object* v___x_165_; 
v___x_165_ = lean_obj_once(&lp_mathlib_isLUB__csSup___auto__1___closed__21, &lp_mathlib_isLUB__csSup___auto__1___closed__21_once, _init_lp_mathlib_isLUB__csSup___auto__1___closed__21);
return v___x_165_;
}
}
static lean_object* _init_lp_mathlib_isGLB__csInf___auto__1(void){
_start:
{
lean_object* v___x_166_; 
v___x_166_ = lean_obj_once(&lp_mathlib_isLUB__csSup___auto__1___closed__21, &lp_mathlib_isLUB__csSup___auto__1___closed__21_once, _init_lp_mathlib_isLUB__csSup___auto__1___closed__21);
return v___x_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConditionallyCompleteLattice_toConditionallyCompletePartialOrder___redArg(lean_object* v_inst_167_){
_start:
{
lean_object* v_toLattice_168_; lean_object* v_toSupSet_169_; lean_object* v_toInfSet_170_; lean_object* v___x_171_; lean_object* v_toPartialOrder_172_; lean_object* v___x_174_; uint8_t v_isShared_175_; uint8_t v_isSharedCheck_180_; 
v_toLattice_168_ = lean_ctor_get(v_inst_167_, 0);
lean_inc_ref(v_toLattice_168_);
v_toSupSet_169_ = lean_ctor_get(v_inst_167_, 1);
lean_inc(v_toSupSet_169_);
v_toInfSet_170_ = lean_ctor_get(v_inst_167_, 2);
lean_inc(v_toInfSet_170_);
lean_dec_ref(v_inst_167_);
v___x_171_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_toLattice_168_);
v_toPartialOrder_172_ = lean_ctor_get(v___x_171_, 0);
v_isSharedCheck_180_ = !lean_is_exclusive(v___x_171_);
if (v_isSharedCheck_180_ == 0)
{
lean_object* v_unused_181_; 
v_unused_181_ = lean_ctor_get(v___x_171_, 1);
lean_dec(v_unused_181_);
v___x_174_ = v___x_171_;
v_isShared_175_ = v_isSharedCheck_180_;
goto v_resetjp_173_;
}
else
{
lean_inc(v_toPartialOrder_172_);
lean_dec(v___x_171_);
v___x_174_ = lean_box(0);
v_isShared_175_ = v_isSharedCheck_180_;
goto v_resetjp_173_;
}
v_resetjp_173_:
{
lean_object* v___x_177_; 
if (v_isShared_175_ == 0)
{
lean_ctor_set(v___x_174_, 1, v_toSupSet_169_);
v___x_177_ = v___x_174_;
goto v_reusejp_176_;
}
else
{
lean_object* v_reuseFailAlloc_179_; 
v_reuseFailAlloc_179_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_179_, 0, v_toPartialOrder_172_);
lean_ctor_set(v_reuseFailAlloc_179_, 1, v_toSupSet_169_);
v___x_177_ = v_reuseFailAlloc_179_;
goto v_reusejp_176_;
}
v_reusejp_176_:
{
lean_object* v___x_178_; 
v___x_178_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_178_, 0, v___x_177_);
lean_ctor_set(v___x_178_, 1, v_toInfSet_170_);
return v___x_178_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConditionallyCompleteLattice_toConditionallyCompletePartialOrder(lean_object* v_00_u03b1_182_, lean_object* v_inst_183_){
_start:
{
lean_object* v___x_184_; 
v___x_184_ = lp_mathlib_ConditionallyCompleteLattice_toConditionallyCompletePartialOrder___redArg(v_inst_183_);
return v___x_184_;
}
}
static lean_object* _init_lp_mathlib_csInf__le__csSup___auto__1(void){
_start:
{
lean_object* v___x_185_; 
v___x_185_ = lean_obj_once(&lp_mathlib_isLUB__csSup___auto__1___closed__21, &lp_mathlib_isLUB__csSup___auto__1___closed__21_once, _init_lp_mathlib_isLUB__csSup___auto__1___closed__21);
return v___x_185_;
}
}
static lean_object* _init_lp_mathlib_csInf__le__csSup___auto__3(void){
_start:
{
lean_object* v___x_186_; 
v___x_186_ = lean_obj_once(&lp_mathlib_isLUB__csSup___auto__1___closed__21, &lp_mathlib_isLUB__csSup___auto__1___closed__21_once, _init_lp_mathlib_isLUB__csSup___auto__1___closed__21);
return v___x_186_;
}
}
static lean_object* _init_lp_mathlib_csInf__le__csSup__of__nonempty__inter___auto__1(void){
_start:
{
lean_object* v___x_187_; 
v___x_187_ = lean_obj_once(&lp_mathlib_isLUB__csSup___auto__1___closed__21, &lp_mathlib_isLUB__csSup___auto__1___closed__21_once, _init_lp_mathlib_isLUB__csSup___auto__1___closed__21);
return v___x_187_;
}
}
static lean_object* _init_lp_mathlib_csInf__le__csSup__of__nonempty__inter___auto__3(void){
_start:
{
lean_object* v___x_188_; 
v___x_188_ = lean_obj_once(&lp_mathlib_isLUB__csSup___auto__1___closed__21, &lp_mathlib_isLUB__csSup___auto__1___closed__21_once, _init_lp_mathlib_isLUB__csSup___auto__1___closed__21);
return v___x_188_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_conditionallyCompleteLattice___redArg___lam__0(lean_object* v_inst_189_, lean_object* v_i_190_){
_start:
{
lean_object* v___x_191_; lean_object* v_toLattice_192_; 
v___x_191_ = lean_apply_1(v_inst_189_, v_i_190_);
v_toLattice_192_ = lean_ctor_get(v___x_191_, 0);
lean_inc_ref(v_toLattice_192_);
lean_dec_ref(v___x_191_);
return v_toLattice_192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_conditionallyCompleteLattice___redArg___lam__1(lean_object* v_inst_193_, lean_object* v_i_194_, lean_object* v___y_195_){
_start:
{
lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v_toConditionallyCompletePartialOrderSup_198_; lean_object* v_toSupSet_199_; lean_object* v___x_200_; 
v___x_196_ = lean_apply_1(v_inst_193_, v_i_194_);
v___x_197_ = lp_mathlib_ConditionallyCompleteLattice_toConditionallyCompletePartialOrder___redArg(v___x_196_);
v_toConditionallyCompletePartialOrderSup_198_ = lean_ctor_get(v___x_197_, 0);
lean_inc_ref(v_toConditionallyCompletePartialOrderSup_198_);
lean_dec_ref(v___x_197_);
v_toSupSet_199_ = lean_ctor_get(v_toConditionallyCompletePartialOrderSup_198_, 1);
lean_inc(v_toSupSet_199_);
lean_dec_ref(v_toConditionallyCompletePartialOrderSup_198_);
v___x_200_ = lean_apply_1(v_toSupSet_199_, lean_box(0));
return v___x_200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_conditionallyCompleteLattice___redArg___lam__2(lean_object* v_inst_201_, lean_object* v_i_202_, lean_object* v___y_203_){
_start:
{
lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v_toInfSet_207_; lean_object* v___x_208_; 
v___x_204_ = lean_apply_1(v_inst_201_, v_i_202_);
v___x_205_ = lp_mathlib_ConditionallyCompleteLattice_toConditionallyCompletePartialOrder___redArg(v___x_204_);
v___x_206_ = lp_mathlib_ConditionallyCompletePartialOrder_toConditionallyCompletePartialOrderInf___redArg(v___x_205_);
v_toInfSet_207_ = lean_ctor_get(v___x_206_, 1);
lean_inc(v_toInfSet_207_);
lean_dec_ref(v___x_206_);
v___x_208_ = lean_apply_1(v_toInfSet_207_, lean_box(0));
return v___x_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_conditionallyCompleteLattice___redArg(lean_object* v_inst_209_){
_start:
{
lean_object* v___f_210_; lean_object* v___f_211_; lean_object* v___f_212_; lean_object* v___x_213_; lean_object* v___f_214_; lean_object* v___f_215_; lean_object* v___x_216_; 
lean_inc_ref_n(v_inst_209_, 2);
v___f_210_ = lean_alloc_closure((void*)(lp_mathlib_Pi_conditionallyCompleteLattice___redArg___lam__0), 2, 1);
lean_closure_set(v___f_210_, 0, v_inst_209_);
v___f_211_ = lean_alloc_closure((void*)(lp_mathlib_Pi_conditionallyCompleteLattice___redArg___lam__1), 3, 1);
lean_closure_set(v___f_211_, 0, v_inst_209_);
v___f_212_ = lean_alloc_closure((void*)(lp_mathlib_Pi_conditionallyCompleteLattice___redArg___lam__2), 3, 1);
lean_closure_set(v___f_212_, 0, v_inst_209_);
v___x_213_ = lp_mathlib_Pi_instLattice___redArg(v___f_210_);
v___f_214_ = lean_alloc_closure((void*)(lp_mathlib_Pi_supSet___redArg___lam__0), 3, 1);
lean_closure_set(v___f_214_, 0, v___f_211_);
v___f_215_ = lean_alloc_closure((void*)(lp_mathlib_Pi_supSet___redArg___lam__0), 3, 1);
lean_closure_set(v___f_215_, 0, v___f_212_);
v___x_216_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_216_, 0, v___x_213_);
lean_ctor_set(v___x_216_, 1, v___f_214_);
lean_ctor_set(v___x_216_, 2, v___f_215_);
return v___x_216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_conditionallyCompleteLattice(lean_object* v_00_u03b9_217_, lean_object* v_00_u03b1_218_, lean_object* v_inst_219_){
_start:
{
lean_object* v___x_220_; 
v___x_220_ = lp_mathlib_Pi_conditionallyCompleteLattice___redArg(v_inst_219_);
return v___x_220_;
}
}
static lean_object* _init_lp_mathlib_csSup__union_x27___auto__1(void){
_start:
{
lean_object* v___x_221_; 
v___x_221_ = lean_obj_once(&lp_mathlib_isLUB__csSup___auto__1___closed__21, &lp_mathlib_isLUB__csSup___auto__1___closed__21_once, _init_lp_mathlib_isLUB__csSup___auto__1___closed__21);
return v___x_221_;
}
}
static lean_object* _init_lp_mathlib_csSup__union_x27___auto__3(void){
_start:
{
lean_object* v___x_222_; 
v___x_222_ = lean_obj_once(&lp_mathlib_isLUB__csSup___auto__1___closed__21, &lp_mathlib_isLUB__csSup___auto__1___closed__21_once, _init_lp_mathlib_isLUB__csSup___auto__1___closed__21);
return v___x_222_;
}
}
static lean_object* _init_lp_mathlib_csSup__inter__le_x27___auto__1(void){
_start:
{
lean_object* v___x_223_; 
v___x_223_ = lean_obj_once(&lp_mathlib_isLUB__csSup___auto__1___closed__21, &lp_mathlib_isLUB__csSup___auto__1___closed__21_once, _init_lp_mathlib_isLUB__csSup___auto__1___closed__21);
return v___x_223_;
}
}
static lean_object* _init_lp_mathlib_csSup__inter__le_x27___auto__3(void){
_start:
{
lean_object* v___x_224_; 
v___x_224_ = lean_obj_once(&lp_mathlib_isLUB__csSup___auto__1___closed__21, &lp_mathlib_isLUB__csSup___auto__1___closed__21_once, _init_lp_mathlib_isLUB__csSup___auto__1___closed__21);
return v___x_224_;
}
}
static lean_object* _init_lp_mathlib_csSup__insert_x27___auto__1(void){
_start:
{
lean_object* v___x_225_; 
v___x_225_ = lean_obj_once(&lp_mathlib_isLUB__csSup___auto__1___closed__21, &lp_mathlib_isLUB__csSup___auto__1___closed__21_once, _init_lp_mathlib_isLUB__csSup___auto__1___closed__21);
return v___x_225_;
}
}
static lean_object* _init_lp_mathlib_Monotone_csSup__image__le__map__csSup___auto__1(void){
_start:
{
lean_object* v___x_226_; 
v___x_226_ = lean_obj_once(&lp_mathlib_isLUB__csSup___auto__1___closed__21, &lp_mathlib_isLUB__csSup___auto__1___closed__21_once, _init_lp_mathlib_isLUB__csSup___auto__1___closed__21);
return v___x_226_;
}
}
static lean_object* _init_lp_mathlib_Monotone_map__csInf__le__csInf__image___auto__1(void){
_start:
{
lean_object* v___x_227_; 
v___x_227_ = lean_obj_once(&lp_mathlib_isLUB__csSup___auto__1___closed__21, &lp_mathlib_isLUB__csSup___auto__1___closed__21_once, _init_lp_mathlib_isLUB__csSup___auto__1___closed__21);
return v___x_227_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Lattice_Indexed(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_ConditionallyCompletePartialOrder_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Lattice_Indexed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_ConditionallyCompletePartialOrder_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Basic(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_isLUB__csSup___auto__1 = _init_lp_mathlib_isLUB__csSup___auto__1();
lean_mark_persistent(lp_mathlib_isLUB__csSup___auto__1);
lp_mathlib_isGLB__csInf___auto__1 = _init_lp_mathlib_isGLB__csInf___auto__1();
lean_mark_persistent(lp_mathlib_isGLB__csInf___auto__1);
lp_mathlib_csInf__le__csSup___auto__1 = _init_lp_mathlib_csInf__le__csSup___auto__1();
lean_mark_persistent(lp_mathlib_csInf__le__csSup___auto__1);
lp_mathlib_csInf__le__csSup___auto__3 = _init_lp_mathlib_csInf__le__csSup___auto__3();
lean_mark_persistent(lp_mathlib_csInf__le__csSup___auto__3);
lp_mathlib_csInf__le__csSup__of__nonempty__inter___auto__1 = _init_lp_mathlib_csInf__le__csSup__of__nonempty__inter___auto__1();
lean_mark_persistent(lp_mathlib_csInf__le__csSup__of__nonempty__inter___auto__1);
lp_mathlib_csInf__le__csSup__of__nonempty__inter___auto__3 = _init_lp_mathlib_csInf__le__csSup__of__nonempty__inter___auto__3();
lean_mark_persistent(lp_mathlib_csInf__le__csSup__of__nonempty__inter___auto__3);
lp_mathlib_csSup__union_x27___auto__1 = _init_lp_mathlib_csSup__union_x27___auto__1();
lean_mark_persistent(lp_mathlib_csSup__union_x27___auto__1);
lp_mathlib_csSup__union_x27___auto__3 = _init_lp_mathlib_csSup__union_x27___auto__3();
lean_mark_persistent(lp_mathlib_csSup__union_x27___auto__3);
lp_mathlib_csSup__inter__le_x27___auto__1 = _init_lp_mathlib_csSup__inter__le_x27___auto__1();
lean_mark_persistent(lp_mathlib_csSup__inter__le_x27___auto__1);
lp_mathlib_csSup__inter__le_x27___auto__3 = _init_lp_mathlib_csSup__inter__le_x27___auto__3();
lean_mark_persistent(lp_mathlib_csSup__inter__le_x27___auto__3);
lp_mathlib_csSup__insert_x27___auto__1 = _init_lp_mathlib_csSup__insert_x27___auto__1();
lean_mark_persistent(lp_mathlib_csSup__insert_x27___auto__1);
lp_mathlib_Monotone_csSup__image__le__map__csSup___auto__1 = _init_lp_mathlib_Monotone_csSup__image__le__map__csSup___auto__1();
lean_mark_persistent(lp_mathlib_Monotone_csSup__image__le__map__csSup___auto__1);
lp_mathlib_Monotone_map__csInf__le__csInf__image___auto__1 = _init_lp_mathlib_Monotone_map__csInf__le__csInf__image___auto__1();
lean_mark_persistent(lp_mathlib_Monotone_map__csInf__le__csInf__image___auto__1);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Lattice_Indexed(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_ConditionallyCompletePartialOrder_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Lattice_Indexed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_ConditionallyCompletePartialOrder_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
