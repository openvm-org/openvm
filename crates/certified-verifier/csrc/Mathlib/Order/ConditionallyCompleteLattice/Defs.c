// Lean compiler output
// Module: Mathlib.Order.ConditionallyCompleteLattice.Defs
// Imports: public import Init public meta import Init public import Mathlib.Order.Bounds.Basic public import Mathlib.Order.SetNotation public import Mathlib.Order.WellFounded
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
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lp_mathlib_Lattice_ofIsLUBofIsGLB___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__0 = (const lean_object*)&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__0_value;
static const lean_string_object lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__1 = (const lean_object*)&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__1_value;
static const lean_string_object lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__2 = (const lean_object*)&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__2_value;
static const lean_string_object lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__3 = (const lean_object*)&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__3_value;
static const lean_ctor_object lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__4_value_aux_0),((lean_object*)&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__4_value_aux_1),((lean_object*)&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__4_value_aux_2),((lean_object*)&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__4 = (const lean_object*)&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__4_value;
static const lean_array_object lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__5 = (const lean_object*)&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__5_value;
static const lean_string_object lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__6 = (const lean_object*)&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__6_value;
static const lean_ctor_object lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__7_value_aux_0),((lean_object*)&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__7_value_aux_1),((lean_object*)&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__7_value_aux_2),((lean_object*)&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__6_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__7 = (const lean_object*)&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__7_value;
static const lean_string_object lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__8 = (const lean_object*)&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__8_value;
static const lean_ctor_object lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__9 = (const lean_object*)&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__9_value;
static const lean_string_object lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "tacticCompareOfLessAndEq_rfl"};
static const lean_object* lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__10 = (const lean_object*)&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__10_value;
static const lean_ctor_object lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__10_value),LEAN_SCALAR_PTR_LITERAL(181, 115, 139, 42, 76, 141, 255, 128)}};
static const lean_object* lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__11 = (const lean_object*)&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__11_value;
static const lean_string_object lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "compareOfLessAndEq_rfl"};
static const lean_object* lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__12 = (const lean_object*)&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__12_value;
static lean_once_cell_t lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__13;
static lean_once_cell_t lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__14;
static lean_once_cell_t lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__15;
static lean_once_cell_t lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__16;
static lean_once_cell_t lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__17;
static lean_once_cell_t lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__18;
static lean_once_cell_t lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__19;
static lean_once_cell_t lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__20;
static lean_once_cell_t lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__21;
LEAN_EXPORT lean_object* lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_conditionallyCompleteLatticeOfsSup___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_conditionallyCompleteLatticeOfsSup___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_conditionallyCompleteLatticeOfsSup___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_conditionallyCompleteLatticeOfsSup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_conditionallyCompleteLatticeOfsSup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_conditionallyCompleteLatticeOfsInf___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_conditionallyCompleteLatticeOfsInf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_conditionallyCompleteLatticeOfLatticeOfsSup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_conditionallyCompleteLatticeOfLatticeOfsSup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_conditionallyCompleteLatticeOfLatticeOfsInf___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_conditionallyCompleteLatticeOfLatticeOfsInf(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__13(void){
_start:
{
lean_object* v___x_25_; lean_object* v___x_26_; 
v___x_25_ = ((lean_object*)(lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__12));
v___x_26_ = l_Lean_mkAtom(v___x_25_);
return v___x_26_;
}
}
static lean_object* _init_lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__14(void){
_start:
{
lean_object* v___x_27_; lean_object* v___x_28_; lean_object* v___x_29_; 
v___x_27_ = lean_obj_once(&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__13, &lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__13_once, _init_lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__13);
v___x_28_ = ((lean_object*)(lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__5));
v___x_29_ = lean_array_push(v___x_28_, v___x_27_);
return v___x_29_;
}
}
static lean_object* _init_lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__15(void){
_start:
{
lean_object* v___x_30_; lean_object* v___x_31_; lean_object* v___x_32_; lean_object* v___x_33_; 
v___x_30_ = lean_obj_once(&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__14, &lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__14_once, _init_lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__14);
v___x_31_ = ((lean_object*)(lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__11));
v___x_32_ = lean_box(2);
v___x_33_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_33_, 0, v___x_32_);
lean_ctor_set(v___x_33_, 1, v___x_31_);
lean_ctor_set(v___x_33_, 2, v___x_30_);
return v___x_33_;
}
}
static lean_object* _init_lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__16(void){
_start:
{
lean_object* v___x_34_; lean_object* v___x_35_; lean_object* v___x_36_; 
v___x_34_ = lean_obj_once(&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__15, &lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__15_once, _init_lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__15);
v___x_35_ = ((lean_object*)(lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__5));
v___x_36_ = lean_array_push(v___x_35_, v___x_34_);
return v___x_36_;
}
}
static lean_object* _init_lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__17(void){
_start:
{
lean_object* v___x_37_; lean_object* v___x_38_; lean_object* v___x_39_; lean_object* v___x_40_; 
v___x_37_ = lean_obj_once(&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__16, &lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__16_once, _init_lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__16);
v___x_38_ = ((lean_object*)(lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__9));
v___x_39_ = lean_box(2);
v___x_40_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_40_, 0, v___x_39_);
lean_ctor_set(v___x_40_, 1, v___x_38_);
lean_ctor_set(v___x_40_, 2, v___x_37_);
return v___x_40_;
}
}
static lean_object* _init_lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__18(void){
_start:
{
lean_object* v___x_41_; lean_object* v___x_42_; lean_object* v___x_43_; 
v___x_41_ = lean_obj_once(&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__17, &lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__17_once, _init_lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__17);
v___x_42_ = ((lean_object*)(lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__5));
v___x_43_ = lean_array_push(v___x_42_, v___x_41_);
return v___x_43_;
}
}
static lean_object* _init_lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__19(void){
_start:
{
lean_object* v___x_44_; lean_object* v___x_45_; lean_object* v___x_46_; lean_object* v___x_47_; 
v___x_44_ = lean_obj_once(&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__18, &lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__18_once, _init_lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__18);
v___x_45_ = ((lean_object*)(lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__7));
v___x_46_ = lean_box(2);
v___x_47_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_47_, 0, v___x_46_);
lean_ctor_set(v___x_47_, 1, v___x_45_);
lean_ctor_set(v___x_47_, 2, v___x_44_);
return v___x_47_;
}
}
static lean_object* _init_lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__20(void){
_start:
{
lean_object* v___x_48_; lean_object* v___x_49_; lean_object* v___x_50_; 
v___x_48_ = lean_obj_once(&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__19, &lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__19_once, _init_lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__19);
v___x_49_ = ((lean_object*)(lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__5));
v___x_50_ = lean_array_push(v___x_49_, v___x_48_);
return v___x_50_;
}
}
static lean_object* _init_lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__21(void){
_start:
{
lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; 
v___x_51_ = lean_obj_once(&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__20, &lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__20_once, _init_lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__20);
v___x_52_ = ((lean_object*)(lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__4));
v___x_53_ = lean_box(2);
v___x_54_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_54_, 0, v___x_53_);
lean_ctor_set(v___x_54_, 1, v___x_52_);
lean_ctor_set(v___x_54_, 2, v___x_51_);
return v___x_54_;
}
}
static lean_object* _init_lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam(void){
_start:
{
lean_object* v___x_55_; 
v___x_55_ = lean_obj_once(&lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__21, &lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__21_once, _init_lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__21);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_conditionallyCompleteLatticeOfsSup___redArg___lam__0(lean_object* v_H2_56_, lean_object* v_a_57_, lean_object* v_b_58_){
_start:
{
lean_object* v___x_59_; 
v___x_59_ = lean_apply_1(v_H2_56_, lean_box(0));
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_conditionallyCompleteLatticeOfsSup___redArg___lam__0___boxed(lean_object* v_H2_60_, lean_object* v_a_61_, lean_object* v_b_62_){
_start:
{
lean_object* v_res_63_; 
v_res_63_ = lp_mathlib_conditionallyCompleteLatticeOfsSup___redArg___lam__0(v_H2_60_, v_a_61_, v_b_62_);
lean_dec(v_b_62_);
lean_dec(v_a_61_);
return v_res_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_conditionallyCompleteLatticeOfsSup___redArg___lam__2(lean_object* v_H2_64_, lean_object* v_s_65_){
_start:
{
lean_object* v___x_66_; 
v___x_66_ = lean_apply_1(v_H2_64_, lean_box(0));
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_conditionallyCompleteLatticeOfsSup___redArg(lean_object* v_H1_67_, lean_object* v_H2_68_){
_start:
{
lean_object* v___f_69_; lean_object* v___f_70_; lean_object* v___x_71_; lean_object* v___x_72_; 
lean_inc_n(v_H2_68_, 2);
v___f_69_ = lean_alloc_closure((void*)(lp_mathlib_conditionallyCompleteLatticeOfsSup___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_69_, 0, v_H2_68_);
v___f_70_ = lean_alloc_closure((void*)(lp_mathlib_conditionallyCompleteLatticeOfsSup___redArg___lam__2), 2, 1);
lean_closure_set(v___f_70_, 0, v_H2_68_);
lean_inc_ref(v___f_69_);
v___x_71_ = lp_mathlib_Lattice_ofIsLUBofIsGLB___redArg(v_H1_67_, v___f_69_, v___f_69_);
v___x_72_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_72_, 0, v___x_71_);
lean_ctor_set(v___x_72_, 1, v_H2_68_);
lean_ctor_set(v___x_72_, 2, v___f_70_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_conditionallyCompleteLatticeOfsSup(lean_object* v_00_u03b1_73_, lean_object* v_H1_74_, lean_object* v_H2_75_, lean_object* v_bddAbove__pair_76_, lean_object* v_bddBelow__pair_77_, lean_object* v_isLUB__sSup_78_){
_start:
{
lean_object* v___x_79_; 
v___x_79_ = lp_mathlib_conditionallyCompleteLatticeOfsSup___redArg(v_H1_74_, v_H2_75_);
return v___x_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_conditionallyCompleteLatticeOfsInf___redArg(lean_object* v_H1_80_, lean_object* v_H2_81_){
_start:
{
lean_object* v___f_82_; lean_object* v___f_83_; lean_object* v___x_84_; lean_object* v___x_85_; 
lean_inc_n(v_H2_81_, 2);
v___f_82_ = lean_alloc_closure((void*)(lp_mathlib_conditionallyCompleteLatticeOfsSup___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_82_, 0, v_H2_81_);
v___f_83_ = lean_alloc_closure((void*)(lp_mathlib_conditionallyCompleteLatticeOfsSup___redArg___lam__2), 2, 1);
lean_closure_set(v___f_83_, 0, v_H2_81_);
lean_inc_ref(v___f_82_);
v___x_84_ = lp_mathlib_Lattice_ofIsLUBofIsGLB___redArg(v_H1_80_, v___f_82_, v___f_82_);
v___x_85_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_85_, 0, v___x_84_);
lean_ctor_set(v___x_85_, 1, v___f_83_);
lean_ctor_set(v___x_85_, 2, v_H2_81_);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_conditionallyCompleteLatticeOfsInf(lean_object* v_00_u03b1_86_, lean_object* v_H1_87_, lean_object* v_H2_88_, lean_object* v_bddBelow__pair_89_, lean_object* v_bddAbove__pair_90_, lean_object* v_isLUB__sSup_91_){
_start:
{
lean_object* v___x_92_; 
v___x_92_ = lp_mathlib_conditionallyCompleteLatticeOfsInf___redArg(v_H1_87_, v_H2_88_);
return v___x_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_conditionallyCompleteLatticeOfLatticeOfsSup___redArg(lean_object* v_H1_93_, lean_object* v_inst_94_){
_start:
{
lean_object* v___f_95_; lean_object* v___x_96_; 
lean_inc(v_inst_94_);
v___f_95_ = lean_alloc_closure((void*)(lp_mathlib_conditionallyCompleteLatticeOfsSup___redArg___lam__2), 2, 1);
lean_closure_set(v___f_95_, 0, v_inst_94_);
v___x_96_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_96_, 0, v_H1_93_);
lean_ctor_set(v___x_96_, 1, v_inst_94_);
lean_ctor_set(v___x_96_, 2, v___f_95_);
return v___x_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_conditionallyCompleteLatticeOfLatticeOfsSup(lean_object* v_00_u03b1_97_, lean_object* v_H1_98_, lean_object* v_inst_99_, lean_object* v_isLUB__sSup_100_){
_start:
{
lean_object* v___x_101_; 
v___x_101_ = lp_mathlib_conditionallyCompleteLatticeOfLatticeOfsSup___redArg(v_H1_98_, v_inst_99_);
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_conditionallyCompleteLatticeOfLatticeOfsInf___redArg(lean_object* v_H1_102_, lean_object* v_inst_103_){
_start:
{
lean_object* v___f_104_; lean_object* v___x_105_; 
lean_inc(v_inst_103_);
v___f_104_ = lean_alloc_closure((void*)(lp_mathlib_conditionallyCompleteLatticeOfsSup___redArg___lam__2), 2, 1);
lean_closure_set(v___f_104_, 0, v_inst_103_);
v___x_105_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_105_, 0, v_H1_102_);
lean_ctor_set(v___x_105_, 1, v___f_104_);
lean_ctor_set(v___x_105_, 2, v_inst_103_);
return v___x_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_conditionallyCompleteLatticeOfLatticeOfsInf(lean_object* v_00_u03b1_106_, lean_object* v_H1_107_, lean_object* v_inst_108_, lean_object* v_isLUB__sSup_109_){
_start:
{
lean_object* v___x_110_; 
v___x_110_ = lp_mathlib_conditionallyCompleteLatticeOfLatticeOfsInf___redArg(v_H1_107_, v_inst_108_);
return v___x_110_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Bounds_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_SetNotation(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_WellFounded(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Bounds_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_SetNotation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_WellFounded(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Defs(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam = _init_lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam();
lean_mark_persistent(lp_mathlib_ConditionallyCompleteLinearOrder_compare__eq__compareOfLessAndEq___autoParam);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Bounds_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_SetNotation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_WellFounded(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Bounds_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_SetNotation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_WellFounded(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
