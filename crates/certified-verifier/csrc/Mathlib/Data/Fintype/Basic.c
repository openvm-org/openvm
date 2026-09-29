// Lean compiler output
// Module: Mathlib.Data.Fintype.Basic
// Imports: public import Init public meta import Init public import Mathlib.Basic.Finite.Defs public import Mathlib.Data.Finset.BooleanAlgebra public import Mathlib.Data.Finset.Image public import Mathlib.Data.Fintype.Defs public import Mathlib.Data.Fintype.OfMap public import Mathlib.Data.Fintype.Sets public import Mathlib.Data.List.FinRange public import Mathlib.Data.List.OfFn
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
lean_object* lp_mathlib_Fintype_ofSubsingleton___redArg(lean_object*);
lean_object* lp_mathlib_Subtype_fintype___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Fintype_ofEquiv___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Quotient_mk_x27_x27___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_image___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_ulift(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_plift(lean_object*);
lean_object* l_List_finRange(lean_object*);
lean_object* lp_mathlib_Fintype_subtype___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_fintype(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_fintype___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_fintype(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_subtypeEq___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_subtypeEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_subtypeEq_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_subtypeEq_x27(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Unit_fintype___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Unit_fintype___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Unit_fintype;
LEAN_EXPORT lean_object* lp_mathlib_PUnit_fintype;
LEAN_EXPORT lean_object* lp_mathlib_Fintype_prodLeft___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_prodLeft___redArg___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Fintype_prodLeft___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Fintype_prodLeft___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Fintype_prodLeft___redArg___closed__0 = (const lean_object*)&lp_mathlib_Fintype_prodLeft___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Fintype_prodLeft___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_prodLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_prodRight___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_prodRight___redArg___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Fintype_prodRight___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Fintype_prodRight___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Fintype_prodRight___redArg___closed__0 = (const lean_object*)&lp_mathlib_Fintype_prodRight___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Fintype_prodRight___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_prodRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_ULift_fintype___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ULift_fintype___redArg___closed__0;
static lean_once_cell_t lp_mathlib_ULift_fintype___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ULift_fintype___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_ULift_fintype___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_fintype(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_PLift_fintype___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_PLift_fintype___redArg___closed__0;
static lean_once_cell_t lp_mathlib_PLift_fintype___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_PLift_fintype___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_PLift_fintype___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PLift_fintype(lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_PLift_fintypeProp___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_PLift_fintypeProp___redArg___closed__0 = (const lean_object*)&lp_mathlib_PLift_fintypeProp___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_PLift_fintypeProp___redArg(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_PLift_fintypeProp___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PLift_fintypeProp(lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_PLift_fintypeProp___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Quotient_fintype___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_fintype___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_fintype___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_fintype(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PSigma_fintypePropLeft___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PSigma_fintypePropLeft___redArg___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PSigma_fintypePropLeft___redArg___lam__1___boxed(lean_object*);
static const lean_closure_object lp_mathlib_PSigma_fintypePropLeft___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_PSigma_fintypePropLeft___redArg___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_PSigma_fintypePropLeft___redArg___closed__0 = (const lean_object*)&lp_mathlib_PSigma_fintypePropLeft___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_PSigma_fintypePropLeft___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_PSigma_fintypePropLeft___redArg___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_PSigma_fintypePropLeft___redArg___closed__1 = (const lean_object*)&lp_mathlib_PSigma_fintypePropLeft___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_PSigma_fintypePropLeft___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_PSigma_fintypePropLeft___redArg___closed__0_value),((lean_object*)&lp_mathlib_PSigma_fintypePropLeft___redArg___closed__1_value)}};
static const lean_object* lp_mathlib_PSigma_fintypePropLeft___redArg___closed__2 = (const lean_object*)&lp_mathlib_PSigma_fintypePropLeft___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_PSigma_fintypePropLeft___redArg(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PSigma_fintypePropLeft___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PSigma_fintypePropLeft(lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PSigma_fintypePropLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PSigma_fintypePropRight___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PSigma_fintypePropRight___redArg___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PSigma_fintypePropRight___redArg___lam__1___boxed(lean_object*);
static const lean_closure_object lp_mathlib_PSigma_fintypePropRight___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_PSigma_fintypePropRight___redArg___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_PSigma_fintypePropRight___redArg___closed__0 = (const lean_object*)&lp_mathlib_PSigma_fintypePropRight___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_PSigma_fintypePropRight___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_PSigma_fintypePropRight___redArg___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_PSigma_fintypePropRight___redArg___closed__1 = (const lean_object*)&lp_mathlib_PSigma_fintypePropRight___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_PSigma_fintypePropRight___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_PSigma_fintypePropRight___redArg___closed__0_value),((lean_object*)&lp_mathlib_PSigma_fintypePropRight___redArg___closed__1_value)}};
static const lean_object* lp_mathlib_PSigma_fintypePropRight___redArg___closed__2 = (const lean_object*)&lp_mathlib_PSigma_fintypePropRight___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_PSigma_fintypePropRight___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PSigma_fintypePropRight(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_PSigma_fintypePropProp___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_PSigma_fintypePropProp___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_PSigma_fintypePropProp___redArg(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PSigma_fintypePropProp___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PSigma_fintypePropProp(lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PSigma_fintypePropProp___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_pfunFintype___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_pfunFintype___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_pfunFintype___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_pfunFintype___redArg___lam__2(lean_object*);
static const lean_closure_object lp_mathlib_pfunFintype___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_pfunFintype___redArg___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_pfunFintype___redArg___closed__0 = (const lean_object*)&lp_mathlib_pfunFintype___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_pfunFintype___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_pfunFintype___redArg___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_pfunFintype___redArg___closed__1 = (const lean_object*)&lp_mathlib_pfunFintype___redArg___closed__1_value;
static const lean_closure_object lp_mathlib_pfunFintype___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_pfunFintype___redArg___lam__1___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_pfunFintype___redArg___closed__2 = (const lean_object*)&lp_mathlib_pfunFintype___redArg___closed__2_value;
static const lean_closure_object lp_mathlib_pfunFintype___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_pfunFintype___redArg___lam__2, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_pfunFintype___redArg___closed__3 = (const lean_object*)&lp_mathlib_pfunFintype___redArg___closed__3_value;
static const lean_ctor_object lp_mathlib_pfunFintype___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_pfunFintype___redArg___closed__2_value),((lean_object*)&lp_mathlib_pfunFintype___redArg___closed__3_value)}};
static const lean_object* lp_mathlib_pfunFintype___redArg___closed__4 = (const lean_object*)&lp_mathlib_pfunFintype___redArg___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_pfunFintype___redArg(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_pfunFintype___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_pfunFintype(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_pfunFintype___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_truncOfMultisetExistsMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_truncOfMultisetExistsMem___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_truncOfMultisetExistsMem(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_truncOfMultisetExistsMem___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_truncOfNonemptyFintype___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_truncOfNonemptyFintype___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_truncOfNonemptyFintype(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_truncOfNonemptyFintype___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_truncSigmaOfExists___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_truncSigmaOfExists(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_fintype(lean_object* v_n_1_){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = l_List_finRange(v_n_1_);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_fintype___redArg(lean_object* v_inst_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lp_mathlib_Fintype_ofSubsingleton___redArg(v_inst_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_fintype(lean_object* v_00_u03b1_5_, lean_object* v_inst_6_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lp_mathlib_Fintype_ofSubsingleton___redArg(v_inst_6_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_subtypeEq___redArg(lean_object* v_y_8_){
_start:
{
lean_object* v___x_9_; lean_object* v___x_10_; lean_object* v___x_11_; 
v___x_9_ = lean_box(0);
v___x_10_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_10_, 0, v_y_8_);
lean_ctor_set(v___x_10_, 1, v___x_9_);
v___x_11_ = lp_mathlib_Fintype_subtype___redArg(v___x_10_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_subtypeEq(lean_object* v_00_u03b1_12_, lean_object* v_y_13_){
_start:
{
lean_object* v___x_14_; 
v___x_14_ = lp_mathlib_Fintype_subtypeEq___redArg(v_y_13_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_subtypeEq_x27___redArg(lean_object* v_y_15_){
_start:
{
lean_object* v___x_16_; lean_object* v___x_17_; lean_object* v___x_18_; 
v___x_16_ = lean_box(0);
v___x_17_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_17_, 0, v_y_15_);
lean_ctor_set(v___x_17_, 1, v___x_16_);
v___x_18_ = lp_mathlib_Fintype_subtype___redArg(v___x_17_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_subtypeEq_x27(lean_object* v_00_u03b1_19_, lean_object* v_y_20_){
_start:
{
lean_object* v___x_21_; 
v___x_21_ = lp_mathlib_Fintype_subtypeEq_x27___redArg(v_y_20_);
return v___x_21_;
}
}
static lean_object* _init_lp_mathlib_Unit_fintype___closed__0(void){
_start:
{
lean_object* v___x_22_; lean_object* v___x_23_; 
v___x_22_ = lean_box(0);
v___x_23_ = lp_mathlib_Fintype_ofSubsingleton___redArg(v___x_22_);
return v___x_23_;
}
}
static lean_object* _init_lp_mathlib_Unit_fintype(void){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = lean_obj_once(&lp_mathlib_Unit_fintype___closed__0, &lp_mathlib_Unit_fintype___closed__0_once, _init_lp_mathlib_Unit_fintype___closed__0);
return v___x_24_;
}
}
static lean_object* _init_lp_mathlib_PUnit_fintype(void){
_start:
{
lean_object* v___x_25_; 
v___x_25_ = lean_obj_once(&lp_mathlib_Unit_fintype___closed__0, &lp_mathlib_Unit_fintype___closed__0_once, _init_lp_mathlib_Unit_fintype___closed__0);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_prodLeft___redArg___lam__0(lean_object* v_self_26_){
_start:
{
lean_object* v_fst_27_; 
v_fst_27_ = lean_ctor_get(v_self_26_, 0);
lean_inc(v_fst_27_);
return v_fst_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_prodLeft___redArg___lam__0___boxed(lean_object* v_self_28_){
_start:
{
lean_object* v_res_29_; 
v_res_29_ = lp_mathlib_Fintype_prodLeft___redArg___lam__0(v_self_28_);
lean_dec_ref(v_self_28_);
return v_res_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_prodLeft___redArg(lean_object* v_inst_31_, lean_object* v_inst_32_){
_start:
{
lean_object* v___f_33_; lean_object* v___x_34_; 
v___f_33_ = ((lean_object*)(lp_mathlib_Fintype_prodLeft___redArg___closed__0));
v___x_34_ = lp_mathlib_Finset_image___redArg(v_inst_31_, v___f_33_, v_inst_32_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_prodLeft(lean_object* v_00_u03b1_35_, lean_object* v_00_u03b2_36_, lean_object* v_inst_37_, lean_object* v_inst_38_, lean_object* v_inst_39_){
_start:
{
lean_object* v___x_40_; 
v___x_40_ = lp_mathlib_Fintype_prodLeft___redArg(v_inst_37_, v_inst_38_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_prodRight___redArg___lam__0(lean_object* v_self_41_){
_start:
{
lean_object* v_snd_42_; 
v_snd_42_ = lean_ctor_get(v_self_41_, 1);
lean_inc(v_snd_42_);
return v_snd_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_prodRight___redArg___lam__0___boxed(lean_object* v_self_43_){
_start:
{
lean_object* v_res_44_; 
v_res_44_ = lp_mathlib_Fintype_prodRight___redArg___lam__0(v_self_43_);
lean_dec_ref(v_self_43_);
return v_res_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_prodRight___redArg(lean_object* v_inst_46_, lean_object* v_inst_47_){
_start:
{
lean_object* v___f_48_; lean_object* v___x_49_; 
v___f_48_ = ((lean_object*)(lp_mathlib_Fintype_prodRight___redArg___closed__0));
v___x_49_ = lp_mathlib_Finset_image___redArg(v_inst_46_, v___f_48_, v_inst_47_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_prodRight(lean_object* v_00_u03b1_50_, lean_object* v_00_u03b2_51_, lean_object* v_inst_52_, lean_object* v_inst_53_, lean_object* v_inst_54_){
_start:
{
lean_object* v___x_55_; 
v___x_55_ = lp_mathlib_Fintype_prodRight___redArg(v_inst_52_, v_inst_53_);
return v___x_55_;
}
}
static lean_object* _init_lp_mathlib_ULift_fintype___redArg___closed__0(void){
_start:
{
lean_object* v___x_56_; 
v___x_56_ = lp_mathlib_Equiv_ulift(lean_box(0));
return v___x_56_;
}
}
static lean_object* _init_lp_mathlib_ULift_fintype___redArg___closed__1(void){
_start:
{
lean_object* v___x_57_; lean_object* v___x_58_; 
v___x_57_ = lean_obj_once(&lp_mathlib_ULift_fintype___redArg___closed__0, &lp_mathlib_ULift_fintype___redArg___closed__0_once, _init_lp_mathlib_ULift_fintype___redArg___closed__0);
v___x_58_ = lp_mathlib_Equiv_symm___redArg(v___x_57_);
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_fintype___redArg(lean_object* v_inst_59_){
_start:
{
lean_object* v___x_60_; lean_object* v___x_61_; 
v___x_60_ = lean_obj_once(&lp_mathlib_ULift_fintype___redArg___closed__1, &lp_mathlib_ULift_fintype___redArg___closed__1_once, _init_lp_mathlib_ULift_fintype___redArg___closed__1);
v___x_61_ = lp_mathlib_Fintype_ofEquiv___redArg(v_inst_59_, v___x_60_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_fintype(lean_object* v_00_u03b1_62_, lean_object* v_inst_63_){
_start:
{
lean_object* v___x_64_; 
v___x_64_ = lp_mathlib_ULift_fintype___redArg(v_inst_63_);
return v___x_64_;
}
}
static lean_object* _init_lp_mathlib_PLift_fintype___redArg___closed__0(void){
_start:
{
lean_object* v___x_65_; 
v___x_65_ = lp_mathlib_Equiv_plift(lean_box(0));
return v___x_65_;
}
}
static lean_object* _init_lp_mathlib_PLift_fintype___redArg___closed__1(void){
_start:
{
lean_object* v___x_66_; lean_object* v___x_67_; 
v___x_66_ = lean_obj_once(&lp_mathlib_PLift_fintype___redArg___closed__0, &lp_mathlib_PLift_fintype___redArg___closed__0_once, _init_lp_mathlib_PLift_fintype___redArg___closed__0);
v___x_67_ = lp_mathlib_Equiv_symm___redArg(v___x_66_);
return v___x_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PLift_fintype___redArg(lean_object* v_inst_68_){
_start:
{
lean_object* v___x_69_; lean_object* v___x_70_; 
v___x_69_ = lean_obj_once(&lp_mathlib_PLift_fintype___redArg___closed__1, &lp_mathlib_PLift_fintype___redArg___closed__1_once, _init_lp_mathlib_PLift_fintype___redArg___closed__1);
v___x_70_ = lp_mathlib_Fintype_ofEquiv___redArg(v_inst_68_, v___x_69_);
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PLift_fintype(lean_object* v_00_u03b1_71_, lean_object* v_inst_72_){
_start:
{
lean_object* v___x_73_; 
v___x_73_ = lp_mathlib_PLift_fintype___redArg(v_inst_72_);
return v___x_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PLift_fintypeProp___redArg(uint8_t v_inst_76_){
_start:
{
if (v_inst_76_ == 0)
{
lean_object* v___x_77_; 
v___x_77_ = lean_box(0);
return v___x_77_;
}
else
{
lean_object* v___x_78_; 
v___x_78_ = ((lean_object*)(lp_mathlib_PLift_fintypeProp___redArg___closed__0));
return v___x_78_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_PLift_fintypeProp___redArg___boxed(lean_object* v_inst_79_){
_start:
{
uint8_t v_inst_34__boxed_80_; lean_object* v_res_81_; 
v_inst_34__boxed_80_ = lean_unbox(v_inst_79_);
v_res_81_ = lp_mathlib_PLift_fintypeProp___redArg(v_inst_34__boxed_80_);
return v_res_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PLift_fintypeProp(lean_object* v_p_82_, uint8_t v_inst_83_){
_start:
{
lean_object* v___x_84_; 
v___x_84_ = lp_mathlib_PLift_fintypeProp___redArg(v_inst_83_);
return v___x_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PLift_fintypeProp___boxed(lean_object* v_p_85_, lean_object* v_inst_86_){
_start:
{
uint8_t v_inst_43__boxed_87_; lean_object* v_res_88_; 
v_inst_43__boxed_87_ = lean_unbox(v_inst_86_);
v_res_88_ = lp_mathlib_PLift_fintypeProp(v_p_85_, v_inst_43__boxed_87_);
return v_res_88_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Quotient_fintype___redArg___lam__0(lean_object* v_inst_89_, lean_object* v_a_90_, lean_object* v_b_91_){
_start:
{
lean_object* v___x_92_; uint8_t v___x_93_; 
v___x_92_ = lean_apply_2(v_inst_89_, v_a_90_, v_b_91_);
v___x_93_ = lean_unbox(v___x_92_);
return v___x_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_fintype___redArg___lam__0___boxed(lean_object* v_inst_94_, lean_object* v_a_95_, lean_object* v_b_96_){
_start:
{
uint8_t v_res_97_; lean_object* v_r_98_; 
v_res_97_ = lp_mathlib_Quotient_fintype___redArg___lam__0(v_inst_94_, v_a_95_, v_b_96_);
v_r_98_ = lean_box(v_res_97_);
return v_r_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_fintype___redArg(lean_object* v_inst_99_, lean_object* v_s_100_, lean_object* v_inst_101_){
_start:
{
lean_object* v___f_102_; lean_object* v___x_103_; lean_object* v___x_104_; 
v___f_102_ = lean_alloc_closure((void*)(lp_mathlib_Quotient_fintype___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_102_, 0, v_inst_101_);
v___x_103_ = lean_alloc_closure((void*)(lp_mathlib_Quotient_mk_x27_x27___boxed), 3, 2);
lean_closure_set(v___x_103_, 0, lean_box(0));
lean_closure_set(v___x_103_, 1, v_s_100_);
v___x_104_ = lp_mathlib_Finset_image___redArg(v___f_102_, v___x_103_, v_inst_99_);
return v___x_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_fintype(lean_object* v_00_u03b1_105_, lean_object* v_inst_106_, lean_object* v_s_107_, lean_object* v_inst_108_){
_start:
{
lean_object* v___x_109_; 
v___x_109_ = lp_mathlib_Quotient_fintype___redArg(v_inst_106_, v_s_107_, v_inst_108_);
return v___x_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSigma_fintypePropLeft___redArg___lam__0(lean_object* v_x_110_){
_start:
{
lean_object* v___x_111_; 
v___x_111_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_111_, 0, lean_box(0));
lean_ctor_set(v___x_111_, 1, v_x_110_);
return v___x_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSigma_fintypePropLeft___redArg___lam__1(lean_object* v_self_112_){
_start:
{
lean_object* v_snd_113_; 
v_snd_113_ = lean_ctor_get(v_self_112_, 1);
lean_inc(v_snd_113_);
return v_snd_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSigma_fintypePropLeft___redArg___lam__1___boxed(lean_object* v_self_114_){
_start:
{
lean_object* v_res_115_; 
v_res_115_ = lp_mathlib_PSigma_fintypePropLeft___redArg___lam__1(v_self_114_);
lean_dec_ref(v_self_114_);
return v_res_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSigma_fintypePropLeft___redArg(uint8_t v_inst_121_, lean_object* v_inst_122_){
_start:
{
if (v_inst_121_ == 0)
{
lean_object* v___x_123_; 
lean_dec(v_inst_122_);
v___x_123_ = lean_box(0);
return v___x_123_;
}
else
{
lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; 
v___x_124_ = lean_apply_1(v_inst_122_, lean_box(0));
v___x_125_ = ((lean_object*)(lp_mathlib_PSigma_fintypePropLeft___redArg___closed__2));
v___x_126_ = lp_mathlib_Fintype_ofEquiv___redArg(v___x_124_, v___x_125_);
return v___x_126_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSigma_fintypePropLeft___redArg___boxed(lean_object* v_inst_127_, lean_object* v_inst_128_){
_start:
{
uint8_t v_inst_34__boxed_129_; lean_object* v_res_130_; 
v_inst_34__boxed_129_ = lean_unbox(v_inst_127_);
v_res_130_ = lp_mathlib_PSigma_fintypePropLeft___redArg(v_inst_34__boxed_129_, v_inst_128_);
return v_res_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSigma_fintypePropLeft(lean_object* v_00_u03b1_131_, lean_object* v_00_u03b2_132_, uint8_t v_inst_133_, lean_object* v_inst_134_){
_start:
{
lean_object* v___x_135_; 
v___x_135_ = lp_mathlib_PSigma_fintypePropLeft___redArg(v_inst_133_, v_inst_134_);
return v___x_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSigma_fintypePropLeft___boxed(lean_object* v_00_u03b1_136_, lean_object* v_00_u03b2_137_, lean_object* v_inst_138_, lean_object* v_inst_139_){
_start:
{
uint8_t v_inst_53__boxed_140_; lean_object* v_res_141_; 
v_inst_53__boxed_140_ = lean_unbox(v_inst_138_);
v_res_141_ = lp_mathlib_PSigma_fintypePropLeft(v_00_u03b1_136_, v_00_u03b2_137_, v_inst_53__boxed_140_, v_inst_139_);
return v_res_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSigma_fintypePropRight___redArg___lam__0(lean_object* v_x_142_){
_start:
{
lean_object* v___x_143_; 
v___x_143_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_143_, 0, v_x_142_);
lean_ctor_set(v___x_143_, 1, lean_box(0));
return v___x_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSigma_fintypePropRight___redArg___lam__1(lean_object* v_x_144_){
_start:
{
lean_object* v_fst_145_; 
v_fst_145_ = lean_ctor_get(v_x_144_, 0);
lean_inc(v_fst_145_);
return v_fst_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSigma_fintypePropRight___redArg___lam__1___boxed(lean_object* v_x_146_){
_start:
{
lean_object* v_res_147_; 
v_res_147_ = lp_mathlib_PSigma_fintypePropRight___redArg___lam__1(v_x_146_);
lean_dec_ref(v_x_146_);
return v_res_147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSigma_fintypePropRight___redArg(lean_object* v_inst_153_, lean_object* v_inst_154_){
_start:
{
lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; 
v___x_155_ = lp_mathlib_Subtype_fintype___redArg(v_inst_153_, v_inst_154_);
v___x_156_ = ((lean_object*)(lp_mathlib_PSigma_fintypePropRight___redArg___closed__2));
v___x_157_ = lp_mathlib_Fintype_ofEquiv___redArg(v___x_155_, v___x_156_);
return v___x_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSigma_fintypePropRight(lean_object* v_00_u03b1_158_, lean_object* v_00_u03b2_159_, lean_object* v_inst_160_, lean_object* v_inst_161_){
_start:
{
lean_object* v___x_162_; 
v___x_162_ = lp_mathlib_PSigma_fintypePropRight___redArg(v_inst_160_, v_inst_161_);
return v___x_162_;
}
}
static lean_object* _init_lp_mathlib_PSigma_fintypePropProp___redArg___closed__0(void){
_start:
{
lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; 
v___x_163_ = lean_box(0);
v___x_164_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_164_, 0, lean_box(0));
lean_ctor_set(v___x_164_, 1, lean_box(0));
v___x_165_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_165_, 0, v___x_164_);
lean_ctor_set(v___x_165_, 1, v___x_163_);
return v___x_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSigma_fintypePropProp___redArg(uint8_t v_inst_166_, lean_object* v_inst_167_){
_start:
{
if (v_inst_166_ == 0)
{
lean_object* v___x_168_; 
lean_dec_ref(v_inst_167_);
v___x_168_ = lean_box(0);
return v___x_168_;
}
else
{
lean_object* v___x_169_; uint8_t v___x_170_; 
v___x_169_ = lean_apply_1(v_inst_167_, lean_box(0));
v___x_170_ = lean_unbox(v___x_169_);
if (v___x_170_ == 0)
{
lean_object* v___x_171_; 
v___x_171_ = lean_box(0);
return v___x_171_;
}
else
{
lean_object* v___x_172_; 
v___x_172_ = lean_obj_once(&lp_mathlib_PSigma_fintypePropProp___redArg___closed__0, &lp_mathlib_PSigma_fintypePropProp___redArg___closed__0_once, _init_lp_mathlib_PSigma_fintypePropProp___redArg___closed__0);
return v___x_172_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSigma_fintypePropProp___redArg___boxed(lean_object* v_inst_173_, lean_object* v_inst_174_){
_start:
{
uint8_t v_inst_46__boxed_175_; lean_object* v_res_176_; 
v_inst_46__boxed_175_ = lean_unbox(v_inst_173_);
v_res_176_ = lp_mathlib_PSigma_fintypePropProp___redArg(v_inst_46__boxed_175_, v_inst_174_);
return v_res_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSigma_fintypePropProp(lean_object* v_00_u03b1_177_, lean_object* v_00_u03b2_178_, uint8_t v_inst_179_, lean_object* v_inst_180_){
_start:
{
lean_object* v___x_181_; 
v___x_181_ = lp_mathlib_PSigma_fintypePropProp___redArg(v_inst_179_, v_inst_180_);
return v___x_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSigma_fintypePropProp___boxed(lean_object* v_00_u03b1_182_, lean_object* v_00_u03b2_183_, lean_object* v_inst_184_, lean_object* v_inst_185_){
_start:
{
uint8_t v_inst_65__boxed_186_; lean_object* v_res_187_; 
v_inst_65__boxed_186_ = lean_unbox(v_inst_184_);
v_res_187_ = lp_mathlib_PSigma_fintypePropProp(v_00_u03b1_182_, v_00_u03b2_183_, v_inst_65__boxed_186_, v_inst_185_);
return v_res_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_pfunFintype___redArg___lam__0(lean_object* v_h_188_){
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib_pfunFintype___redArg___lam__1(lean_object* v_a_189_, lean_object* v_x_190_){
_start:
{
lean_inc(v_a_189_);
return v_a_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_pfunFintype___redArg___lam__1___boxed(lean_object* v_a_191_, lean_object* v_x_192_){
_start:
{
lean_object* v_res_193_; 
v_res_193_ = lp_mathlib_pfunFintype___redArg___lam__1(v_a_191_, v_x_192_);
lean_dec(v_a_191_);
return v_res_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_pfunFintype___redArg___lam__2(lean_object* v_f_194_){
_start:
{
lean_object* v___x_195_; 
v___x_195_ = lean_apply_1(v_f_194_, lean_box(0));
return v___x_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_pfunFintype___redArg(uint8_t v_inst_205_, lean_object* v_inst_206_){
_start:
{
if (v_inst_205_ == 0)
{
lean_object* v___x_207_; 
lean_dec(v_inst_206_);
v___x_207_ = ((lean_object*)(lp_mathlib_pfunFintype___redArg___closed__1));
return v___x_207_;
}
else
{
lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_210_; 
v___x_208_ = lean_apply_1(v_inst_206_, lean_box(0));
v___x_209_ = ((lean_object*)(lp_mathlib_pfunFintype___redArg___closed__4));
v___x_210_ = lp_mathlib_Fintype_ofEquiv___redArg(v___x_208_, v___x_209_);
return v___x_210_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_pfunFintype___redArg___boxed(lean_object* v_inst_211_, lean_object* v_inst_212_){
_start:
{
uint8_t v_inst_56__boxed_213_; lean_object* v_res_214_; 
v_inst_56__boxed_213_ = lean_unbox(v_inst_211_);
v_res_214_ = lp_mathlib_pfunFintype___redArg(v_inst_56__boxed_213_, v_inst_212_);
return v_res_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_pfunFintype(lean_object* v_p_215_, uint8_t v_inst_216_, lean_object* v_00_u03b1_217_, lean_object* v_inst_218_){
_start:
{
lean_object* v___x_219_; 
v___x_219_ = lp_mathlib_pfunFintype___redArg(v_inst_216_, v_inst_218_);
return v___x_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_pfunFintype___boxed(lean_object* v_p_220_, lean_object* v_inst_221_, lean_object* v_00_u03b1_222_, lean_object* v_inst_223_){
_start:
{
uint8_t v_inst_79__boxed_224_; lean_object* v_res_225_; 
v_inst_79__boxed_224_ = lean_unbox(v_inst_221_);
v_res_225_ = lp_mathlib_pfunFintype(v_p_220_, v_inst_79__boxed_224_, v_00_u03b1_222_, v_inst_223_);
return v_res_225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_truncOfMultisetExistsMem___redArg(lean_object* v_s_226_){
_start:
{
lean_object* v_head_227_; 
v_head_227_ = lean_ctor_get(v_s_226_, 0);
lean_inc(v_head_227_);
return v_head_227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_truncOfMultisetExistsMem___redArg___boxed(lean_object* v_s_228_){
_start:
{
lean_object* v_res_229_; 
v_res_229_ = lp_mathlib_truncOfMultisetExistsMem___redArg(v_s_228_);
lean_dec(v_s_228_);
return v_res_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_truncOfMultisetExistsMem(lean_object* v_00_u03b1_230_, lean_object* v_s_231_, lean_object* v_a_232_){
_start:
{
lean_object* v_head_233_; 
v_head_233_ = lean_ctor_get(v_s_231_, 0);
lean_inc(v_head_233_);
return v_head_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_truncOfMultisetExistsMem___boxed(lean_object* v_00_u03b1_234_, lean_object* v_s_235_, lean_object* v_a_236_){
_start:
{
lean_object* v_res_237_; 
v_res_237_ = lp_mathlib_truncOfMultisetExistsMem(v_00_u03b1_234_, v_s_235_, v_a_236_);
lean_dec(v_s_235_);
return v_res_237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_truncOfNonemptyFintype___redArg(lean_object* v_inst_238_){
_start:
{
lean_object* v_head_239_; 
v_head_239_ = lean_ctor_get(v_inst_238_, 0);
lean_inc(v_head_239_);
return v_head_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_truncOfNonemptyFintype___redArg___boxed(lean_object* v_inst_240_){
_start:
{
lean_object* v_res_241_; 
v_res_241_ = lp_mathlib_truncOfNonemptyFintype___redArg(v_inst_240_);
lean_dec(v_inst_240_);
return v_res_241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_truncOfNonemptyFintype(lean_object* v_00_u03b1_242_, lean_object* v_inst_243_, lean_object* v_inst_244_){
_start:
{
lean_object* v_head_245_; 
v_head_245_ = lean_ctor_get(v_inst_244_, 0);
lean_inc(v_head_245_);
return v_head_245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_truncOfNonemptyFintype___boxed(lean_object* v_00_u03b1_246_, lean_object* v_inst_247_, lean_object* v_inst_248_){
_start:
{
lean_object* v_res_249_; 
v_res_249_ = lp_mathlib_truncOfNonemptyFintype(v_00_u03b1_246_, v_inst_247_, v_inst_248_);
lean_dec(v_inst_248_);
return v_res_249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_truncSigmaOfExists___redArg(lean_object* v_inst_250_, lean_object* v_inst_251_){
_start:
{
lean_object* v___x_252_; lean_object* v_head_253_; 
v___x_252_ = lp_mathlib_PSigma_fintypePropRight___redArg(v_inst_251_, v_inst_250_);
v_head_253_ = lean_ctor_get(v___x_252_, 0);
lean_inc(v_head_253_);
lean_dec(v___x_252_);
return v_head_253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_truncSigmaOfExists(lean_object* v_00_u03b1_254_, lean_object* v_inst_255_, lean_object* v_P_256_, lean_object* v_inst_257_, lean_object* v_h_258_){
_start:
{
lean_object* v___x_259_; 
v___x_259_ = lp_mathlib_truncSigmaOfExists___redArg(v_inst_255_, v_inst_257_);
return v___x_259_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_Finite_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_BooleanAlgebra(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Image(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_OfMap(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Sets(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_FinRange(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_OfFn(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Finite_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_BooleanAlgebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_OfMap(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Sets(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_FinRange(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_OfFn(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Unit_fintype = _init_lp_mathlib_Unit_fintype();
lean_mark_persistent(lp_mathlib_Unit_fintype);
lp_mathlib_PUnit_fintype = _init_lp_mathlib_PUnit_fintype();
lean_mark_persistent(lp_mathlib_PUnit_fintype);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Fintype_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Basic_Finite_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_BooleanAlgebra(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Image(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_OfMap(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Sets(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_FinRange(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_OfFn(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Fintype_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_Finite_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_BooleanAlgebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_OfMap(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Sets(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_FinRange(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_OfFn(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Fintype_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Fintype_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
