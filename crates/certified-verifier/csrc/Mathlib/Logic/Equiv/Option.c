// Lean compiler output
// Module: Mathlib.Logic.Equiv.Option
// Imports: public import Init public meta import Init public import Mathlib.Control.EquivFunctor public import Mathlib.Data.Option.Basic public import Mathlib.Data.Subtype public import Mathlib.Logic.Equiv.Defs
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
lean_object* l_Sum_elim___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Option_casesOn_x27___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionCongr___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionCongr___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionCongr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionCongr(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_removeNoneAux___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_removeNoneAux___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_removeNoneAux(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_removeNone__aux___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_removeNone__aux(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_removeNone___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_removeNone(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___redArg___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___redArg___lam__3(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___redArg___lam__3___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___redArg___lam__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___redArg___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___redArg___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___redArg___lam__7(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_optionSubtype___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_optionSubtype___redArg___lam__2, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_optionSubtype___redArg___closed__0 = (const lean_object*)&lp_mathlib_Equiv_optionSubtype___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_optionSubtype___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_optionSubtype___redArg___lam__3___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_optionSubtype___redArg___closed__1 = (const lean_object*)&lp_mathlib_Equiv_optionSubtype___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_optionSubtypeNe___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_optionSubtypeNe___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtypeNe___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtypeNe(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Equiv_optionEquivSumPUnit___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Equiv_optionEquivSumPUnit___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Equiv_optionEquivSumPUnit___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionEquivSumPUnit___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionEquivSumPUnit___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionEquivSumPUnit___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionEquivSumPUnit___lam__3(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_optionEquivSumPUnit___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_optionEquivSumPUnit___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_optionEquivSumPUnit___closed__0 = (const lean_object*)&lp_mathlib_Equiv_optionEquivSumPUnit___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_optionEquivSumPUnit___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_optionEquivSumPUnit___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_optionEquivSumPUnit___closed__1 = (const lean_object*)&lp_mathlib_Equiv_optionEquivSumPUnit___closed__1_value;
static const lean_closure_object lp_mathlib_Equiv_optionEquivSumPUnit___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_optionEquivSumPUnit___lam__2, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_optionEquivSumPUnit___closed__2 = (const lean_object*)&lp_mathlib_Equiv_optionEquivSumPUnit___closed__2_value;
static const lean_closure_object lp_mathlib_Equiv_optionEquivSumPUnit___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_optionEquivSumPUnit___lam__3, .m_arity = 3, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_Equiv_optionEquivSumPUnit___closed__1_value),((lean_object*)&lp_mathlib_Equiv_optionEquivSumPUnit___closed__2_value)} };
static const lean_object* lp_mathlib_Equiv_optionEquivSumPUnit___closed__3 = (const lean_object*)&lp_mathlib_Equiv_optionEquivSumPUnit___closed__3_value;
static const lean_ctor_object lp_mathlib_Equiv_optionEquivSumPUnit___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_optionEquivSumPUnit___closed__0_value),((lean_object*)&lp_mathlib_Equiv_optionEquivSumPUnit___closed__3_value)}};
static const lean_object* lp_mathlib_Equiv_optionEquivSumPUnit___closed__4 = (const lean_object*)&lp_mathlib_Equiv_optionEquivSumPUnit___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionEquivSumPUnit(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionIsSomeEquiv___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionIsSomeEquiv___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionIsSomeEquiv___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_optionIsSomeEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_optionIsSomeEquiv___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_optionIsSomeEquiv___closed__0 = (const lean_object*)&lp_mathlib_Equiv_optionIsSomeEquiv___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_optionIsSomeEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_optionIsSomeEquiv___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_optionIsSomeEquiv___closed__1 = (const lean_object*)&lp_mathlib_Equiv_optionIsSomeEquiv___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_optionIsSomeEquiv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_optionIsSomeEquiv___closed__0_value),((lean_object*)&lp_mathlib_Equiv_optionIsSomeEquiv___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_optionIsSomeEquiv___closed__2 = (const lean_object*)&lp_mathlib_Equiv_optionIsSomeEquiv___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionIsSomeEquiv(lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_subtypeNeSumPUnit___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_subtypeNeSumPUnit___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Equiv_subtypeNeSumPUnit___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_subtypeNeSumPUnit___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeNeSumPUnit___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeNeSumPUnit(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionCongr___redArg___lam__0(lean_object* v_e_1_, lean_object* v___y_2_){
_start:
{
if (lean_obj_tag(v___y_2_) == 0)
{
lean_object* v___x_3_; 
lean_dec_ref(v_e_1_);
v___x_3_ = lean_box(0);
return v___x_3_;
}
else
{
lean_object* v_val_4_; lean_object* v___x_6_; uint8_t v_isShared_7_; uint8_t v_isSharedCheck_13_; 
v_val_4_ = lean_ctor_get(v___y_2_, 0);
v_isSharedCheck_13_ = !lean_is_exclusive(v___y_2_);
if (v_isSharedCheck_13_ == 0)
{
v___x_6_ = v___y_2_;
v_isShared_7_ = v_isSharedCheck_13_;
goto v_resetjp_5_;
}
else
{
lean_inc(v_val_4_);
lean_dec(v___y_2_);
v___x_6_ = lean_box(0);
v_isShared_7_ = v_isSharedCheck_13_;
goto v_resetjp_5_;
}
v_resetjp_5_:
{
lean_object* v_toFun_8_; lean_object* v___x_9_; lean_object* v___x_11_; 
v_toFun_8_ = lean_ctor_get(v_e_1_, 0);
lean_inc(v_toFun_8_);
lean_dec_ref(v_e_1_);
v___x_9_ = lean_apply_1(v_toFun_8_, v_val_4_);
if (v_isShared_7_ == 0)
{
lean_ctor_set(v___x_6_, 0, v___x_9_);
v___x_11_ = v___x_6_;
goto v_reusejp_10_;
}
else
{
lean_object* v_reuseFailAlloc_12_; 
v_reuseFailAlloc_12_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_12_, 0, v___x_9_);
v___x_11_ = v_reuseFailAlloc_12_;
goto v_reusejp_10_;
}
v_reusejp_10_:
{
return v___x_11_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionCongr___redArg___lam__1(lean_object* v___x_14_, lean_object* v___y_15_){
_start:
{
if (lean_obj_tag(v___y_15_) == 0)
{
lean_object* v___x_16_; 
lean_dec_ref(v___x_14_);
v___x_16_ = lean_box(0);
return v___x_16_;
}
else
{
lean_object* v_val_17_; lean_object* v___x_19_; uint8_t v_isShared_20_; uint8_t v_isSharedCheck_26_; 
v_val_17_ = lean_ctor_get(v___y_15_, 0);
v_isSharedCheck_26_ = !lean_is_exclusive(v___y_15_);
if (v_isSharedCheck_26_ == 0)
{
v___x_19_ = v___y_15_;
v_isShared_20_ = v_isSharedCheck_26_;
goto v_resetjp_18_;
}
else
{
lean_inc(v_val_17_);
lean_dec(v___y_15_);
v___x_19_ = lean_box(0);
v_isShared_20_ = v_isSharedCheck_26_;
goto v_resetjp_18_;
}
v_resetjp_18_:
{
lean_object* v_toFun_21_; lean_object* v___x_22_; lean_object* v___x_24_; 
v_toFun_21_ = lean_ctor_get(v___x_14_, 0);
lean_inc(v_toFun_21_);
lean_dec_ref(v___x_14_);
v___x_22_ = lean_apply_1(v_toFun_21_, v_val_17_);
if (v_isShared_20_ == 0)
{
lean_ctor_set(v___x_19_, 0, v___x_22_);
v___x_24_ = v___x_19_;
goto v_reusejp_23_;
}
else
{
lean_object* v_reuseFailAlloc_25_; 
v_reuseFailAlloc_25_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_25_, 0, v___x_22_);
v___x_24_ = v_reuseFailAlloc_25_;
goto v_reusejp_23_;
}
v_reusejp_23_:
{
return v___x_24_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionCongr___redArg(lean_object* v_e_27_){
_start:
{
lean_object* v___f_28_; lean_object* v___x_29_; lean_object* v___f_30_; lean_object* v___x_31_; 
lean_inc_ref(v_e_27_);
v___f_28_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_optionCongr___redArg___lam__0), 2, 1);
lean_closure_set(v___f_28_, 0, v_e_27_);
v___x_29_ = lp_mathlib_Equiv_symm___redArg(v_e_27_);
v___f_30_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_optionCongr___redArg___lam__1), 2, 1);
lean_closure_set(v___f_30_, 0, v___x_29_);
v___x_31_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_31_, 0, v___f_28_);
lean_ctor_set(v___x_31_, 1, v___f_30_);
return v___x_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionCongr(lean_object* v_00_u03b1_32_, lean_object* v_00_u03b2_33_, lean_object* v_e_34_){
_start:
{
lean_object* v___x_35_; 
v___x_35_ = lp_mathlib_Equiv_optionCongr___redArg(v_e_34_);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_removeNoneAux___redArg___lam__0(lean_object* v_self_36_, lean_object* v___y_37_){
_start:
{
lean_object* v_toFun_38_; lean_object* v___x_39_; 
v_toFun_38_ = lean_ctor_get(v_self_36_, 0);
lean_inc(v_toFun_38_);
lean_dec_ref(v_self_36_);
v___x_39_ = lean_apply_1(v_toFun_38_, v___y_37_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_removeNoneAux___redArg(lean_object* v_e_40_, lean_object* v_x_41_){
_start:
{
lean_object* v___x_42_; lean_object* v___x_43_; 
v___x_42_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_42_, 0, v_x_41_);
lean_inc_ref(v_e_40_);
v___x_43_ = lp_mathlib_Equiv_removeNoneAux___redArg___lam__0(v_e_40_, v___x_42_);
if (lean_obj_tag(v___x_43_) == 0)
{
lean_object* v___x_44_; lean_object* v___x_45_; lean_object* v_val_46_; 
v___x_44_ = lean_box(0);
v___x_45_ = lp_mathlib_Equiv_removeNoneAux___redArg___lam__0(v_e_40_, v___x_44_);
v_val_46_ = lean_ctor_get(v___x_45_, 0);
lean_inc(v_val_46_);
lean_dec(v___x_45_);
return v_val_46_;
}
else
{
lean_object* v_val_47_; 
lean_dec_ref(v_e_40_);
v_val_47_ = lean_ctor_get(v___x_43_, 0);
lean_inc(v_val_47_);
lean_dec_ref_known(v___x_43_, 1);
return v_val_47_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_removeNoneAux(lean_object* v_00_u03b1_48_, lean_object* v_00_u03b2_49_, lean_object* v_e_50_, lean_object* v_x_51_){
_start:
{
lean_object* v___x_52_; 
v___x_52_ = lp_mathlib_Equiv_removeNoneAux___redArg(v_e_50_, v_x_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_removeNone__aux___redArg(lean_object* v_e_53_, lean_object* v_x_54_){
_start:
{
lean_object* v___x_55_; 
v___x_55_ = lp_mathlib_Equiv_removeNoneAux___redArg(v_e_53_, v_x_54_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_removeNone__aux(lean_object* v_00_u03b1_56_, lean_object* v_00_u03b2_57_, lean_object* v_e_58_, lean_object* v_x_59_){
_start:
{
lean_object* v___x_60_; 
v___x_60_ = lp_mathlib_Equiv_removeNoneAux___redArg(v_e_58_, v_x_59_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_removeNone___redArg(lean_object* v_e_61_){
_start:
{
lean_object* v___x_62_; lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; 
lean_inc_ref(v_e_61_);
v___x_62_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_removeNoneAux), 4, 3);
lean_closure_set(v___x_62_, 0, lean_box(0));
lean_closure_set(v___x_62_, 1, lean_box(0));
lean_closure_set(v___x_62_, 2, v_e_61_);
v___x_63_ = lp_mathlib_Equiv_symm___redArg(v_e_61_);
v___x_64_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_removeNoneAux), 4, 3);
lean_closure_set(v___x_64_, 0, lean_box(0));
lean_closure_set(v___x_64_, 1, lean_box(0));
lean_closure_set(v___x_64_, 2, v___x_63_);
v___x_65_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_65_, 0, v___x_62_);
lean_ctor_set(v___x_65_, 1, v___x_64_);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_removeNone(lean_object* v_00_u03b1_66_, lean_object* v_00_u03b2_67_, lean_object* v_e_68_){
_start:
{
lean_object* v___x_69_; 
v___x_69_ = lp_mathlib_Equiv_removeNone___redArg(v_e_68_);
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___redArg___lam__0(lean_object* v_e_70_, lean_object* v_a_71_){
_start:
{
lean_object* v_toFun_72_; lean_object* v___x_73_; lean_object* v___x_74_; 
v_toFun_72_ = lean_ctor_get(v_e_70_, 0);
lean_inc(v_toFun_72_);
lean_dec_ref(v_e_70_);
v___x_73_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_73_, 0, v_a_71_);
v___x_74_ = lean_apply_1(v_toFun_72_, v___x_73_);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___redArg___lam__1(lean_object* v_e_75_, lean_object* v_b_76_){
_start:
{
lean_object* v___x_77_; lean_object* v_toFun_78_; lean_object* v___x_79_; lean_object* v_val_80_; 
v___x_77_ = lp_mathlib_Equiv_symm___redArg(v_e_75_);
v_toFun_78_ = lean_ctor_get(v___x_77_, 0);
lean_inc(v_toFun_78_);
lean_dec_ref(v___x_77_);
v___x_79_ = lean_apply_1(v_toFun_78_, v_b_76_);
v_val_80_ = lean_ctor_get(v___x_79_, 0);
lean_inc(v_val_80_);
lean_dec(v___x_79_);
return v_val_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___redArg___lam__2(lean_object* v_e_81_){
_start:
{
lean_object* v___f_82_; lean_object* v___f_83_; lean_object* v___x_84_; 
lean_inc_ref(v_e_81_);
v___f_82_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_optionSubtype___redArg___lam__0), 2, 1);
lean_closure_set(v___f_82_, 0, v_e_81_);
v___f_83_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_optionSubtype___redArg___lam__1), 2, 1);
lean_closure_set(v___f_83_, 0, v_e_81_);
v___x_84_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_84_, 0, v___f_82_);
lean_ctor_set(v___x_84_, 1, v___f_83_);
return v___x_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___redArg___lam__3(lean_object* v_self_85_){
_start:
{
lean_inc(v_self_85_);
return v_self_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___redArg___lam__3___boxed(lean_object* v_self_86_){
_start:
{
lean_object* v_res_87_; 
v_res_87_ = lp_mathlib_Equiv_optionSubtype___redArg___lam__3(v_self_86_);
lean_dec(v_self_86_);
return v_res_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___redArg___lam__4(lean_object* v_e_88_, lean_object* v___y_89_){
_start:
{
lean_object* v_toFun_90_; lean_object* v___x_91_; 
v_toFun_90_ = lean_ctor_get(v_e_88_, 0);
lean_inc(v_toFun_90_);
lean_dec_ref(v_e_88_);
v___x_91_ = lean_apply_1(v_toFun_90_, v___y_89_);
return v___x_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___redArg___lam__5(lean_object* v___f_92_, lean_object* v___f_93_, lean_object* v_x_94_, lean_object* v_a_95_){
_start:
{
lean_object* v___x_96_; lean_object* v___x_97_; 
v___x_96_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_96_, 0, lean_box(0));
lean_closure_set(v___x_96_, 1, lean_box(0));
lean_closure_set(v___x_96_, 2, lean_box(0));
lean_closure_set(v___x_96_, 3, v___f_92_);
lean_closure_set(v___x_96_, 4, v___f_93_);
v___x_97_ = lp_mathlib_Option_casesOn_x27___redArg(v_a_95_, v_x_94_, v___x_96_);
return v___x_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___redArg___lam__5___boxed(lean_object* v___f_98_, lean_object* v___f_99_, lean_object* v_x_100_, lean_object* v_a_101_){
_start:
{
lean_object* v_res_102_; 
v_res_102_ = lp_mathlib_Equiv_optionSubtype___redArg___lam__5(v___f_98_, v___f_99_, v_x_100_, v_a_101_);
lean_dec(v_x_100_);
return v_res_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___redArg___lam__6(lean_object* v_inst_103_, lean_object* v_x_104_, lean_object* v_e_105_, lean_object* v_b_106_){
_start:
{
lean_object* v___x_107_; uint8_t v___x_108_; 
lean_inc(v_b_106_);
v___x_107_ = lean_apply_2(v_inst_103_, v_b_106_, v_x_104_);
v___x_108_ = lean_unbox(v___x_107_);
if (v___x_108_ == 0)
{
lean_object* v___x_109_; lean_object* v_toFun_110_; lean_object* v___x_111_; lean_object* v___x_112_; 
v___x_109_ = lp_mathlib_Equiv_symm___redArg(v_e_105_);
v_toFun_110_ = lean_ctor_get(v___x_109_, 0);
lean_inc(v_toFun_110_);
lean_dec_ref(v___x_109_);
v___x_111_ = lean_apply_1(v_toFun_110_, v_b_106_);
v___x_112_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_112_, 0, v___x_111_);
return v___x_112_;
}
else
{
lean_object* v___x_113_; 
lean_dec(v_b_106_);
lean_dec_ref(v_e_105_);
v___x_113_ = lean_box(0);
return v___x_113_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___redArg___lam__7(lean_object* v___f_114_, lean_object* v_x_115_, lean_object* v_inst_116_, lean_object* v_e_117_){
_start:
{
lean_object* v___f_118_; lean_object* v___f_119_; lean_object* v___f_120_; lean_object* v___x_121_; 
lean_inc_ref(v_e_117_);
v___f_118_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_optionSubtype___redArg___lam__4), 2, 1);
lean_closure_set(v___f_118_, 0, v_e_117_);
lean_inc(v_x_115_);
v___f_119_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_optionSubtype___redArg___lam__5___boxed), 4, 3);
lean_closure_set(v___f_119_, 0, v___f_114_);
lean_closure_set(v___f_119_, 1, v___f_118_);
lean_closure_set(v___f_119_, 2, v_x_115_);
v___f_120_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_optionSubtype___redArg___lam__6), 4, 3);
lean_closure_set(v___f_120_, 0, v_inst_116_);
lean_closure_set(v___f_120_, 1, v_x_115_);
lean_closure_set(v___f_120_, 2, v_e_117_);
v___x_121_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_121_, 0, v___f_119_);
lean_ctor_set(v___x_121_, 1, v___f_120_);
return v___x_121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___redArg(lean_object* v_inst_124_, lean_object* v_x_125_){
_start:
{
lean_object* v___f_126_; lean_object* v___f_127_; lean_object* v___f_128_; lean_object* v___x_129_; 
v___f_126_ = ((lean_object*)(lp_mathlib_Equiv_optionSubtype___redArg___closed__0));
v___f_127_ = ((lean_object*)(lp_mathlib_Equiv_optionSubtype___redArg___closed__1));
v___f_128_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_optionSubtype___redArg___lam__7), 4, 3);
lean_closure_set(v___f_128_, 0, v___f_127_);
lean_closure_set(v___f_128_, 1, v_x_125_);
lean_closure_set(v___f_128_, 2, v_inst_124_);
v___x_129_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_129_, 0, v___f_126_);
lean_ctor_set(v___x_129_, 1, v___f_128_);
return v___x_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype(lean_object* v_00_u03b1_130_, lean_object* v_00_u03b2_131_, lean_object* v_inst_132_, lean_object* v_x_133_){
_start:
{
lean_object* v___x_134_; 
v___x_134_ = lp_mathlib_Equiv_optionSubtype___redArg(v_inst_132_, v_x_133_);
return v___x_134_;
}
}
static lean_object* _init_lp_mathlib_Equiv_optionSubtypeNe___redArg___closed__0(void){
_start:
{
lean_object* v___x_135_; 
v___x_135_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtypeNe___redArg(lean_object* v_inst_136_, lean_object* v_a_137_){
_start:
{
lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v_toFun_140_; lean_object* v___x_141_; lean_object* v___x_142_; 
v___x_138_ = lp_mathlib_Equiv_optionSubtype___redArg(v_inst_136_, v_a_137_);
v___x_139_ = lp_mathlib_Equiv_symm___redArg(v___x_138_);
v_toFun_140_ = lean_ctor_get(v___x_139_, 0);
lean_inc(v_toFun_140_);
lean_dec_ref(v___x_139_);
v___x_141_ = lean_obj_once(&lp_mathlib_Equiv_optionSubtypeNe___redArg___closed__0, &lp_mathlib_Equiv_optionSubtypeNe___redArg___closed__0_once, _init_lp_mathlib_Equiv_optionSubtypeNe___redArg___closed__0);
v___x_142_ = lean_apply_1(v_toFun_140_, v___x_141_);
return v___x_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtypeNe(lean_object* v_00_u03b1_143_, lean_object* v_inst_144_, lean_object* v_a_145_){
_start:
{
lean_object* v___x_146_; 
v___x_146_ = lp_mathlib_Equiv_optionSubtypeNe___redArg(v_inst_144_, v_a_145_);
return v___x_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionEquivSumPUnit___lam__0(lean_object* v_o_149_){
_start:
{
if (lean_obj_tag(v_o_149_) == 0)
{
lean_object* v___x_150_; 
v___x_150_ = ((lean_object*)(lp_mathlib_Equiv_optionEquivSumPUnit___lam__0___closed__0));
return v___x_150_;
}
else
{
lean_object* v_val_151_; lean_object* v___x_153_; uint8_t v_isShared_154_; uint8_t v_isSharedCheck_158_; 
v_val_151_ = lean_ctor_get(v_o_149_, 0);
v_isSharedCheck_158_ = !lean_is_exclusive(v_o_149_);
if (v_isSharedCheck_158_ == 0)
{
v___x_153_ = v_o_149_;
v_isShared_154_ = v_isSharedCheck_158_;
goto v_resetjp_152_;
}
else
{
lean_inc(v_val_151_);
lean_dec(v_o_149_);
v___x_153_ = lean_box(0);
v_isShared_154_ = v_isSharedCheck_158_;
goto v_resetjp_152_;
}
v_resetjp_152_:
{
lean_object* v___x_156_; 
if (v_isShared_154_ == 0)
{
lean_ctor_set_tag(v___x_153_, 0);
v___x_156_ = v___x_153_;
goto v_reusejp_155_;
}
else
{
lean_object* v_reuseFailAlloc_157_; 
v_reuseFailAlloc_157_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_157_, 0, v_val_151_);
v___x_156_ = v_reuseFailAlloc_157_;
goto v_reusejp_155_;
}
v_reusejp_155_:
{
return v___x_156_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionEquivSumPUnit___lam__1(lean_object* v_val_159_){
_start:
{
lean_object* v___x_160_; 
v___x_160_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_160_, 0, v_val_159_);
return v___x_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionEquivSumPUnit___lam__2(lean_object* v_x_161_){
_start:
{
lean_object* v___x_162_; 
v___x_162_ = lean_box(0);
return v___x_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionEquivSumPUnit___lam__3(lean_object* v___f_163_, lean_object* v___f_164_, lean_object* v_s_165_){
_start:
{
lean_object* v___x_166_; 
v___x_166_ = l_Sum_elim___redArg(v___f_163_, v___f_164_, v_s_165_);
return v___x_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionEquivSumPUnit(lean_object* v_00_u03b1_176_){
_start:
{
lean_object* v___x_177_; 
v___x_177_ = ((lean_object*)(lp_mathlib_Equiv_optionEquivSumPUnit___closed__4));
return v___x_177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionIsSomeEquiv___lam__0(lean_object* v_o_178_){
_start:
{
lean_object* v_val_179_; 
v_val_179_ = lean_ctor_get(v_o_178_, 0);
lean_inc(v_val_179_);
return v_val_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionIsSomeEquiv___lam__0___boxed(lean_object* v_o_180_){
_start:
{
lean_object* v_res_181_; 
v_res_181_ = lp_mathlib_Equiv_optionIsSomeEquiv___lam__0(v_o_180_);
lean_dec(v_o_180_);
return v_res_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionIsSomeEquiv___lam__1(lean_object* v_x_182_){
_start:
{
lean_object* v___x_183_; 
v___x_183_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_183_, 0, v_x_182_);
return v___x_183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionIsSomeEquiv(lean_object* v_00_u03b1_189_){
_start:
{
lean_object* v___x_190_; 
v___x_190_ = ((lean_object*)(lp_mathlib_Equiv_optionIsSomeEquiv___closed__2));
return v___x_190_;
}
}
static lean_object* _init_lp_mathlib_Equiv_subtypeNeSumPUnit___redArg___closed__0(void){
_start:
{
lean_object* v___x_191_; 
v___x_191_ = lp_mathlib_Equiv_optionEquivSumPUnit(lean_box(0));
return v___x_191_;
}
}
static lean_object* _init_lp_mathlib_Equiv_subtypeNeSumPUnit___redArg___closed__1(void){
_start:
{
lean_object* v___x_192_; lean_object* v___x_193_; 
v___x_192_ = lean_obj_once(&lp_mathlib_Equiv_subtypeNeSumPUnit___redArg___closed__0, &lp_mathlib_Equiv_subtypeNeSumPUnit___redArg___closed__0_once, _init_lp_mathlib_Equiv_subtypeNeSumPUnit___redArg___closed__0);
v___x_193_ = lp_mathlib_Equiv_symm___redArg(v___x_192_);
return v___x_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeNeSumPUnit___redArg(lean_object* v_inst_194_, lean_object* v_i_u2080_195_){
_start:
{
lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; 
v___x_196_ = lean_obj_once(&lp_mathlib_Equiv_subtypeNeSumPUnit___redArg___closed__1, &lp_mathlib_Equiv_subtypeNeSumPUnit___redArg___closed__1_once, _init_lp_mathlib_Equiv_subtypeNeSumPUnit___redArg___closed__1);
v___x_197_ = lp_mathlib_Equiv_optionSubtypeNe___redArg(v_inst_194_, v_i_u2080_195_);
v___x_198_ = lp_mathlib_Equiv_trans___redArg(v___x_196_, v___x_197_);
return v___x_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeNeSumPUnit(lean_object* v_00_u03b1_199_, lean_object* v_inst_200_, lean_object* v_i_u2080_201_){
_start:
{
lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; 
v___x_202_ = lean_obj_once(&lp_mathlib_Equiv_subtypeNeSumPUnit___redArg___closed__1, &lp_mathlib_Equiv_subtypeNeSumPUnit___redArg___closed__1_once, _init_lp_mathlib_Equiv_subtypeNeSumPUnit___redArg___closed__1);
v___x_203_ = lp_mathlib_Equiv_optionSubtypeNe___redArg(v_inst_200_, v_i_u2080_201_);
v___x_204_ = lp_mathlib_Equiv_trans___redArg(v___x_202_, v___x_203_);
return v___x_204_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Control_EquivFunctor(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Option_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Subtype(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Option(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Control_EquivFunctor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Option_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Subtype(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Logic_Equiv_Option(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Control_EquivFunctor(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Option_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Subtype(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Option(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Control_EquivFunctor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Option_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Subtype(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Option(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Logic_Equiv_Option(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Logic_Equiv_Option(builtin);
}
#ifdef __cplusplus
}
#endif
