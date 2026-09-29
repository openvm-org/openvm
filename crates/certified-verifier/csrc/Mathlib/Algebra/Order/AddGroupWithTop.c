// Lean compiler output
// Module: Mathlib.Algebra.Order.AddGroupWithTop
// Imports: public import Init public meta import Init public import Mathlib.Algebra.CharZero.Defs public import Mathlib.Algebra.Group.Hom.Defs public import Mathlib.Algebra.Order.Monoid.Canonical.Defs public import Mathlib.Algebra.Order.Monoid.WithTop public import Mathlib.Algebra.Regular.Basic
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
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_WithTop_map(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_WithTop_addMonoid___redArg(lean_object*);
lean_object* lp_mathlib_WithTop_linearOrder___redArg(lean_object*);
lean_object* lp_mathlib_WithTop_add___redArg(lean_object*);
lean_object* l_nsmulRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_zsmulRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_toSubNegMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_toSubNegMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_toSubNegMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_toSubNegMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_instLinearOrderedAddCommMonoidWithTop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_instLinearOrderedAddCommMonoidWithTop___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_instLinearOrderedAddCommMonoidWithTop(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_instLinearOrderedAddCommMonoidWithTop___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_toSubtractionMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_toSubtractionMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_toSubtractionMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_toSubtractionMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_linearOrderedAddCommMonoidWithTop___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_linearOrderedAddCommMonoidWithTop(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instNeg___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instNeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instNeg___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instNeg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instNeg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instSub___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instSub___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instSub___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instSub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instLinearOrderedAddCommGroupWithTopOfIsOrderedAddMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instLinearOrderedAddCommGroupWithTopOfIsOrderedAddMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_toSubNegMonoid___redArg(lean_object* v_self_1_){
_start:
{
lean_object* v_toAddCommMonoid_2_; lean_object* v_toNeg_3_; lean_object* v_toSub_4_; lean_object* v_toZSMul_5_; lean_object* v___x_6_; 
v_toAddCommMonoid_2_ = lean_ctor_get(v_self_1_, 0);
v_toNeg_3_ = lean_ctor_get(v_self_1_, 3);
v_toSub_4_ = lean_ctor_get(v_self_1_, 4);
v_toZSMul_5_ = lean_ctor_get(v_self_1_, 5);
lean_inc(v_toZSMul_5_);
lean_inc(v_toSub_4_);
lean_inc(v_toNeg_3_);
lean_inc_ref(v_toAddCommMonoid_2_);
v___x_6_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_6_, 0, v_toAddCommMonoid_2_);
lean_ctor_set(v___x_6_, 1, v_toNeg_3_);
lean_ctor_set(v___x_6_, 2, v_toSub_4_);
lean_ctor_set(v___x_6_, 3, v_toZSMul_5_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_toSubNegMonoid___redArg___boxed(lean_object* v_self_7_){
_start:
{
lean_object* v_res_8_; 
v_res_8_ = lp_mathlib_LinearOrderedAddCommGroupWithTop_toSubNegMonoid___redArg(v_self_7_);
lean_dec_ref(v_self_7_);
return v_res_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_toSubNegMonoid(lean_object* v_00_u03b1_9_, lean_object* v_self_10_){
_start:
{
lean_object* v___x_11_; 
v___x_11_ = lp_mathlib_LinearOrderedAddCommGroupWithTop_toSubNegMonoid___redArg(v_self_10_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_toSubNegMonoid___boxed(lean_object* v_00_u03b1_12_, lean_object* v_self_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_LinearOrderedAddCommGroupWithTop_toSubNegMonoid(v_00_u03b1_12_, v_self_13_);
lean_dec_ref(v_self_13_);
return v_res_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_instLinearOrderedAddCommMonoidWithTop___redArg(lean_object* v_inst_15_){
_start:
{
lean_object* v_toAddCommMonoid_16_; lean_object* v_toLinearOrder_17_; lean_object* v_toOrderTop_18_; lean_object* v___x_19_; 
v_toAddCommMonoid_16_ = lean_ctor_get(v_inst_15_, 0);
v_toLinearOrder_17_ = lean_ctor_get(v_inst_15_, 1);
v_toOrderTop_18_ = lean_ctor_get(v_inst_15_, 2);
lean_inc(v_toOrderTop_18_);
lean_inc_ref(v_toLinearOrder_17_);
lean_inc_ref(v_toAddCommMonoid_16_);
v___x_19_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_19_, 0, v_toAddCommMonoid_16_);
lean_ctor_set(v___x_19_, 1, v_toLinearOrder_17_);
lean_ctor_set(v___x_19_, 2, v_toOrderTop_18_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_instLinearOrderedAddCommMonoidWithTop___redArg___boxed(lean_object* v_inst_20_){
_start:
{
lean_object* v_res_21_; 
v_res_21_ = lp_mathlib_LinearOrderedAddCommGroupWithTop_instLinearOrderedAddCommMonoidWithTop___redArg(v_inst_20_);
lean_dec_ref(v_inst_20_);
return v_res_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_instLinearOrderedAddCommMonoidWithTop(lean_object* v_00_u03b1_22_, lean_object* v_inst_23_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = lp_mathlib_LinearOrderedAddCommGroupWithTop_instLinearOrderedAddCommMonoidWithTop___redArg(v_inst_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_instLinearOrderedAddCommMonoidWithTop___boxed(lean_object* v_00_u03b1_25_, lean_object* v_inst_26_){
_start:
{
lean_object* v_res_27_; 
v_res_27_ = lp_mathlib_LinearOrderedAddCommGroupWithTop_instLinearOrderedAddCommMonoidWithTop(v_00_u03b1_25_, v_inst_26_);
lean_dec_ref(v_inst_26_);
return v_res_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_toSubtractionMonoid___redArg(lean_object* v_inst_28_){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lp_mathlib_LinearOrderedAddCommGroupWithTop_toSubNegMonoid___redArg(v_inst_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_toSubtractionMonoid___redArg___boxed(lean_object* v_inst_30_){
_start:
{
lean_object* v_res_31_; 
v_res_31_ = lp_mathlib_LinearOrderedAddCommGroupWithTop_toSubtractionMonoid___redArg(v_inst_30_);
lean_dec_ref(v_inst_30_);
return v_res_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_toSubtractionMonoid(lean_object* v_00_u03b1_32_, lean_object* v_inst_33_){
_start:
{
lean_object* v___x_34_; 
v___x_34_ = lp_mathlib_LinearOrderedAddCommGroupWithTop_toSubNegMonoid___redArg(v_inst_33_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrderedAddCommGroupWithTop_toSubtractionMonoid___boxed(lean_object* v_00_u03b1_35_, lean_object* v_inst_36_){
_start:
{
lean_object* v_res_37_; 
v_res_37_ = lp_mathlib_LinearOrderedAddCommGroupWithTop_toSubtractionMonoid(v_00_u03b1_35_, v_inst_36_);
lean_dec_ref(v_inst_36_);
return v_res_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_linearOrderedAddCommMonoidWithTop___redArg(lean_object* v_inst_38_, lean_object* v_inst_39_){
_start:
{
lean_object* v___x_40_; lean_object* v___x_41_; lean_object* v___x_42_; lean_object* v___x_43_; 
v___x_40_ = lp_mathlib_WithTop_addMonoid___redArg(v_inst_38_);
v___x_41_ = lp_mathlib_WithTop_linearOrder___redArg(v_inst_39_);
v___x_42_ = lean_box(0);
v___x_43_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_43_, 0, v___x_40_);
lean_ctor_set(v___x_43_, 1, v___x_41_);
lean_ctor_set(v___x_43_, 2, v___x_42_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_linearOrderedAddCommMonoidWithTop(lean_object* v_00_u03b1_44_, lean_object* v_inst_45_, lean_object* v_inst_46_, lean_object* v_inst_47_){
_start:
{
lean_object* v___x_48_; 
v___x_48_ = lp_mathlib_WithTop_linearOrderedAddCommMonoidWithTop___redArg(v_inst_45_, v_inst_46_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instNeg___redArg___lam__0(lean_object* v_toNeg_49_, lean_object* v_a_50_){
_start:
{
lean_object* v___x_51_; 
v___x_51_ = lean_apply_1(v_toNeg_49_, v_a_50_);
return v___x_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instNeg___redArg(lean_object* v_inst_52_){
_start:
{
lean_object* v___x_53_; lean_object* v_toNeg_54_; lean_object* v___f_55_; lean_object* v___x_56_; 
v___x_53_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_52_);
v_toNeg_54_ = lean_ctor_get(v___x_53_, 1);
lean_inc(v_toNeg_54_);
lean_dec_ref(v___x_53_);
v___f_55_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_LinearOrderedAddCommGroup_instNeg___redArg___lam__0), 2, 1);
lean_closure_set(v___f_55_, 0, v_toNeg_54_);
v___x_56_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_map), 4, 3);
lean_closure_set(v___x_56_, 0, lean_box(0));
lean_closure_set(v___x_56_, 1, lean_box(0));
lean_closure_set(v___x_56_, 2, v___f_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instNeg___redArg___boxed(lean_object* v_inst_57_){
_start:
{
lean_object* v_res_58_; 
v_res_58_ = lp_mathlib_WithTop_LinearOrderedAddCommGroup_instNeg___redArg(v_inst_57_);
lean_dec_ref(v_inst_57_);
return v_res_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instNeg(lean_object* v_G_59_, lean_object* v_inst_60_){
_start:
{
lean_object* v___x_61_; 
v___x_61_ = lp_mathlib_WithTop_LinearOrderedAddCommGroup_instNeg___redArg(v_inst_60_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instNeg___boxed(lean_object* v_G_62_, lean_object* v_inst_63_){
_start:
{
lean_object* v_res_64_; 
v_res_64_ = lp_mathlib_WithTop_LinearOrderedAddCommGroup_instNeg(v_G_62_, v_inst_63_);
lean_dec_ref(v_inst_63_);
return v_res_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instSub___redArg___lam__0(lean_object* v___x_65_, lean_object* v_toSub_66_, lean_object* v_x_67_, lean_object* v_x_68_){
_start:
{
if (lean_obj_tag(v_x_68_) == 0)
{
lean_dec(v_x_67_);
lean_dec(v_toSub_66_);
lean_inc(v___x_65_);
return v___x_65_;
}
else
{
if (lean_obj_tag(v_x_67_) == 0)
{
lean_dec_ref_known(v_x_68_, 1);
lean_dec(v_toSub_66_);
lean_inc(v___x_65_);
return v___x_65_;
}
else
{
lean_object* v_val_69_; lean_object* v_val_70_; lean_object* v___x_72_; uint8_t v_isShared_73_; uint8_t v_isSharedCheck_78_; 
v_val_69_ = lean_ctor_get(v_x_68_, 0);
lean_inc(v_val_69_);
lean_dec_ref_known(v_x_68_, 1);
v_val_70_ = lean_ctor_get(v_x_67_, 0);
v_isSharedCheck_78_ = !lean_is_exclusive(v_x_67_);
if (v_isSharedCheck_78_ == 0)
{
v___x_72_ = v_x_67_;
v_isShared_73_ = v_isSharedCheck_78_;
goto v_resetjp_71_;
}
else
{
lean_inc(v_val_70_);
lean_dec(v_x_67_);
v___x_72_ = lean_box(0);
v_isShared_73_ = v_isSharedCheck_78_;
goto v_resetjp_71_;
}
v_resetjp_71_:
{
lean_object* v___x_74_; lean_object* v___x_76_; 
v___x_74_ = lean_apply_2(v_toSub_66_, v_val_70_, v_val_69_);
if (v_isShared_73_ == 0)
{
lean_ctor_set(v___x_72_, 0, v___x_74_);
v___x_76_ = v___x_72_;
goto v_reusejp_75_;
}
else
{
lean_object* v_reuseFailAlloc_77_; 
v_reuseFailAlloc_77_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_77_, 0, v___x_74_);
v___x_76_ = v_reuseFailAlloc_77_;
goto v_reusejp_75_;
}
v_reusejp_75_:
{
return v___x_76_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instSub___redArg___lam__0___boxed(lean_object* v___x_79_, lean_object* v_toSub_80_, lean_object* v_x_81_, lean_object* v_x_82_){
_start:
{
lean_object* v_res_83_; 
v_res_83_ = lp_mathlib_WithTop_LinearOrderedAddCommGroup_instSub___redArg___lam__0(v___x_79_, v_toSub_80_, v_x_81_, v_x_82_);
lean_dec(v___x_79_);
return v_res_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instSub___redArg(lean_object* v_inst_84_){
_start:
{
lean_object* v_toSub_85_; lean_object* v___x_86_; lean_object* v___f_87_; 
v_toSub_85_ = lean_ctor_get(v_inst_84_, 2);
lean_inc(v_toSub_85_);
lean_dec_ref(v_inst_84_);
v___x_86_ = lean_box(0);
v___f_87_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_LinearOrderedAddCommGroup_instSub___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_87_, 0, v___x_86_);
lean_closure_set(v___f_87_, 1, v_toSub_85_);
return v___f_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instSub(lean_object* v_G_88_, lean_object* v_inst_89_){
_start:
{
lean_object* v___x_90_; 
v___x_90_ = lp_mathlib_WithTop_LinearOrderedAddCommGroup_instSub___redArg(v_inst_89_);
return v___x_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instLinearOrderedAddCommGroupWithTopOfIsOrderedAddMonoid___redArg(lean_object* v_inst_91_, lean_object* v_inst_92_){
_start:
{
lean_object* v_toAddMonoid_93_; lean_object* v___x_94_; lean_object* v_toAddCommMonoid_95_; lean_object* v_toLinearOrder_96_; lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v_toZero_99_; lean_object* v_toAdd_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; 
v_toAddMonoid_93_ = lean_ctor_get(v_inst_91_, 0);
lean_inc_ref(v_toAddMonoid_93_);
v___x_94_ = lp_mathlib_WithTop_linearOrderedAddCommMonoidWithTop___redArg(v_toAddMonoid_93_, v_inst_92_);
v_toAddCommMonoid_95_ = lean_ctor_get(v___x_94_, 0);
lean_inc_ref(v_toAddCommMonoid_95_);
v_toLinearOrder_96_ = lean_ctor_get(v___x_94_, 1);
lean_inc_ref(v_toLinearOrder_96_);
lean_dec_ref(v___x_94_);
v___x_97_ = lean_box(0);
v___x_98_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_91_);
v_toZero_99_ = lean_ctor_get(v___x_98_, 0);
lean_inc(v_toZero_99_);
lean_dec_ref(v___x_98_);
v_toAdd_100_ = lean_ctor_get(v_toAddMonoid_93_, 1);
lean_inc(v_toAdd_100_);
v___x_101_ = lp_mathlib_WithTop_LinearOrderedAddCommGroup_instNeg___redArg(v_inst_91_);
v___x_102_ = lp_mathlib_WithTop_LinearOrderedAddCommGroup_instSub___redArg(v_inst_91_);
v___x_103_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_103_, 0, v_toZero_99_);
v___x_104_ = lp_mathlib_WithTop_add___redArg(v_toAdd_100_);
lean_inc_ref(v___x_104_);
lean_inc_ref(v___x_103_);
v___x_105_ = lean_alloc_closure((void*)(l_nsmulRec___boxed), 5, 3);
lean_closure_set(v___x_105_, 0, lean_box(0));
lean_closure_set(v___x_105_, 1, v___x_103_);
lean_closure_set(v___x_105_, 2, v___x_104_);
lean_inc_ref(v___x_101_);
v___x_106_ = lean_alloc_closure((void*)(lp_mathlib_zsmulRec___boxed), 7, 5);
lean_closure_set(v___x_106_, 0, lean_box(0));
lean_closure_set(v___x_106_, 1, v___x_103_);
lean_closure_set(v___x_106_, 2, v___x_104_);
lean_closure_set(v___x_106_, 3, v___x_101_);
lean_closure_set(v___x_106_, 4, v___x_105_);
v___x_107_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_107_, 0, v_toAddCommMonoid_95_);
lean_ctor_set(v___x_107_, 1, v_toLinearOrder_96_);
lean_ctor_set(v___x_107_, 2, v___x_97_);
lean_ctor_set(v___x_107_, 3, v___x_101_);
lean_ctor_set(v___x_107_, 4, v___x_102_);
lean_ctor_set(v___x_107_, 5, v___x_106_);
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_LinearOrderedAddCommGroup_instLinearOrderedAddCommGroupWithTopOfIsOrderedAddMonoid(lean_object* v_G_108_, lean_object* v_inst_109_, lean_object* v_inst_110_, lean_object* v_inst_111_){
_start:
{
lean_object* v___x_112_; 
v___x_112_ = lp_mathlib_WithTop_LinearOrderedAddCommGroup_instLinearOrderedAddCommGroupWithTopOfIsOrderedAddMonoid___redArg(v_inst_109_, v_inst_110_);
return v___x_112_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_CharZero_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_Canonical_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_WithTop(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Regular_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_AddGroupWithTop(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_CharZero_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_Canonical_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_WithTop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Regular_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Order_AddGroupWithTop(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_CharZero_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Monoid_Canonical_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Monoid_WithTop(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Regular_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_AddGroupWithTop(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_CharZero_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Monoid_Canonical_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Monoid_WithTop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Regular_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_AddGroupWithTop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Order_AddGroupWithTop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Order_AddGroupWithTop(builtin);
}
#ifdef __cplusplus
}
#endif
