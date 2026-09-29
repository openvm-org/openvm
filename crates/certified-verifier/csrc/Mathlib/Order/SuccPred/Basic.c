// Lean compiler output
// Module: Mathlib.Order.SuccPred.Basic
// Imports: public import Init public meta import Init public import Mathlib.Order.ConditionallyCompleteLattice.Basic public import Mathlib.Order.Cover public import Mathlib.Order.Iterate
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
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_toEmbedding___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instPredOrderOrderDualOfSuccOrder___redArg___lam__0(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_instPredOrderOrderDualOfSuccOrder___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instPredOrderOrderDualOfSuccOrder___redArg___closed__0;
static lean_once_cell_t lp_mathlib_instPredOrderOrderDualOfSuccOrder___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instPredOrderOrderDualOfSuccOrder___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_instPredOrderOrderDualOfSuccOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instPredOrderOrderDualOfSuccOrder(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instPredOrderOrderDualOfSuccOrder___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSuccOrderOrderDualOfPredOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSuccOrderOrderDualOfPredOrder(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSuccOrderOrderDualOfPredOrder___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SuccOrder_ofSuccLeIff___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SuccOrder_ofSuccLeIff___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SuccOrder_ofSuccLeIff(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SuccOrder_ofSuccLeIff___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PredOrder_ofPredLeIff___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PredOrder_ofPredLeIff___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PredOrder_ofPredLeIff(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PredOrder_ofPredLeIff___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SuccOrder_ofCore___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SuccOrder_ofCore___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SuccOrder_ofCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SuccOrder_ofCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PredOrder_ofCore___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PredOrder_ofCore___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PredOrder_ofCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PredOrder_ofCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Order_succ___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Order_succ(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Order_succ___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Order_pred___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Order_pred(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Order_pred___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instSuccOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instSuccOrder___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instSuccOrder___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instSuccOrder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instSuccOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instPredOrder_match__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instPredOrder_match__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instPredOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instPredOrder___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instPredOrder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instPredOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instPredOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instPredOrder___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instPredOrder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instPredOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instSuccOrder___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instSuccOrder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instSuccOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SuccOrder_ofOrderIso___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SuccOrder_ofOrderIso___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SuccOrder_ofOrderIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SuccOrder_ofOrderIso___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PredOrder_ofOrderIso___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PredOrder_ofOrderIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PredOrder_ofOrderIso___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_OrdConnected_succOrder___aux__6___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_OrdConnected_succOrder___aux__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_OrdConnected_succOrder___aux__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instPredOrderOrderDualOfSuccOrder___redArg___lam__0(lean_object* v___x_1_, lean_object* v___y_2_){
_start:
{
lean_object* v_toFun_3_; lean_object* v___x_4_; 
v_toFun_3_ = lean_ctor_get(v___x_1_, 0);
lean_inc(v_toFun_3_);
lean_dec_ref(v___x_1_);
v___x_4_ = lean_apply_1(v_toFun_3_, v___y_2_);
return v___x_4_;
}
}
static lean_object* _init_lp_mathlib_instPredOrderOrderDualOfSuccOrder___redArg___closed__0(void){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_5_;
}
}
static lean_object* _init_lp_mathlib_instPredOrderOrderDualOfSuccOrder___redArg___closed__1(void){
_start:
{
lean_object* v___x_6_; lean_object* v___f_7_; 
v___x_6_ = lean_obj_once(&lp_mathlib_instPredOrderOrderDualOfSuccOrder___redArg___closed__0, &lp_mathlib_instPredOrderOrderDualOfSuccOrder___redArg___closed__0_once, _init_lp_mathlib_instPredOrderOrderDualOfSuccOrder___redArg___closed__0);
v___f_7_ = lean_alloc_closure((void*)(lp_mathlib_instPredOrderOrderDualOfSuccOrder___redArg___lam__0), 2, 1);
lean_closure_set(v___f_7_, 0, v___x_6_);
return v___f_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instPredOrderOrderDualOfSuccOrder___redArg(lean_object* v_inst_8_){
_start:
{
lean_object* v___f_9_; lean_object* v___x_10_; lean_object* v___x_11_; 
v___f_9_ = lean_obj_once(&lp_mathlib_instPredOrderOrderDualOfSuccOrder___redArg___closed__1, &lp_mathlib_instPredOrderOrderDualOfSuccOrder___redArg___closed__1_once, _init_lp_mathlib_instPredOrderOrderDualOfSuccOrder___redArg___closed__1);
v___x_10_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_10_, 0, lean_box(0));
lean_closure_set(v___x_10_, 1, lean_box(0));
lean_closure_set(v___x_10_, 2, lean_box(0));
lean_closure_set(v___x_10_, 3, v_inst_8_);
lean_closure_set(v___x_10_, 4, v___f_9_);
v___x_11_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_11_, 0, lean_box(0));
lean_closure_set(v___x_11_, 1, lean_box(0));
lean_closure_set(v___x_11_, 2, lean_box(0));
lean_closure_set(v___x_11_, 3, v___f_9_);
lean_closure_set(v___x_11_, 4, v___x_10_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instPredOrderOrderDualOfSuccOrder(lean_object* v_00_u03b1_12_, lean_object* v_inst_13_, lean_object* v_inst_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lp_mathlib_instPredOrderOrderDualOfSuccOrder___redArg(v_inst_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instPredOrderOrderDualOfSuccOrder___boxed(lean_object* v_00_u03b1_16_, lean_object* v_inst_17_, lean_object* v_inst_18_){
_start:
{
lean_object* v_res_19_; 
v_res_19_ = lp_mathlib_instPredOrderOrderDualOfSuccOrder(v_00_u03b1_16_, v_inst_17_, v_inst_18_);
lean_dec_ref(v_inst_17_);
return v_res_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSuccOrderOrderDualOfPredOrder___redArg(lean_object* v_inst_20_){
_start:
{
lean_object* v___f_21_; lean_object* v___x_22_; lean_object* v___x_23_; 
v___f_21_ = lean_obj_once(&lp_mathlib_instPredOrderOrderDualOfSuccOrder___redArg___closed__1, &lp_mathlib_instPredOrderOrderDualOfSuccOrder___redArg___closed__1_once, _init_lp_mathlib_instPredOrderOrderDualOfSuccOrder___redArg___closed__1);
v___x_22_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_22_, 0, lean_box(0));
lean_closure_set(v___x_22_, 1, lean_box(0));
lean_closure_set(v___x_22_, 2, lean_box(0));
lean_closure_set(v___x_22_, 3, v_inst_20_);
lean_closure_set(v___x_22_, 4, v___f_21_);
v___x_23_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_23_, 0, lean_box(0));
lean_closure_set(v___x_23_, 1, lean_box(0));
lean_closure_set(v___x_23_, 2, lean_box(0));
lean_closure_set(v___x_23_, 3, v___f_21_);
lean_closure_set(v___x_23_, 4, v___x_22_);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSuccOrderOrderDualOfPredOrder(lean_object* v_00_u03b1_24_, lean_object* v_inst_25_, lean_object* v_inst_26_){
_start:
{
lean_object* v___x_27_; 
v___x_27_ = lp_mathlib_instSuccOrderOrderDualOfPredOrder___redArg(v_inst_26_);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSuccOrderOrderDualOfPredOrder___boxed(lean_object* v_00_u03b1_28_, lean_object* v_inst_29_, lean_object* v_inst_30_){
_start:
{
lean_object* v_res_31_; 
v_res_31_ = lp_mathlib_instSuccOrderOrderDualOfPredOrder(v_00_u03b1_28_, v_inst_29_, v_inst_30_);
lean_dec_ref(v_inst_29_);
return v_res_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SuccOrder_ofSuccLeIff___redArg(lean_object* v_succ_32_){
_start:
{
lean_inc(v_succ_32_);
return v_succ_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SuccOrder_ofSuccLeIff___redArg___boxed(lean_object* v_succ_33_){
_start:
{
lean_object* v_res_34_; 
v_res_34_ = lp_mathlib_SuccOrder_ofSuccLeIff___redArg(v_succ_33_);
lean_dec(v_succ_33_);
return v_res_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SuccOrder_ofSuccLeIff(lean_object* v_00_u03b1_35_, lean_object* v_inst_36_, lean_object* v_succ_37_, lean_object* v_hsucc__le__iff_38_){
_start:
{
lean_inc(v_succ_37_);
return v_succ_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SuccOrder_ofSuccLeIff___boxed(lean_object* v_00_u03b1_39_, lean_object* v_inst_40_, lean_object* v_succ_41_, lean_object* v_hsucc__le__iff_42_){
_start:
{
lean_object* v_res_43_; 
v_res_43_ = lp_mathlib_SuccOrder_ofSuccLeIff(v_00_u03b1_39_, v_inst_40_, v_succ_41_, v_hsucc__le__iff_42_);
lean_dec(v_succ_41_);
lean_dec_ref(v_inst_40_);
return v_res_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PredOrder_ofPredLeIff___redArg(lean_object* v_succ_44_){
_start:
{
lean_inc(v_succ_44_);
return v_succ_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PredOrder_ofPredLeIff___redArg___boxed(lean_object* v_succ_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_mathlib_PredOrder_ofPredLeIff___redArg(v_succ_45_);
lean_dec(v_succ_45_);
return v_res_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PredOrder_ofPredLeIff(lean_object* v_00_u03b1_47_, lean_object* v_inst_48_, lean_object* v_succ_49_, lean_object* v_hsucc__le__iff_50_){
_start:
{
lean_inc(v_succ_49_);
return v_succ_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PredOrder_ofPredLeIff___boxed(lean_object* v_00_u03b1_51_, lean_object* v_inst_52_, lean_object* v_succ_53_, lean_object* v_hsucc__le__iff_54_){
_start:
{
lean_object* v_res_55_; 
v_res_55_ = lp_mathlib_PredOrder_ofPredLeIff(v_00_u03b1_51_, v_inst_52_, v_succ_53_, v_hsucc__le__iff_54_);
lean_dec(v_succ_53_);
lean_dec_ref(v_inst_52_);
return v_res_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SuccOrder_ofCore___redArg(lean_object* v_succ_56_){
_start:
{
lean_inc(v_succ_56_);
return v_succ_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SuccOrder_ofCore___redArg___boxed(lean_object* v_succ_57_){
_start:
{
lean_object* v_res_58_; 
v_res_58_ = lp_mathlib_SuccOrder_ofCore___redArg(v_succ_57_);
lean_dec(v_succ_57_);
return v_res_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SuccOrder_ofCore(lean_object* v_00_u03b1_59_, lean_object* v_inst_60_, lean_object* v_succ_61_, lean_object* v_hn_62_, lean_object* v_hm_63_){
_start:
{
lean_inc(v_succ_61_);
return v_succ_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SuccOrder_ofCore___boxed(lean_object* v_00_u03b1_64_, lean_object* v_inst_65_, lean_object* v_succ_66_, lean_object* v_hn_67_, lean_object* v_hm_68_){
_start:
{
lean_object* v_res_69_; 
v_res_69_ = lp_mathlib_SuccOrder_ofCore(v_00_u03b1_64_, v_inst_65_, v_succ_66_, v_hn_67_, v_hm_68_);
lean_dec(v_succ_66_);
lean_dec_ref(v_inst_65_);
return v_res_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PredOrder_ofCore___redArg(lean_object* v_succ_70_){
_start:
{
lean_inc(v_succ_70_);
return v_succ_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PredOrder_ofCore___redArg___boxed(lean_object* v_succ_71_){
_start:
{
lean_object* v_res_72_; 
v_res_72_ = lp_mathlib_PredOrder_ofCore___redArg(v_succ_71_);
lean_dec(v_succ_71_);
return v_res_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PredOrder_ofCore(lean_object* v_00_u03b1_73_, lean_object* v_inst_74_, lean_object* v_succ_75_, lean_object* v_hn_76_, lean_object* v_hm_77_){
_start:
{
lean_inc(v_succ_75_);
return v_succ_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PredOrder_ofCore___boxed(lean_object* v_00_u03b1_78_, lean_object* v_inst_79_, lean_object* v_succ_80_, lean_object* v_hn_81_, lean_object* v_hm_82_){
_start:
{
lean_object* v_res_83_; 
v_res_83_ = lp_mathlib_PredOrder_ofCore(v_00_u03b1_78_, v_inst_79_, v_succ_80_, v_hn_81_, v_hm_82_);
lean_dec(v_succ_80_);
lean_dec_ref(v_inst_79_);
return v_res_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Order_succ___redArg(lean_object* v_inst_84_, lean_object* v_a_85_){
_start:
{
lean_object* v___x_86_; 
v___x_86_ = lean_apply_1(v_inst_84_, v_a_85_);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Order_succ(lean_object* v_00_u03b1_87_, lean_object* v_inst_88_, lean_object* v_inst_89_, lean_object* v_a_90_){
_start:
{
lean_object* v___x_91_; 
v___x_91_ = lean_apply_1(v_inst_89_, v_a_90_);
return v___x_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Order_succ___boxed(lean_object* v_00_u03b1_92_, lean_object* v_inst_93_, lean_object* v_inst_94_, lean_object* v_a_95_){
_start:
{
lean_object* v_res_96_; 
v_res_96_ = lp_mathlib_Order_succ(v_00_u03b1_92_, v_inst_93_, v_inst_94_, v_a_95_);
lean_dec_ref(v_inst_93_);
return v_res_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Order_pred___redArg(lean_object* v_inst_97_, lean_object* v_a_98_){
_start:
{
lean_object* v___x_99_; 
v___x_99_ = lean_apply_1(v_inst_97_, v_a_98_);
return v___x_99_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Order_pred(lean_object* v_00_u03b1_100_, lean_object* v_inst_101_, lean_object* v_inst_102_, lean_object* v_a_103_){
_start:
{
lean_object* v___x_104_; 
v___x_104_ = lean_apply_1(v_inst_102_, v_a_103_);
return v___x_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Order_pred___boxed(lean_object* v_00_u03b1_105_, lean_object* v_inst_106_, lean_object* v_inst_107_, lean_object* v_a_108_){
_start:
{
lean_object* v_res_109_; 
v_res_109_ = lp_mathlib_Order_pred(v_00_u03b1_105_, v_inst_106_, v_inst_107_, v_a_108_);
lean_dec_ref(v_inst_106_);
return v_res_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instSuccOrder___redArg___lam__0(lean_object* v___x_110_, lean_object* v_inst_111_, lean_object* v_inst_112_, lean_object* v_x_113_){
_start:
{
if (lean_obj_tag(v_x_113_) == 0)
{
lean_dec(v_inst_112_);
lean_dec_ref(v_inst_111_);
lean_inc(v___x_110_);
return v___x_110_;
}
else
{
lean_object* v_val_114_; lean_object* v___x_116_; uint8_t v_isShared_117_; uint8_t v_isSharedCheck_124_; 
v_val_114_ = lean_ctor_get(v_x_113_, 0);
v_isSharedCheck_124_ = !lean_is_exclusive(v_x_113_);
if (v_isSharedCheck_124_ == 0)
{
v___x_116_ = v_x_113_;
v_isShared_117_ = v_isSharedCheck_124_;
goto v_resetjp_115_;
}
else
{
lean_inc(v_val_114_);
lean_dec(v_x_113_);
v___x_116_ = lean_box(0);
v_isShared_117_ = v_isSharedCheck_124_;
goto v_resetjp_115_;
}
v_resetjp_115_:
{
lean_object* v___x_118_; uint8_t v___x_119_; 
lean_inc(v_val_114_);
v___x_118_ = lean_apply_1(v_inst_111_, v_val_114_);
v___x_119_ = lean_unbox(v___x_118_);
if (v___x_119_ == 0)
{
lean_object* v___x_120_; lean_object* v___x_122_; 
v___x_120_ = lean_apply_1(v_inst_112_, v_val_114_);
if (v_isShared_117_ == 0)
{
lean_ctor_set(v___x_116_, 0, v___x_120_);
v___x_122_ = v___x_116_;
goto v_reusejp_121_;
}
else
{
lean_object* v_reuseFailAlloc_123_; 
v_reuseFailAlloc_123_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_123_, 0, v___x_120_);
v___x_122_ = v_reuseFailAlloc_123_;
goto v_reusejp_121_;
}
v_reusejp_121_:
{
return v___x_122_;
}
}
else
{
lean_del_object(v___x_116_);
lean_dec(v_val_114_);
lean_dec(v_inst_112_);
lean_inc(v___x_110_);
return v___x_110_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instSuccOrder___redArg___lam__0___boxed(lean_object* v___x_125_, lean_object* v_inst_126_, lean_object* v_inst_127_, lean_object* v_x_128_){
_start:
{
lean_object* v_res_129_; 
v_res_129_ = lp_mathlib_WithTop_instSuccOrder___redArg___lam__0(v___x_125_, v_inst_126_, v_inst_127_, v_x_128_);
lean_dec(v___x_125_);
return v_res_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instSuccOrder___redArg(lean_object* v_inst_130_, lean_object* v_inst_131_){
_start:
{
lean_object* v___x_132_; lean_object* v___f_133_; 
v___x_132_ = lean_box(0);
v___f_133_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_instSuccOrder___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_133_, 0, v___x_132_);
lean_closure_set(v___f_133_, 1, v_inst_131_);
lean_closure_set(v___f_133_, 2, v_inst_130_);
return v___f_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instSuccOrder(lean_object* v_00_u03b1_134_, lean_object* v_inst_135_, lean_object* v_inst_136_, lean_object* v_inst_137_){
_start:
{
lean_object* v___x_138_; 
v___x_138_ = lp_mathlib_WithTop_instSuccOrder___redArg(v_inst_136_, v_inst_137_);
return v___x_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instSuccOrder___boxed(lean_object* v_00_u03b1_139_, lean_object* v_inst_140_, lean_object* v_inst_141_, lean_object* v_inst_142_){
_start:
{
lean_object* v_res_143_; 
v_res_143_ = lp_mathlib_WithTop_instSuccOrder(v_00_u03b1_139_, v_inst_140_, v_inst_141_, v_inst_142_);
lean_dec_ref(v_inst_140_);
return v_res_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instPredOrder_match__1___redArg(lean_object* v_x_144_, lean_object* v_h__1_145_, lean_object* v_h__2_146_){
_start:
{
if (lean_obj_tag(v_x_144_) == 0)
{
lean_object* v___x_147_; lean_object* v___x_148_; 
lean_dec(v_h__2_146_);
v___x_147_ = lean_box(0);
v___x_148_ = lean_apply_1(v_h__1_145_, v___x_147_);
return v___x_148_;
}
else
{
lean_object* v_val_149_; lean_object* v___x_150_; 
lean_dec(v_h__1_145_);
v_val_149_ = lean_ctor_get(v_x_144_, 0);
lean_inc(v_val_149_);
lean_dec_ref_known(v_x_144_, 1);
v___x_150_ = lean_apply_1(v_h__2_146_, v_val_149_);
return v___x_150_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instPredOrder_match__1(lean_object* v_00_u03b1_151_, lean_object* v_motive_152_, lean_object* v_x_153_, lean_object* v_h__1_154_, lean_object* v_h__2_155_){
_start:
{
lean_object* v___x_156_; 
v___x_156_ = lp_mathlib_WithBot_instPredOrder_match__1___redArg(v_x_153_, v_h__1_154_, v_h__2_155_);
return v___x_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instPredOrder___redArg___lam__0(lean_object* v_inst_157_, lean_object* v_inst_158_, lean_object* v_x_159_){
_start:
{
if (lean_obj_tag(v_x_159_) == 0)
{
lean_dec(v_inst_158_);
lean_dec_ref(v_inst_157_);
return v_x_159_;
}
else
{
lean_object* v_val_160_; lean_object* v___x_162_; uint8_t v_isShared_163_; uint8_t v_isSharedCheck_171_; 
v_val_160_ = lean_ctor_get(v_x_159_, 0);
v_isSharedCheck_171_ = !lean_is_exclusive(v_x_159_);
if (v_isSharedCheck_171_ == 0)
{
v___x_162_ = v_x_159_;
v_isShared_163_ = v_isSharedCheck_171_;
goto v_resetjp_161_;
}
else
{
lean_inc(v_val_160_);
lean_dec(v_x_159_);
v___x_162_ = lean_box(0);
v_isShared_163_ = v_isSharedCheck_171_;
goto v_resetjp_161_;
}
v_resetjp_161_:
{
lean_object* v___x_164_; uint8_t v___x_165_; 
lean_inc(v_val_160_);
v___x_164_ = lean_apply_1(v_inst_157_, v_val_160_);
v___x_165_ = lean_unbox(v___x_164_);
if (v___x_165_ == 0)
{
lean_object* v___x_166_; lean_object* v___x_168_; 
v___x_166_ = lean_apply_1(v_inst_158_, v_val_160_);
if (v_isShared_163_ == 0)
{
lean_ctor_set(v___x_162_, 0, v___x_166_);
v___x_168_ = v___x_162_;
goto v_reusejp_167_;
}
else
{
lean_object* v_reuseFailAlloc_169_; 
v_reuseFailAlloc_169_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_169_, 0, v___x_166_);
v___x_168_ = v_reuseFailAlloc_169_;
goto v_reusejp_167_;
}
v_reusejp_167_:
{
return v___x_168_;
}
}
else
{
lean_object* v___x_170_; 
lean_del_object(v___x_162_);
lean_dec(v_val_160_);
lean_dec(v_inst_158_);
v___x_170_ = lean_box(0);
return v___x_170_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instPredOrder___redArg(lean_object* v_inst_172_, lean_object* v_inst_173_){
_start:
{
lean_object* v___f_174_; 
v___f_174_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_instPredOrder___redArg___lam__0), 3, 2);
lean_closure_set(v___f_174_, 0, v_inst_173_);
lean_closure_set(v___f_174_, 1, v_inst_172_);
return v___f_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instPredOrder(lean_object* v_00_u03b1_175_, lean_object* v_inst_176_, lean_object* v_inst_177_, lean_object* v_inst_178_){
_start:
{
lean_object* v___f_179_; 
v___f_179_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_instPredOrder___redArg___lam__0), 3, 2);
lean_closure_set(v___f_179_, 0, v_inst_178_);
lean_closure_set(v___f_179_, 1, v_inst_177_);
return v___f_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instPredOrder___boxed(lean_object* v_00_u03b1_180_, lean_object* v_inst_181_, lean_object* v_inst_182_, lean_object* v_inst_183_){
_start:
{
lean_object* v_res_184_; 
v_res_184_ = lp_mathlib_WithBot_instPredOrder(v_00_u03b1_180_, v_inst_181_, v_inst_182_, v_inst_183_);
lean_dec_ref(v_inst_181_);
return v_res_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instPredOrder___redArg___lam__0(lean_object* v_inst_185_, lean_object* v_inst_186_, lean_object* v_x_187_){
_start:
{
if (lean_obj_tag(v_x_187_) == 0)
{
lean_object* v___x_188_; 
lean_dec(v_inst_186_);
v___x_188_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_188_, 0, v_inst_185_);
return v___x_188_;
}
else
{
lean_object* v_val_189_; lean_object* v___x_191_; uint8_t v_isShared_192_; uint8_t v_isSharedCheck_197_; 
lean_dec(v_inst_185_);
v_val_189_ = lean_ctor_get(v_x_187_, 0);
v_isSharedCheck_197_ = !lean_is_exclusive(v_x_187_);
if (v_isSharedCheck_197_ == 0)
{
v___x_191_ = v_x_187_;
v_isShared_192_ = v_isSharedCheck_197_;
goto v_resetjp_190_;
}
else
{
lean_inc(v_val_189_);
lean_dec(v_x_187_);
v___x_191_ = lean_box(0);
v_isShared_192_ = v_isSharedCheck_197_;
goto v_resetjp_190_;
}
v_resetjp_190_:
{
lean_object* v___x_193_; lean_object* v___x_195_; 
v___x_193_ = lean_apply_1(v_inst_186_, v_val_189_);
if (v_isShared_192_ == 0)
{
lean_ctor_set(v___x_191_, 0, v___x_193_);
v___x_195_ = v___x_191_;
goto v_reusejp_194_;
}
else
{
lean_object* v_reuseFailAlloc_196_; 
v_reuseFailAlloc_196_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_196_, 0, v___x_193_);
v___x_195_ = v_reuseFailAlloc_196_;
goto v_reusejp_194_;
}
v_reusejp_194_:
{
return v___x_195_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instPredOrder___redArg(lean_object* v_inst_198_, lean_object* v_inst_199_){
_start:
{
lean_object* v___f_200_; 
v___f_200_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_instPredOrder___redArg___lam__0), 3, 2);
lean_closure_set(v___f_200_, 0, v_inst_198_);
lean_closure_set(v___f_200_, 1, v_inst_199_);
return v___f_200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instPredOrder(lean_object* v_00_u03b1_201_, lean_object* v_inst_202_, lean_object* v_inst_203_, lean_object* v_inst_204_){
_start:
{
lean_object* v___f_205_; 
v___f_205_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_instPredOrder___redArg___lam__0), 3, 2);
lean_closure_set(v___f_205_, 0, v_inst_203_);
lean_closure_set(v___f_205_, 1, v_inst_204_);
return v___f_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instPredOrder___boxed(lean_object* v_00_u03b1_206_, lean_object* v_inst_207_, lean_object* v_inst_208_, lean_object* v_inst_209_){
_start:
{
lean_object* v_res_210_; 
v_res_210_ = lp_mathlib_WithTop_instPredOrder(v_00_u03b1_206_, v_inst_207_, v_inst_208_, v_inst_209_);
lean_dec_ref(v_inst_207_);
return v_res_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instSuccOrder___redArg(lean_object* v_inst_211_, lean_object* v_inst_212_){
_start:
{
lean_object* v___f_213_; 
v___f_213_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_instPredOrder___redArg___lam__0), 3, 2);
lean_closure_set(v___f_213_, 0, v_inst_211_);
lean_closure_set(v___f_213_, 1, v_inst_212_);
return v___f_213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instSuccOrder(lean_object* v_00_u03b1_214_, lean_object* v_inst_215_, lean_object* v_inst_216_, lean_object* v_inst_217_){
_start:
{
lean_object* v___f_218_; 
v___f_218_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_instPredOrder___redArg___lam__0), 3, 2);
lean_closure_set(v___f_218_, 0, v_inst_216_);
lean_closure_set(v___f_218_, 1, v_inst_217_);
return v___f_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instSuccOrder___boxed(lean_object* v_00_u03b1_219_, lean_object* v_inst_220_, lean_object* v_inst_221_, lean_object* v_inst_222_){
_start:
{
lean_object* v_res_223_; 
v_res_223_ = lp_mathlib_WithBot_instSuccOrder(v_00_u03b1_219_, v_inst_220_, v_inst_221_, v_inst_222_);
lean_dec_ref(v_inst_220_);
return v_res_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SuccOrder_ofOrderIso___redArg___lam__0(lean_object* v_f_224_, lean_object* v_inst_225_, lean_object* v_y_226_){
_start:
{
lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; 
lean_inc_ref(v_f_224_);
v___x_227_ = lp_mathlib_Equiv_symm___redArg(v_f_224_);
v___x_228_ = lp_mathlib_Equiv_toEmbedding___redArg___lam__0(v___x_227_, v_y_226_);
v___x_229_ = lean_apply_1(v_inst_225_, v___x_228_);
v___x_230_ = lp_mathlib_Equiv_toEmbedding___redArg___lam__0(v_f_224_, v___x_229_);
return v___x_230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SuccOrder_ofOrderIso___redArg(lean_object* v_inst_231_, lean_object* v_f_232_){
_start:
{
lean_object* v___f_233_; 
v___f_233_ = lean_alloc_closure((void*)(lp_mathlib_SuccOrder_ofOrderIso___redArg___lam__0), 3, 2);
lean_closure_set(v___f_233_, 0, v_f_232_);
lean_closure_set(v___f_233_, 1, v_inst_231_);
return v___f_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SuccOrder_ofOrderIso(lean_object* v_X_234_, lean_object* v_Y_235_, lean_object* v_inst_236_, lean_object* v_inst_237_, lean_object* v_inst_238_, lean_object* v_f_239_){
_start:
{
lean_object* v___f_240_; 
v___f_240_ = lean_alloc_closure((void*)(lp_mathlib_SuccOrder_ofOrderIso___redArg___lam__0), 3, 2);
lean_closure_set(v___f_240_, 0, v_f_239_);
lean_closure_set(v___f_240_, 1, v_inst_238_);
return v___f_240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SuccOrder_ofOrderIso___boxed(lean_object* v_X_241_, lean_object* v_Y_242_, lean_object* v_inst_243_, lean_object* v_inst_244_, lean_object* v_inst_245_, lean_object* v_f_246_){
_start:
{
lean_object* v_res_247_; 
v_res_247_ = lp_mathlib_SuccOrder_ofOrderIso(v_X_241_, v_Y_242_, v_inst_243_, v_inst_244_, v_inst_245_, v_f_246_);
lean_dec_ref(v_inst_244_);
lean_dec_ref(v_inst_243_);
return v_res_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PredOrder_ofOrderIso___redArg(lean_object* v_inst_248_, lean_object* v_f_249_){
_start:
{
lean_object* v___f_250_; 
v___f_250_ = lean_alloc_closure((void*)(lp_mathlib_SuccOrder_ofOrderIso___redArg___lam__0), 3, 2);
lean_closure_set(v___f_250_, 0, v_f_249_);
lean_closure_set(v___f_250_, 1, v_inst_248_);
return v___f_250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PredOrder_ofOrderIso(lean_object* v_X_251_, lean_object* v_Y_252_, lean_object* v_inst_253_, lean_object* v_inst_254_, lean_object* v_inst_255_, lean_object* v_f_256_){
_start:
{
lean_object* v___f_257_; 
v___f_257_ = lean_alloc_closure((void*)(lp_mathlib_SuccOrder_ofOrderIso___redArg___lam__0), 3, 2);
lean_closure_set(v___f_257_, 0, v_f_256_);
lean_closure_set(v___f_257_, 1, v_inst_255_);
return v___f_257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PredOrder_ofOrderIso___boxed(lean_object* v_X_258_, lean_object* v_Y_259_, lean_object* v_inst_260_, lean_object* v_inst_261_, lean_object* v_inst_262_, lean_object* v_f_263_){
_start:
{
lean_object* v_res_264_; 
v_res_264_ = lp_mathlib_PredOrder_ofOrderIso(v_X_258_, v_Y_259_, v_inst_260_, v_inst_261_, v_inst_262_, v_f_263_);
lean_dec_ref(v_inst_261_);
lean_dec_ref(v_inst_260_);
return v_res_264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_OrdConnected_succOrder___aux__6___redArg(lean_object* v_this_265_, lean_object* v_a_266_){
_start:
{
lean_object* v___x_267_; lean_object* v_toFun_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; 
v___x_267_ = lean_obj_once(&lp_mathlib_instPredOrderOrderDualOfSuccOrder___redArg___closed__0, &lp_mathlib_instPredOrderOrderDualOfSuccOrder___redArg___closed__0_once, _init_lp_mathlib_instPredOrderOrderDualOfSuccOrder___redArg___closed__0);
v_toFun_268_ = lean_ctor_get(v___x_267_, 0);
lean_inc_n(v_toFun_268_, 2);
v___x_269_ = lean_apply_1(v_toFun_268_, v_a_266_);
v___x_270_ = lean_apply_1(v_this_265_, v___x_269_);
v___x_271_ = lean_apply_1(v_toFun_268_, v___x_270_);
return v___x_271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_OrdConnected_succOrder___aux__6(lean_object* v_00_u03b1_272_, lean_object* v_inst_273_, lean_object* v_s_274_, lean_object* v_this_275_, lean_object* v_a_276_){
_start:
{
lean_object* v___x_277_; lean_object* v_toFun_278_; lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; 
v___x_277_ = lean_obj_once(&lp_mathlib_instPredOrderOrderDualOfSuccOrder___redArg___closed__0, &lp_mathlib_instPredOrderOrderDualOfSuccOrder___redArg___closed__0_once, _init_lp_mathlib_instPredOrderOrderDualOfSuccOrder___redArg___closed__0);
v_toFun_278_ = lean_ctor_get(v___x_277_, 0);
lean_inc_n(v_toFun_278_, 2);
v___x_279_ = lean_apply_1(v_toFun_278_, v_a_276_);
v___x_280_ = lean_apply_1(v_this_275_, v___x_279_);
v___x_281_ = lean_apply_1(v_toFun_278_, v___x_280_);
return v___x_281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_OrdConnected_succOrder___aux__6___boxed(lean_object* v_00_u03b1_282_, lean_object* v_inst_283_, lean_object* v_s_284_, lean_object* v_this_285_, lean_object* v_a_286_){
_start:
{
lean_object* v_res_287_; 
v_res_287_ = lp_mathlib_Set_OrdConnected_succOrder___aux__6(v_00_u03b1_282_, v_inst_283_, v_s_284_, v_this_285_, v_a_286_);
lean_dec_ref(v_inst_283_);
return v_res_287_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Cover(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Iterate(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_SuccPred_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Cover(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Iterate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_SuccPred_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Cover(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Iterate(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_SuccPred_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Cover(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Iterate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_SuccPred_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_SuccPred_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_SuccPred_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
