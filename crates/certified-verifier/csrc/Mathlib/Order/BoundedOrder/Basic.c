// Lean compiler output
// Module: Mathlib.Order.BoundedOrder.Basic
// Imports: public import Init public meta import Init public import Mathlib.Order.Max public import Mathlib.Order.ULift public import Mathlib.Tactic.ByCases public import Mathlib.Tactic.Finiteness.Attr
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
LEAN_EXPORT lean_object* lp_mathlib_IsTop_rec___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsTop_rec(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsBot_rec___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsBot_rec(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instTopOfBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instTopOfBot___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instTopOfBot(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instTopOfBot___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instBotOfTop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instBotOfTop___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instBotOfTop(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instBotOfTop___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instOrderTopOfOrderBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instOrderTopOfOrderBot___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instOrderTopOfOrderBot(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instOrderTopOfOrderBot___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instOrderBotOfOrderTop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instOrderBotOfOrderTop___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instOrderBotOfOrderTop(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instOrderBotOfOrderTop___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instBoundedOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instBoundedOrder(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBotForall___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBotForall___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBotForall(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instTopForall___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instTopForall(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderBot___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderBot(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderTop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderTop(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderTop___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBoundedOrder___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBoundedOrder___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBoundedOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBoundedOrder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBoundedOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrder_ofUnique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrder_ofUnique(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrder_ofUnique___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderTop_lift___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderTop_lift___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderTop_lift(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderTop_lift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderBot_lift___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderBot_lift___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderBot_lift(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderBot_lift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrder_lift___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrder_lift(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrder_lift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_orderBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_orderBot___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_orderBot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_orderBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_orderTop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_orderTop___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_orderTop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_orderTop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_boundedOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_boundedOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instTop___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instTop(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instBot___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instBot(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instOrderTop___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instOrderTop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instOrderBot___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instOrderBot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instBoundedOrder___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instBoundedOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instTop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instTop___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instTop(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instTop___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instBot___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instBot(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instBot___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instOrderBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instOrderBot___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instOrderBot(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instOrderBot___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instOrderTop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instOrderTop___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instOrderTop(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instOrderTop___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instBoundedOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instBoundedOrder(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Bool_instBoundedOrder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Bool_instBoundedOrder___closed__0 = (const lean_object*)&lp_mathlib_Bool_instBoundedOrder___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Bool_instBoundedOrder = (const lean_object*)&lp_mathlib_Bool_instBoundedOrder___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_IsTop_rec___redArg(lean_object* v_top_1_, lean_object* v_x_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_apply_1(v_top_1_, v_x_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsTop_rec(lean_object* v_00_u03b1_4_, lean_object* v_inst_5_, lean_object* v_motive_6_, lean_object* v_top_7_, lean_object* v_x_8_, lean_object* v_hx_9_){
_start:
{
lean_object* v___x_10_; 
v___x_10_ = lean_apply_1(v_top_7_, v_x_8_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsBot_rec___redArg(lean_object* v_top_11_, lean_object* v_x_12_){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = lean_apply_1(v_top_11_, v_x_12_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsBot_rec(lean_object* v_00_u03b1_14_, lean_object* v_inst_15_, lean_object* v_motive_16_, lean_object* v_top_17_, lean_object* v_x_18_, lean_object* v_hx_19_){
_start:
{
lean_object* v___x_20_; 
v___x_20_ = lean_apply_1(v_top_17_, v_x_18_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instTopOfBot___redArg(lean_object* v_h_21_){
_start:
{
lean_inc(v_h_21_);
return v_h_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instTopOfBot___redArg___boxed(lean_object* v_h_22_){
_start:
{
lean_object* v_res_23_; 
v_res_23_ = lp_mathlib_OrderDual_instTopOfBot___redArg(v_h_22_);
lean_dec(v_h_22_);
return v_res_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instTopOfBot(lean_object* v_00_u03b1_24_, lean_object* v_h_25_){
_start:
{
lean_inc(v_h_25_);
return v_h_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instTopOfBot___boxed(lean_object* v_00_u03b1_26_, lean_object* v_h_27_){
_start:
{
lean_object* v_res_28_; 
v_res_28_ = lp_mathlib_OrderDual_instTopOfBot(v_00_u03b1_26_, v_h_27_);
lean_dec(v_h_27_);
return v_res_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instBotOfTop___redArg(lean_object* v_h_29_){
_start:
{
lean_inc(v_h_29_);
return v_h_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instBotOfTop___redArg___boxed(lean_object* v_h_30_){
_start:
{
lean_object* v_res_31_; 
v_res_31_ = lp_mathlib_OrderDual_instBotOfTop___redArg(v_h_30_);
lean_dec(v_h_30_);
return v_res_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instBotOfTop(lean_object* v_00_u03b1_32_, lean_object* v_h_33_){
_start:
{
lean_inc(v_h_33_);
return v_h_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instBotOfTop___boxed(lean_object* v_00_u03b1_34_, lean_object* v_h_35_){
_start:
{
lean_object* v_res_36_; 
v_res_36_ = lp_mathlib_OrderDual_instBotOfTop(v_00_u03b1_34_, v_h_35_);
lean_dec(v_h_35_);
return v_res_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instOrderTopOfOrderBot___redArg(lean_object* v_h_37_){
_start:
{
lean_inc(v_h_37_);
return v_h_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instOrderTopOfOrderBot___redArg___boxed(lean_object* v_h_38_){
_start:
{
lean_object* v_res_39_; 
v_res_39_ = lp_mathlib_OrderDual_instOrderTopOfOrderBot___redArg(v_h_38_);
lean_dec(v_h_38_);
return v_res_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instOrderTopOfOrderBot(lean_object* v_00_u03b1_40_, lean_object* v_inst_41_, lean_object* v_h_42_){
_start:
{
lean_inc(v_h_42_);
return v_h_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instOrderTopOfOrderBot___boxed(lean_object* v_00_u03b1_43_, lean_object* v_inst_44_, lean_object* v_h_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_mathlib_OrderDual_instOrderTopOfOrderBot(v_00_u03b1_43_, v_inst_44_, v_h_45_);
lean_dec(v_h_45_);
return v_res_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instOrderBotOfOrderTop___redArg(lean_object* v_h_47_){
_start:
{
lean_inc(v_h_47_);
return v_h_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instOrderBotOfOrderTop___redArg___boxed(lean_object* v_h_48_){
_start:
{
lean_object* v_res_49_; 
v_res_49_ = lp_mathlib_OrderDual_instOrderBotOfOrderTop___redArg(v_h_48_);
lean_dec(v_h_48_);
return v_res_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instOrderBotOfOrderTop(lean_object* v_00_u03b1_50_, lean_object* v_inst_51_, lean_object* v_h_52_){
_start:
{
lean_inc(v_h_52_);
return v_h_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instOrderBotOfOrderTop___boxed(lean_object* v_00_u03b1_53_, lean_object* v_inst_54_, lean_object* v_h_55_){
_start:
{
lean_object* v_res_56_; 
v_res_56_ = lp_mathlib_OrderDual_instOrderBotOfOrderTop(v_00_u03b1_53_, v_inst_54_, v_h_55_);
lean_dec(v_h_55_);
return v_res_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instBoundedOrder___redArg(lean_object* v_inst_57_){
_start:
{
lean_object* v_toOrderTop_58_; lean_object* v_toOrderBot_59_; lean_object* v___x_61_; uint8_t v_isShared_62_; uint8_t v_isSharedCheck_66_; 
v_toOrderTop_58_ = lean_ctor_get(v_inst_57_, 0);
v_toOrderBot_59_ = lean_ctor_get(v_inst_57_, 1);
v_isSharedCheck_66_ = !lean_is_exclusive(v_inst_57_);
if (v_isSharedCheck_66_ == 0)
{
v___x_61_ = v_inst_57_;
v_isShared_62_ = v_isSharedCheck_66_;
goto v_resetjp_60_;
}
else
{
lean_inc(v_toOrderBot_59_);
lean_inc(v_toOrderTop_58_);
lean_dec(v_inst_57_);
v___x_61_ = lean_box(0);
v_isShared_62_ = v_isSharedCheck_66_;
goto v_resetjp_60_;
}
v_resetjp_60_:
{
lean_object* v___x_64_; 
if (v_isShared_62_ == 0)
{
lean_ctor_set(v___x_61_, 1, v_toOrderTop_58_);
lean_ctor_set(v___x_61_, 0, v_toOrderBot_59_);
v___x_64_ = v___x_61_;
goto v_reusejp_63_;
}
else
{
lean_object* v_reuseFailAlloc_65_; 
v_reuseFailAlloc_65_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_65_, 0, v_toOrderBot_59_);
lean_ctor_set(v_reuseFailAlloc_65_, 1, v_toOrderTop_58_);
v___x_64_ = v_reuseFailAlloc_65_;
goto v_reusejp_63_;
}
v_reusejp_63_:
{
return v___x_64_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instBoundedOrder(lean_object* v_00_u03b1_67_, lean_object* v_inst_68_, lean_object* v_inst_69_){
_start:
{
lean_object* v___x_70_; 
v___x_70_ = lp_mathlib_OrderDual_instBoundedOrder___redArg(v_inst_69_);
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBotForall___redArg___lam__0(lean_object* v_inst_71_, lean_object* v_x_72_){
_start:
{
lean_object* v___x_73_; 
v___x_73_ = lean_apply_1(v_inst_71_, v_x_72_);
return v___x_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBotForall___redArg(lean_object* v_inst_74_){
_start:
{
lean_object* v___f_75_; 
v___f_75_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instBotForall___redArg___lam__0), 2, 1);
lean_closure_set(v___f_75_, 0, v_inst_74_);
return v___f_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBotForall(lean_object* v_00_u03b9_76_, lean_object* v_00_u03b1_x27_77_, lean_object* v_inst_78_){
_start:
{
lean_object* v___f_79_; 
v___f_79_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instBotForall___redArg___lam__0), 2, 1);
lean_closure_set(v___f_79_, 0, v_inst_78_);
return v___f_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instTopForall___redArg(lean_object* v_inst_80_){
_start:
{
lean_object* v___f_81_; 
v___f_81_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instBotForall___redArg___lam__0), 2, 1);
lean_closure_set(v___f_81_, 0, v_inst_80_);
return v___f_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instTopForall(lean_object* v_00_u03b9_82_, lean_object* v_00_u03b1_x27_83_, lean_object* v_inst_84_){
_start:
{
lean_object* v___f_85_; 
v___f_85_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instBotForall___redArg___lam__0), 2, 1);
lean_closure_set(v___f_85_, 0, v_inst_84_);
return v___f_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderBot___redArg___lam__0(lean_object* v_inst_86_, lean_object* v_i_87_){
_start:
{
lean_object* v___x_88_; 
v___x_88_ = lean_apply_1(v_inst_86_, v_i_87_);
return v___x_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderBot___redArg(lean_object* v_inst_89_){
_start:
{
lean_object* v___f_90_; lean_object* v___f_91_; 
v___f_90_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instOrderBot___redArg___lam__0), 2, 1);
lean_closure_set(v___f_90_, 0, v_inst_89_);
v___f_91_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instBotForall___redArg___lam__0), 2, 1);
lean_closure_set(v___f_91_, 0, v___f_90_);
return v___f_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderBot(lean_object* v_00_u03b9_92_, lean_object* v_00_u03b1_x27_93_, lean_object* v_inst_94_, lean_object* v_inst_95_){
_start:
{
lean_object* v___x_96_; 
v___x_96_ = lp_mathlib_Pi_instOrderBot___redArg(v_inst_95_);
return v___x_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderBot___boxed(lean_object* v_00_u03b9_97_, lean_object* v_00_u03b1_x27_98_, lean_object* v_inst_99_, lean_object* v_inst_100_){
_start:
{
lean_object* v_res_101_; 
v_res_101_ = lp_mathlib_Pi_instOrderBot(v_00_u03b9_97_, v_00_u03b1_x27_98_, v_inst_99_, v_inst_100_);
lean_dec_ref(v_inst_99_);
return v_res_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderTop___redArg(lean_object* v_inst_102_){
_start:
{
lean_object* v___f_103_; lean_object* v___f_104_; 
v___f_103_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instOrderBot___redArg___lam__0), 2, 1);
lean_closure_set(v___f_103_, 0, v_inst_102_);
v___f_104_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instBotForall___redArg___lam__0), 2, 1);
lean_closure_set(v___f_104_, 0, v___f_103_);
return v___f_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderTop(lean_object* v_00_u03b9_105_, lean_object* v_00_u03b1_x27_106_, lean_object* v_inst_107_, lean_object* v_inst_108_){
_start:
{
lean_object* v___x_109_; 
v___x_109_ = lp_mathlib_Pi_instOrderTop___redArg(v_inst_108_);
return v___x_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOrderTop___boxed(lean_object* v_00_u03b9_110_, lean_object* v_00_u03b1_x27_111_, lean_object* v_inst_112_, lean_object* v_inst_113_){
_start:
{
lean_object* v_res_114_; 
v_res_114_ = lp_mathlib_Pi_instOrderTop(v_00_u03b9_110_, v_00_u03b1_x27_111_, v_inst_112_, v_inst_113_);
lean_dec_ref(v_inst_112_);
return v_res_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBoundedOrder___redArg___lam__0(lean_object* v_inst_115_, lean_object* v_i_116_){
_start:
{
lean_object* v___x_117_; lean_object* v_toOrderTop_118_; 
v___x_117_ = lean_apply_1(v_inst_115_, v_i_116_);
v_toOrderTop_118_ = lean_ctor_get(v___x_117_, 0);
lean_inc(v_toOrderTop_118_);
lean_dec_ref(v___x_117_);
return v_toOrderTop_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBoundedOrder___redArg___lam__1(lean_object* v_inst_119_, lean_object* v_i_120_){
_start:
{
lean_object* v___x_121_; lean_object* v_toOrderBot_122_; 
v___x_121_ = lean_apply_1(v_inst_119_, v_i_120_);
v_toOrderBot_122_ = lean_ctor_get(v___x_121_, 1);
lean_inc(v_toOrderBot_122_);
lean_dec_ref(v___x_121_);
return v_toOrderBot_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBoundedOrder___redArg(lean_object* v_inst_123_){
_start:
{
lean_object* v___f_124_; lean_object* v___f_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; 
lean_inc_ref(v_inst_123_);
v___f_124_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instBoundedOrder___redArg___lam__0), 2, 1);
lean_closure_set(v___f_124_, 0, v_inst_123_);
v___f_125_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instBoundedOrder___redArg___lam__1), 2, 1);
lean_closure_set(v___f_125_, 0, v_inst_123_);
v___x_126_ = lp_mathlib_Pi_instOrderTop___redArg(v___f_124_);
v___x_127_ = lp_mathlib_Pi_instOrderBot___redArg(v___f_125_);
v___x_128_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_128_, 0, v___x_126_);
lean_ctor_set(v___x_128_, 1, v___x_127_);
return v___x_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBoundedOrder(lean_object* v_00_u03b9_129_, lean_object* v_00_u03b1_x27_130_, lean_object* v_inst_131_, lean_object* v_inst_132_){
_start:
{
lean_object* v___x_133_; 
v___x_133_ = lp_mathlib_Pi_instBoundedOrder___redArg(v_inst_132_);
return v___x_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBoundedOrder___boxed(lean_object* v_00_u03b9_134_, lean_object* v_00_u03b1_x27_135_, lean_object* v_inst_136_, lean_object* v_inst_137_){
_start:
{
lean_object* v_res_138_; 
v_res_138_ = lp_mathlib_Pi_instBoundedOrder(v_00_u03b9_134_, v_00_u03b1_x27_135_, v_inst_136_, v_inst_137_);
lean_dec_ref(v_inst_136_);
return v_res_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrder_ofUnique___redArg(lean_object* v_inst_139_){
_start:
{
lean_object* v___x_140_; 
lean_inc(v_inst_139_);
v___x_140_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_140_, 0, v_inst_139_);
lean_ctor_set(v___x_140_, 1, v_inst_139_);
return v___x_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrder_ofUnique(lean_object* v_00_u03b1_141_, lean_object* v_inst_142_, lean_object* v_inst_143_){
_start:
{
lean_object* v___x_144_; 
lean_inc(v_inst_143_);
v___x_144_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_144_, 0, v_inst_143_);
lean_ctor_set(v___x_144_, 1, v_inst_143_);
return v___x_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrder_ofUnique___boxed(lean_object* v_00_u03b1_145_, lean_object* v_inst_146_, lean_object* v_inst_147_){
_start:
{
lean_object* v_res_148_; 
v_res_148_ = lp_mathlib_BoundedOrder_ofUnique(v_00_u03b1_145_, v_inst_146_, v_inst_147_);
lean_dec_ref(v_inst_146_);
return v_res_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderTop_lift___redArg(lean_object* v_inst_149_){
_start:
{
lean_inc(v_inst_149_);
return v_inst_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderTop_lift___redArg___boxed(lean_object* v_inst_150_){
_start:
{
lean_object* v_res_151_; 
v_res_151_ = lp_mathlib_OrderTop_lift___redArg(v_inst_150_);
lean_dec(v_inst_150_);
return v_res_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderTop_lift(lean_object* v_00_u03b1_152_, lean_object* v_00_u03b2_153_, lean_object* v_inst_154_, lean_object* v_inst_155_, lean_object* v_inst_156_, lean_object* v_inst_157_, lean_object* v_f_158_, lean_object* v_map__le_159_, lean_object* v_map__top_160_){
_start:
{
lean_inc(v_inst_155_);
return v_inst_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderTop_lift___boxed(lean_object* v_00_u03b1_161_, lean_object* v_00_u03b2_162_, lean_object* v_inst_163_, lean_object* v_inst_164_, lean_object* v_inst_165_, lean_object* v_inst_166_, lean_object* v_f_167_, lean_object* v_map__le_168_, lean_object* v_map__top_169_){
_start:
{
lean_object* v_res_170_; 
v_res_170_ = lp_mathlib_OrderTop_lift(v_00_u03b1_161_, v_00_u03b2_162_, v_inst_163_, v_inst_164_, v_inst_165_, v_inst_166_, v_f_167_, v_map__le_168_, v_map__top_169_);
lean_dec(v_f_167_);
lean_dec(v_inst_166_);
lean_dec(v_inst_164_);
return v_res_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderBot_lift___redArg(lean_object* v_inst_171_){
_start:
{
lean_inc(v_inst_171_);
return v_inst_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderBot_lift___redArg___boxed(lean_object* v_inst_172_){
_start:
{
lean_object* v_res_173_; 
v_res_173_ = lp_mathlib_OrderBot_lift___redArg(v_inst_172_);
lean_dec(v_inst_172_);
return v_res_173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderBot_lift(lean_object* v_00_u03b1_174_, lean_object* v_00_u03b2_175_, lean_object* v_inst_176_, lean_object* v_inst_177_, lean_object* v_inst_178_, lean_object* v_inst_179_, lean_object* v_f_180_, lean_object* v_map__le_181_, lean_object* v_map__top_182_){
_start:
{
lean_inc(v_inst_177_);
return v_inst_177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderBot_lift___boxed(lean_object* v_00_u03b1_183_, lean_object* v_00_u03b2_184_, lean_object* v_inst_185_, lean_object* v_inst_186_, lean_object* v_inst_187_, lean_object* v_inst_188_, lean_object* v_f_189_, lean_object* v_map__le_190_, lean_object* v_map__top_191_){
_start:
{
lean_object* v_res_192_; 
v_res_192_ = lp_mathlib_OrderBot_lift(v_00_u03b1_183_, v_00_u03b2_184_, v_inst_185_, v_inst_186_, v_inst_187_, v_inst_188_, v_f_189_, v_map__le_190_, v_map__top_191_);
lean_dec(v_f_189_);
lean_dec(v_inst_188_);
lean_dec(v_inst_186_);
return v_res_192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrder_lift___redArg(lean_object* v_inst_193_, lean_object* v_inst_194_){
_start:
{
lean_object* v___x_195_; 
v___x_195_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_195_, 0, v_inst_193_);
lean_ctor_set(v___x_195_, 1, v_inst_194_);
return v___x_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrder_lift(lean_object* v_00_u03b1_196_, lean_object* v_00_u03b2_197_, lean_object* v_inst_198_, lean_object* v_inst_199_, lean_object* v_inst_200_, lean_object* v_inst_201_, lean_object* v_inst_202_, lean_object* v_f_203_, lean_object* v_map__le_204_, lean_object* v_map__top_205_, lean_object* v_map__bot_206_){
_start:
{
lean_object* v___x_207_; 
v___x_207_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_207_, 0, v_inst_199_);
lean_ctor_set(v___x_207_, 1, v_inst_200_);
return v___x_207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrder_lift___boxed(lean_object* v_00_u03b1_208_, lean_object* v_00_u03b2_209_, lean_object* v_inst_210_, lean_object* v_inst_211_, lean_object* v_inst_212_, lean_object* v_inst_213_, lean_object* v_inst_214_, lean_object* v_f_215_, lean_object* v_map__le_216_, lean_object* v_map__top_217_, lean_object* v_map__bot_218_){
_start:
{
lean_object* v_res_219_; 
v_res_219_ = lp_mathlib_BoundedOrder_lift(v_00_u03b1_208_, v_00_u03b2_209_, v_inst_210_, v_inst_211_, v_inst_212_, v_inst_213_, v_inst_214_, v_f_215_, v_map__le_216_, v_map__top_217_, v_map__bot_218_);
lean_dec(v_f_215_);
lean_dec_ref(v_inst_214_);
return v_res_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_orderBot___redArg(lean_object* v_inst_220_){
_start:
{
lean_inc(v_inst_220_);
return v_inst_220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_orderBot___redArg___boxed(lean_object* v_inst_221_){
_start:
{
lean_object* v_res_222_; 
v_res_222_ = lp_mathlib_Subtype_orderBot___redArg(v_inst_221_);
lean_dec(v_inst_221_);
return v_res_222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_orderBot(lean_object* v_00_u03b1_223_, lean_object* v_p_224_, lean_object* v_inst_225_, lean_object* v_inst_226_, lean_object* v_hbot_227_){
_start:
{
lean_inc(v_inst_226_);
return v_inst_226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_orderBot___boxed(lean_object* v_00_u03b1_228_, lean_object* v_p_229_, lean_object* v_inst_230_, lean_object* v_inst_231_, lean_object* v_hbot_232_){
_start:
{
lean_object* v_res_233_; 
v_res_233_ = lp_mathlib_Subtype_orderBot(v_00_u03b1_228_, v_p_229_, v_inst_230_, v_inst_231_, v_hbot_232_);
lean_dec(v_inst_231_);
return v_res_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_orderTop___redArg(lean_object* v_inst_234_){
_start:
{
lean_inc(v_inst_234_);
return v_inst_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_orderTop___redArg___boxed(lean_object* v_inst_235_){
_start:
{
lean_object* v_res_236_; 
v_res_236_ = lp_mathlib_Subtype_orderTop___redArg(v_inst_235_);
lean_dec(v_inst_235_);
return v_res_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_orderTop(lean_object* v_00_u03b1_237_, lean_object* v_p_238_, lean_object* v_inst_239_, lean_object* v_inst_240_, lean_object* v_hbot_241_){
_start:
{
lean_inc(v_inst_240_);
return v_inst_240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_orderTop___boxed(lean_object* v_00_u03b1_242_, lean_object* v_p_243_, lean_object* v_inst_244_, lean_object* v_inst_245_, lean_object* v_hbot_246_){
_start:
{
lean_object* v_res_247_; 
v_res_247_ = lp_mathlib_Subtype_orderTop(v_00_u03b1_242_, v_p_243_, v_inst_244_, v_inst_245_, v_hbot_246_);
lean_dec(v_inst_245_);
return v_res_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_boundedOrder___redArg(lean_object* v_inst_248_){
_start:
{
lean_object* v_toOrderTop_249_; lean_object* v_toOrderBot_250_; lean_object* v___x_252_; uint8_t v_isShared_253_; uint8_t v_isSharedCheck_257_; 
v_toOrderTop_249_ = lean_ctor_get(v_inst_248_, 0);
v_toOrderBot_250_ = lean_ctor_get(v_inst_248_, 1);
v_isSharedCheck_257_ = !lean_is_exclusive(v_inst_248_);
if (v_isSharedCheck_257_ == 0)
{
v___x_252_ = v_inst_248_;
v_isShared_253_ = v_isSharedCheck_257_;
goto v_resetjp_251_;
}
else
{
lean_inc(v_toOrderBot_250_);
lean_inc(v_toOrderTop_249_);
lean_dec(v_inst_248_);
v___x_252_ = lean_box(0);
v_isShared_253_ = v_isSharedCheck_257_;
goto v_resetjp_251_;
}
v_resetjp_251_:
{
lean_object* v___x_255_; 
if (v_isShared_253_ == 0)
{
v___x_255_ = v___x_252_;
goto v_reusejp_254_;
}
else
{
lean_object* v_reuseFailAlloc_256_; 
v_reuseFailAlloc_256_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_256_, 0, v_toOrderTop_249_);
lean_ctor_set(v_reuseFailAlloc_256_, 1, v_toOrderBot_250_);
v___x_255_ = v_reuseFailAlloc_256_;
goto v_reusejp_254_;
}
v_reusejp_254_:
{
return v___x_255_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_boundedOrder(lean_object* v_00_u03b1_258_, lean_object* v_p_259_, lean_object* v_inst_260_, lean_object* v_inst_261_, lean_object* v_hbot_262_, lean_object* v_htop_263_){
_start:
{
lean_object* v_toOrderTop_264_; lean_object* v_toOrderBot_265_; lean_object* v___x_267_; uint8_t v_isShared_268_; uint8_t v_isSharedCheck_272_; 
v_toOrderTop_264_ = lean_ctor_get(v_inst_261_, 0);
v_toOrderBot_265_ = lean_ctor_get(v_inst_261_, 1);
v_isSharedCheck_272_ = !lean_is_exclusive(v_inst_261_);
if (v_isSharedCheck_272_ == 0)
{
v___x_267_ = v_inst_261_;
v_isShared_268_ = v_isSharedCheck_272_;
goto v_resetjp_266_;
}
else
{
lean_inc(v_toOrderBot_265_);
lean_inc(v_toOrderTop_264_);
lean_dec(v_inst_261_);
v___x_267_ = lean_box(0);
v_isShared_268_ = v_isSharedCheck_272_;
goto v_resetjp_266_;
}
v_resetjp_266_:
{
lean_object* v___x_270_; 
if (v_isShared_268_ == 0)
{
v___x_270_ = v___x_267_;
goto v_reusejp_269_;
}
else
{
lean_object* v_reuseFailAlloc_271_; 
v_reuseFailAlloc_271_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_271_, 0, v_toOrderTop_264_);
lean_ctor_set(v_reuseFailAlloc_271_, 1, v_toOrderBot_265_);
v___x_270_ = v_reuseFailAlloc_271_;
goto v_reusejp_269_;
}
v_reusejp_269_:
{
return v___x_270_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instTop___redArg(lean_object* v_inst_273_, lean_object* v_inst_274_){
_start:
{
lean_object* v___x_275_; 
v___x_275_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_275_, 0, v_inst_273_);
lean_ctor_set(v___x_275_, 1, v_inst_274_);
return v___x_275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instTop(lean_object* v_00_u03b1_276_, lean_object* v_00_u03b2_277_, lean_object* v_inst_278_, lean_object* v_inst_279_){
_start:
{
lean_object* v___x_280_; 
v___x_280_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_280_, 0, v_inst_278_);
lean_ctor_set(v___x_280_, 1, v_inst_279_);
return v___x_280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instBot___redArg(lean_object* v_inst_281_, lean_object* v_inst_282_){
_start:
{
lean_object* v___x_283_; 
v___x_283_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_283_, 0, v_inst_281_);
lean_ctor_set(v___x_283_, 1, v_inst_282_);
return v___x_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instBot(lean_object* v_00_u03b1_284_, lean_object* v_00_u03b2_285_, lean_object* v_inst_286_, lean_object* v_inst_287_){
_start:
{
lean_object* v___x_288_; 
v___x_288_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_288_, 0, v_inst_286_);
lean_ctor_set(v___x_288_, 1, v_inst_287_);
return v___x_288_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instOrderTop___redArg(lean_object* v_inst_289_, lean_object* v_inst_290_){
_start:
{
lean_object* v___x_291_; 
v___x_291_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_291_, 0, v_inst_289_);
lean_ctor_set(v___x_291_, 1, v_inst_290_);
return v___x_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instOrderTop(lean_object* v_00_u03b1_292_, lean_object* v_00_u03b2_293_, lean_object* v_inst_294_, lean_object* v_inst_295_, lean_object* v_inst_296_, lean_object* v_inst_297_){
_start:
{
lean_object* v___x_298_; 
v___x_298_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_298_, 0, v_inst_296_);
lean_ctor_set(v___x_298_, 1, v_inst_297_);
return v___x_298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instOrderBot___redArg(lean_object* v_inst_299_, lean_object* v_inst_300_){
_start:
{
lean_object* v___x_301_; 
v___x_301_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_301_, 0, v_inst_299_);
lean_ctor_set(v___x_301_, 1, v_inst_300_);
return v___x_301_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instOrderBot(lean_object* v_00_u03b1_302_, lean_object* v_00_u03b2_303_, lean_object* v_inst_304_, lean_object* v_inst_305_, lean_object* v_inst_306_, lean_object* v_inst_307_){
_start:
{
lean_object* v___x_308_; 
v___x_308_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_308_, 0, v_inst_306_);
lean_ctor_set(v___x_308_, 1, v_inst_307_);
return v___x_308_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instBoundedOrder___redArg(lean_object* v_inst_309_, lean_object* v_inst_310_){
_start:
{
lean_object* v_toOrderTop_311_; lean_object* v_toOrderBot_312_; lean_object* v___x_314_; uint8_t v_isShared_315_; uint8_t v_isSharedCheck_329_; 
v_toOrderTop_311_ = lean_ctor_get(v_inst_309_, 0);
v_toOrderBot_312_ = lean_ctor_get(v_inst_309_, 1);
v_isSharedCheck_329_ = !lean_is_exclusive(v_inst_309_);
if (v_isSharedCheck_329_ == 0)
{
v___x_314_ = v_inst_309_;
v_isShared_315_ = v_isSharedCheck_329_;
goto v_resetjp_313_;
}
else
{
lean_inc(v_toOrderBot_312_);
lean_inc(v_toOrderTop_311_);
lean_dec(v_inst_309_);
v___x_314_ = lean_box(0);
v_isShared_315_ = v_isSharedCheck_329_;
goto v_resetjp_313_;
}
v_resetjp_313_:
{
lean_object* v_toOrderTop_316_; lean_object* v_toOrderBot_317_; lean_object* v___x_319_; uint8_t v_isShared_320_; uint8_t v_isSharedCheck_328_; 
v_toOrderTop_316_ = lean_ctor_get(v_inst_310_, 0);
v_toOrderBot_317_ = lean_ctor_get(v_inst_310_, 1);
v_isSharedCheck_328_ = !lean_is_exclusive(v_inst_310_);
if (v_isSharedCheck_328_ == 0)
{
v___x_319_ = v_inst_310_;
v_isShared_320_ = v_isSharedCheck_328_;
goto v_resetjp_318_;
}
else
{
lean_inc(v_toOrderBot_317_);
lean_inc(v_toOrderTop_316_);
lean_dec(v_inst_310_);
v___x_319_ = lean_box(0);
v_isShared_320_ = v_isSharedCheck_328_;
goto v_resetjp_318_;
}
v_resetjp_318_:
{
lean_object* v___x_322_; 
if (v_isShared_315_ == 0)
{
lean_ctor_set(v___x_314_, 1, v_toOrderTop_316_);
v___x_322_ = v___x_314_;
goto v_reusejp_321_;
}
else
{
lean_object* v_reuseFailAlloc_327_; 
v_reuseFailAlloc_327_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_327_, 0, v_toOrderTop_311_);
lean_ctor_set(v_reuseFailAlloc_327_, 1, v_toOrderTop_316_);
v___x_322_ = v_reuseFailAlloc_327_;
goto v_reusejp_321_;
}
v_reusejp_321_:
{
lean_object* v___x_323_; lean_object* v___x_325_; 
v___x_323_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_323_, 0, v_toOrderBot_312_);
lean_ctor_set(v___x_323_, 1, v_toOrderBot_317_);
if (v_isShared_320_ == 0)
{
lean_ctor_set(v___x_319_, 1, v___x_323_);
lean_ctor_set(v___x_319_, 0, v___x_322_);
v___x_325_ = v___x_319_;
goto v_reusejp_324_;
}
else
{
lean_object* v_reuseFailAlloc_326_; 
v_reuseFailAlloc_326_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_326_, 0, v___x_322_);
lean_ctor_set(v_reuseFailAlloc_326_, 1, v___x_323_);
v___x_325_ = v_reuseFailAlloc_326_;
goto v_reusejp_324_;
}
v_reusejp_324_:
{
return v___x_325_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instBoundedOrder(lean_object* v_00_u03b1_330_, lean_object* v_00_u03b2_331_, lean_object* v_inst_332_, lean_object* v_inst_333_, lean_object* v_inst_334_, lean_object* v_inst_335_){
_start:
{
lean_object* v___x_336_; 
v___x_336_ = lp_mathlib_Prod_instBoundedOrder___redArg(v_inst_334_, v_inst_335_);
return v___x_336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instTop___redArg(lean_object* v_inst_337_){
_start:
{
lean_inc(v_inst_337_);
return v_inst_337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instTop___redArg___boxed(lean_object* v_inst_338_){
_start:
{
lean_object* v_res_339_; 
v_res_339_ = lp_mathlib_ULift_instTop___redArg(v_inst_338_);
lean_dec(v_inst_338_);
return v_res_339_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instTop(lean_object* v_00_u03b1_340_, lean_object* v_inst_341_){
_start:
{
lean_inc(v_inst_341_);
return v_inst_341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instTop___boxed(lean_object* v_00_u03b1_342_, lean_object* v_inst_343_){
_start:
{
lean_object* v_res_344_; 
v_res_344_ = lp_mathlib_ULift_instTop(v_00_u03b1_342_, v_inst_343_);
lean_dec(v_inst_343_);
return v_res_344_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instBot___redArg(lean_object* v_inst_345_){
_start:
{
lean_inc(v_inst_345_);
return v_inst_345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instBot___redArg___boxed(lean_object* v_inst_346_){
_start:
{
lean_object* v_res_347_; 
v_res_347_ = lp_mathlib_ULift_instBot___redArg(v_inst_346_);
lean_dec(v_inst_346_);
return v_res_347_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instBot(lean_object* v_00_u03b1_348_, lean_object* v_inst_349_){
_start:
{
lean_inc(v_inst_349_);
return v_inst_349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instBot___boxed(lean_object* v_00_u03b1_350_, lean_object* v_inst_351_){
_start:
{
lean_object* v_res_352_; 
v_res_352_ = lp_mathlib_ULift_instBot(v_00_u03b1_350_, v_inst_351_);
lean_dec(v_inst_351_);
return v_res_352_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instOrderBot___redArg(lean_object* v_inst_353_){
_start:
{
lean_inc(v_inst_353_);
return v_inst_353_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instOrderBot___redArg___boxed(lean_object* v_inst_354_){
_start:
{
lean_object* v_res_355_; 
v_res_355_ = lp_mathlib_ULift_instOrderBot___redArg(v_inst_354_);
lean_dec(v_inst_354_);
return v_res_355_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instOrderBot(lean_object* v_00_u03b1_356_, lean_object* v_inst_357_, lean_object* v_inst_358_){
_start:
{
lean_inc(v_inst_358_);
return v_inst_358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instOrderBot___boxed(lean_object* v_00_u03b1_359_, lean_object* v_inst_360_, lean_object* v_inst_361_){
_start:
{
lean_object* v_res_362_; 
v_res_362_ = lp_mathlib_ULift_instOrderBot(v_00_u03b1_359_, v_inst_360_, v_inst_361_);
lean_dec(v_inst_361_);
return v_res_362_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instOrderTop___redArg(lean_object* v_inst_363_){
_start:
{
lean_inc(v_inst_363_);
return v_inst_363_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instOrderTop___redArg___boxed(lean_object* v_inst_364_){
_start:
{
lean_object* v_res_365_; 
v_res_365_ = lp_mathlib_ULift_instOrderTop___redArg(v_inst_364_);
lean_dec(v_inst_364_);
return v_res_365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instOrderTop(lean_object* v_00_u03b1_366_, lean_object* v_inst_367_, lean_object* v_inst_368_){
_start:
{
lean_inc(v_inst_368_);
return v_inst_368_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instOrderTop___boxed(lean_object* v_00_u03b1_369_, lean_object* v_inst_370_, lean_object* v_inst_371_){
_start:
{
lean_object* v_res_372_; 
v_res_372_ = lp_mathlib_ULift_instOrderTop(v_00_u03b1_369_, v_inst_370_, v_inst_371_);
lean_dec(v_inst_371_);
return v_res_372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instBoundedOrder___redArg(lean_object* v_inst_373_){
_start:
{
lean_object* v_toOrderTop_374_; lean_object* v_toOrderBot_375_; lean_object* v___x_377_; uint8_t v_isShared_378_; uint8_t v_isSharedCheck_382_; 
v_toOrderTop_374_ = lean_ctor_get(v_inst_373_, 0);
v_toOrderBot_375_ = lean_ctor_get(v_inst_373_, 1);
v_isSharedCheck_382_ = !lean_is_exclusive(v_inst_373_);
if (v_isSharedCheck_382_ == 0)
{
v___x_377_ = v_inst_373_;
v_isShared_378_ = v_isSharedCheck_382_;
goto v_resetjp_376_;
}
else
{
lean_inc(v_toOrderBot_375_);
lean_inc(v_toOrderTop_374_);
lean_dec(v_inst_373_);
v___x_377_ = lean_box(0);
v_isShared_378_ = v_isSharedCheck_382_;
goto v_resetjp_376_;
}
v_resetjp_376_:
{
lean_object* v___x_380_; 
if (v_isShared_378_ == 0)
{
v___x_380_ = v___x_377_;
goto v_reusejp_379_;
}
else
{
lean_object* v_reuseFailAlloc_381_; 
v_reuseFailAlloc_381_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_381_, 0, v_toOrderTop_374_);
lean_ctor_set(v_reuseFailAlloc_381_, 1, v_toOrderBot_375_);
v___x_380_ = v_reuseFailAlloc_381_;
goto v_reusejp_379_;
}
v_reusejp_379_:
{
return v___x_380_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instBoundedOrder(lean_object* v_00_u03b1_383_, lean_object* v_inst_384_, lean_object* v_inst_385_){
_start:
{
lean_object* v___x_386_; 
v___x_386_ = lp_mathlib_ULift_instBoundedOrder___redArg(v_inst_385_);
return v___x_386_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Max(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_ULift(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ByCases(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Finiteness_Attr(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_BoundedOrder_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Max(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ByCases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Finiteness_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_BoundedOrder_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Order_Max(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_ULift(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ByCases(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Finiteness_Attr(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_BoundedOrder_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Max(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ByCases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Finiteness_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_BoundedOrder_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_BoundedOrder_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_BoundedOrder_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
