// Lean compiler output
// Module: Mathlib.Order.LatticeIntervals
// Imports: public import Init public meta import Init public import Mathlib.Order.Bounds.Basic
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
lean_object* lp_mathlib_Lattice_toSemilatticeInf___redArg(lean_object*);
lean_object* lp_mathlib_Subtype_semilatticeInf___redArg(lean_object*);
lean_object* lp_mathlib_Lattice_mkDual___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Subtype_preorder(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ico_semilatticeInf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ico_semilatticeInf(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ico_semilatticeInf___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ioc_semilatticeSup___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ioc_semilatticeSup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ioc_semilatticeSup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ioc_semilatticeSup___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ico_orderBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ico_orderBot___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ico_orderBot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ico_orderBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ioc_orderTop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ioc_orderTop___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ioc_orderTop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ioc_orderTop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Iio_semilatticeInf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Iio_semilatticeInf(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Iio_semilatticeInf___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ioi_semilatticeSup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ioi_semilatticeSup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ioi_semilatticeSup___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_semilatticeInf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_semilatticeInf(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_semilatticeInf___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_semilatticeSup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_semilatticeSup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_semilatticeSup___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_semilatticeSup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_semilatticeSup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_semilatticeSup___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_semilatticeInf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_semilatticeInf(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_semilatticeInf___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_instLatticeElem___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_instLatticeElem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_instLatticeElem(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_instLatticeElem___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_instLatticeElem___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_instLatticeElem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_instLatticeElem(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_instLatticeElem___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_instDistribLatticeElem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_instDistribLatticeElem(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_instDistribLatticeElem___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_instDistribLatticeElem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_instDistribLatticeElem(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_instDistribLatticeElem___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_orderTop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_orderTop___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_orderTop(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_orderTop___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_orderBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_orderBot___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_orderBot(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_orderBot___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_orderBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_orderBot___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_orderBot(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_orderBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_orderTop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_orderTop___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_orderTop(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_orderTop___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_instBoundedOrderElemOfOrderBot___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_instBoundedOrderElemOfOrderBot(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_instBoundedOrderElemOfOrderBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_instBoundedOrderElemOfOrderTop___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_instBoundedOrderElemOfOrderTop(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_instBoundedOrderElemOfOrderTop___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_semilatticeInf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_semilatticeInf(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_semilatticeInf___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_semilatticeSup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_semilatticeSup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_semilatticeSup___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_lattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_lattice(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_lattice___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_instOrderBotElemOfFactLe___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_instOrderBotElemOfFactLe___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_instOrderBotElemOfFactLe(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_instOrderBotElemOfFactLe___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_instOrderTopElemOfFactLe___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_instOrderTopElemOfFactLe___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_instOrderTopElemOfFactLe(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_instOrderTopElemOfFactLe___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_instBoundedOrderElemOfFactLe___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_instBoundedOrderElemOfFactLe(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_instBoundedOrderElemOfFactLe___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Ico_semilatticeInf___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lp_mathlib_Subtype_semilatticeInf___redArg(v_inst_1_);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ico_semilatticeInf(lean_object* v_00_u03b1_3_, lean_object* v_inst_4_, lean_object* v_a_5_, lean_object* v_b_6_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lp_mathlib_Subtype_semilatticeInf___redArg(v_inst_4_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ico_semilatticeInf___boxed(lean_object* v_00_u03b1_8_, lean_object* v_inst_9_, lean_object* v_a_10_, lean_object* v_b_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_mathlib_Set_Ico_semilatticeInf(v_00_u03b1_8_, v_inst_9_, v_a_10_, v_b_11_);
lean_dec(v_b_11_);
lean_dec(v_a_10_);
return v_res_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ioc_semilatticeSup___redArg___lam__0(lean_object* v_sup_13_, lean_object* v_x_14_, lean_object* v_y_15_){
_start:
{
lean_object* v___x_16_; 
v___x_16_ = lean_apply_2(v_sup_13_, v_x_14_, v_y_15_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ioc_semilatticeSup___redArg(lean_object* v_inst_17_){
_start:
{
lean_object* v_toPartialOrder_18_; lean_object* v_sup_19_; lean_object* v___x_21_; uint8_t v_isShared_22_; uint8_t v_isSharedCheck_28_; 
v_toPartialOrder_18_ = lean_ctor_get(v_inst_17_, 0);
v_sup_19_ = lean_ctor_get(v_inst_17_, 1);
v_isSharedCheck_28_ = !lean_is_exclusive(v_inst_17_);
if (v_isSharedCheck_28_ == 0)
{
v___x_21_ = v_inst_17_;
v_isShared_22_ = v_isSharedCheck_28_;
goto v_resetjp_20_;
}
else
{
lean_inc(v_sup_19_);
lean_inc(v_toPartialOrder_18_);
lean_dec(v_inst_17_);
v___x_21_ = lean_box(0);
v_isShared_22_ = v_isSharedCheck_28_;
goto v_resetjp_20_;
}
v_resetjp_20_:
{
lean_object* v___f_23_; lean_object* v___x_24_; lean_object* v___x_26_; 
v___f_23_ = lean_alloc_closure((void*)(lp_mathlib_Set_Ioc_semilatticeSup___redArg___lam__0), 3, 1);
lean_closure_set(v___f_23_, 0, v_sup_19_);
v___x_24_ = lp_mathlib_Subtype_preorder(lean_box(0), v_toPartialOrder_18_, lean_box(0));
lean_dec_ref(v_toPartialOrder_18_);
if (v_isShared_22_ == 0)
{
lean_ctor_set(v___x_21_, 1, v___f_23_);
lean_ctor_set(v___x_21_, 0, v___x_24_);
v___x_26_ = v___x_21_;
goto v_reusejp_25_;
}
else
{
lean_object* v_reuseFailAlloc_27_; 
v_reuseFailAlloc_27_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_27_, 0, v___x_24_);
lean_ctor_set(v_reuseFailAlloc_27_, 1, v___f_23_);
v___x_26_ = v_reuseFailAlloc_27_;
goto v_reusejp_25_;
}
v_reusejp_25_:
{
return v___x_26_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ioc_semilatticeSup(lean_object* v_00_u03b1_29_, lean_object* v_inst_30_, lean_object* v_a_31_, lean_object* v_b_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = lp_mathlib_Set_Ioc_semilatticeSup___redArg(v_inst_30_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ioc_semilatticeSup___boxed(lean_object* v_00_u03b1_34_, lean_object* v_inst_35_, lean_object* v_a_36_, lean_object* v_b_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_mathlib_Set_Ioc_semilatticeSup(v_00_u03b1_34_, v_inst_35_, v_a_36_, v_b_37_);
lean_dec(v_b_37_);
lean_dec(v_a_36_);
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ico_orderBot___redArg(lean_object* v_a_39_){
_start:
{
lean_inc(v_a_39_);
return v_a_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ico_orderBot___redArg___boxed(lean_object* v_a_40_){
_start:
{
lean_object* v_res_41_; 
v_res_41_ = lp_mathlib_Set_Ico_orderBot___redArg(v_a_40_);
lean_dec(v_a_40_);
return v_res_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ico_orderBot(lean_object* v_00_u03b1_42_, lean_object* v_inst_43_, lean_object* v_a_44_, lean_object* v_b_45_, lean_object* v_inst_46_){
_start:
{
lean_inc(v_a_44_);
return v_a_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ico_orderBot___boxed(lean_object* v_00_u03b1_47_, lean_object* v_inst_48_, lean_object* v_a_49_, lean_object* v_b_50_, lean_object* v_inst_51_){
_start:
{
lean_object* v_res_52_; 
v_res_52_ = lp_mathlib_Set_Ico_orderBot(v_00_u03b1_47_, v_inst_48_, v_a_49_, v_b_50_, v_inst_51_);
lean_dec(v_b_50_);
lean_dec(v_a_49_);
lean_dec_ref(v_inst_48_);
return v_res_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ioc_orderTop___redArg(lean_object* v_a_53_){
_start:
{
lean_inc(v_a_53_);
return v_a_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ioc_orderTop___redArg___boxed(lean_object* v_a_54_){
_start:
{
lean_object* v_res_55_; 
v_res_55_ = lp_mathlib_Set_Ioc_orderTop___redArg(v_a_54_);
lean_dec(v_a_54_);
return v_res_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ioc_orderTop(lean_object* v_00_u03b1_56_, lean_object* v_inst_57_, lean_object* v_a_58_, lean_object* v_b_59_, lean_object* v_inst_60_){
_start:
{
lean_inc(v_a_58_);
return v_a_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ioc_orderTop___boxed(lean_object* v_00_u03b1_61_, lean_object* v_inst_62_, lean_object* v_a_63_, lean_object* v_b_64_, lean_object* v_inst_65_){
_start:
{
lean_object* v_res_66_; 
v_res_66_ = lp_mathlib_Set_Ioc_orderTop(v_00_u03b1_61_, v_inst_62_, v_a_63_, v_b_64_, v_inst_65_);
lean_dec(v_b_64_);
lean_dec(v_a_63_);
lean_dec_ref(v_inst_62_);
return v_res_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Iio_semilatticeInf___redArg(lean_object* v_inst_67_){
_start:
{
lean_object* v___x_68_; 
v___x_68_ = lp_mathlib_Subtype_semilatticeInf___redArg(v_inst_67_);
return v___x_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Iio_semilatticeInf(lean_object* v_00_u03b1_69_, lean_object* v_inst_70_, lean_object* v_a_71_){
_start:
{
lean_object* v___x_72_; 
v___x_72_ = lp_mathlib_Subtype_semilatticeInf___redArg(v_inst_70_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Iio_semilatticeInf___boxed(lean_object* v_00_u03b1_73_, lean_object* v_inst_74_, lean_object* v_a_75_){
_start:
{
lean_object* v_res_76_; 
v_res_76_ = lp_mathlib_Set_Iio_semilatticeInf(v_00_u03b1_73_, v_inst_74_, v_a_75_);
lean_dec(v_a_75_);
return v_res_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ioi_semilatticeSup___redArg(lean_object* v_inst_77_){
_start:
{
lean_object* v_toPartialOrder_78_; lean_object* v_sup_79_; lean_object* v___x_81_; uint8_t v_isShared_82_; uint8_t v_isSharedCheck_88_; 
v_toPartialOrder_78_ = lean_ctor_get(v_inst_77_, 0);
v_sup_79_ = lean_ctor_get(v_inst_77_, 1);
v_isSharedCheck_88_ = !lean_is_exclusive(v_inst_77_);
if (v_isSharedCheck_88_ == 0)
{
v___x_81_ = v_inst_77_;
v_isShared_82_ = v_isSharedCheck_88_;
goto v_resetjp_80_;
}
else
{
lean_inc(v_sup_79_);
lean_inc(v_toPartialOrder_78_);
lean_dec(v_inst_77_);
v___x_81_ = lean_box(0);
v_isShared_82_ = v_isSharedCheck_88_;
goto v_resetjp_80_;
}
v_resetjp_80_:
{
lean_object* v___f_83_; lean_object* v___x_84_; lean_object* v___x_86_; 
v___f_83_ = lean_alloc_closure((void*)(lp_mathlib_Set_Ioc_semilatticeSup___redArg___lam__0), 3, 1);
lean_closure_set(v___f_83_, 0, v_sup_79_);
v___x_84_ = lp_mathlib_Subtype_preorder(lean_box(0), v_toPartialOrder_78_, lean_box(0));
lean_dec_ref(v_toPartialOrder_78_);
if (v_isShared_82_ == 0)
{
lean_ctor_set(v___x_81_, 1, v___f_83_);
lean_ctor_set(v___x_81_, 0, v___x_84_);
v___x_86_ = v___x_81_;
goto v_reusejp_85_;
}
else
{
lean_object* v_reuseFailAlloc_87_; 
v_reuseFailAlloc_87_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_87_, 0, v___x_84_);
lean_ctor_set(v_reuseFailAlloc_87_, 1, v___f_83_);
v___x_86_ = v_reuseFailAlloc_87_;
goto v_reusejp_85_;
}
v_reusejp_85_:
{
return v___x_86_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ioi_semilatticeSup(lean_object* v_00_u03b1_89_, lean_object* v_inst_90_, lean_object* v_a_91_){
_start:
{
lean_object* v___x_92_; 
v___x_92_ = lp_mathlib_Set_Ioi_semilatticeSup___redArg(v_inst_90_);
return v___x_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ioi_semilatticeSup___boxed(lean_object* v_00_u03b1_93_, lean_object* v_inst_94_, lean_object* v_a_95_){
_start:
{
lean_object* v_res_96_; 
v_res_96_ = lp_mathlib_Set_Ioi_semilatticeSup(v_00_u03b1_93_, v_inst_94_, v_a_95_);
lean_dec(v_a_95_);
return v_res_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_semilatticeInf___redArg(lean_object* v_inst_97_){
_start:
{
lean_object* v___x_98_; 
v___x_98_ = lp_mathlib_Subtype_semilatticeInf___redArg(v_inst_97_);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_semilatticeInf(lean_object* v_00_u03b1_99_, lean_object* v_a_100_, lean_object* v_inst_101_){
_start:
{
lean_object* v___x_102_; 
v___x_102_ = lp_mathlib_Subtype_semilatticeInf___redArg(v_inst_101_);
return v___x_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_semilatticeInf___boxed(lean_object* v_00_u03b1_103_, lean_object* v_a_104_, lean_object* v_inst_105_){
_start:
{
lean_object* v_res_106_; 
v_res_106_ = lp_mathlib_Set_Iic_semilatticeInf(v_00_u03b1_103_, v_a_104_, v_inst_105_);
lean_dec(v_a_104_);
return v_res_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_semilatticeSup___redArg(lean_object* v_inst_107_){
_start:
{
lean_object* v_toPartialOrder_108_; lean_object* v_sup_109_; lean_object* v___x_111_; uint8_t v_isShared_112_; uint8_t v_isSharedCheck_118_; 
v_toPartialOrder_108_ = lean_ctor_get(v_inst_107_, 0);
v_sup_109_ = lean_ctor_get(v_inst_107_, 1);
v_isSharedCheck_118_ = !lean_is_exclusive(v_inst_107_);
if (v_isSharedCheck_118_ == 0)
{
v___x_111_ = v_inst_107_;
v_isShared_112_ = v_isSharedCheck_118_;
goto v_resetjp_110_;
}
else
{
lean_inc(v_sup_109_);
lean_inc(v_toPartialOrder_108_);
lean_dec(v_inst_107_);
v___x_111_ = lean_box(0);
v_isShared_112_ = v_isSharedCheck_118_;
goto v_resetjp_110_;
}
v_resetjp_110_:
{
lean_object* v___f_113_; lean_object* v___x_114_; lean_object* v___x_116_; 
v___f_113_ = lean_alloc_closure((void*)(lp_mathlib_Set_Ioc_semilatticeSup___redArg___lam__0), 3, 1);
lean_closure_set(v___f_113_, 0, v_sup_109_);
v___x_114_ = lp_mathlib_Subtype_preorder(lean_box(0), v_toPartialOrder_108_, lean_box(0));
lean_dec_ref(v_toPartialOrder_108_);
if (v_isShared_112_ == 0)
{
lean_ctor_set(v___x_111_, 1, v___f_113_);
lean_ctor_set(v___x_111_, 0, v___x_114_);
v___x_116_ = v___x_111_;
goto v_reusejp_115_;
}
else
{
lean_object* v_reuseFailAlloc_117_; 
v_reuseFailAlloc_117_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_117_, 0, v___x_114_);
lean_ctor_set(v_reuseFailAlloc_117_, 1, v___f_113_);
v___x_116_ = v_reuseFailAlloc_117_;
goto v_reusejp_115_;
}
v_reusejp_115_:
{
return v___x_116_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_semilatticeSup(lean_object* v_00_u03b1_119_, lean_object* v_a_120_, lean_object* v_inst_121_){
_start:
{
lean_object* v___x_122_; 
v___x_122_ = lp_mathlib_Set_Ici_semilatticeSup___redArg(v_inst_121_);
return v___x_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_semilatticeSup___boxed(lean_object* v_00_u03b1_123_, lean_object* v_a_124_, lean_object* v_inst_125_){
_start:
{
lean_object* v_res_126_; 
v_res_126_ = lp_mathlib_Set_Ici_semilatticeSup(v_00_u03b1_123_, v_a_124_, v_inst_125_);
lean_dec(v_a_124_);
return v_res_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_semilatticeSup___redArg(lean_object* v_inst_127_){
_start:
{
lean_object* v_toPartialOrder_128_; lean_object* v_sup_129_; lean_object* v___x_131_; uint8_t v_isShared_132_; uint8_t v_isSharedCheck_138_; 
v_toPartialOrder_128_ = lean_ctor_get(v_inst_127_, 0);
v_sup_129_ = lean_ctor_get(v_inst_127_, 1);
v_isSharedCheck_138_ = !lean_is_exclusive(v_inst_127_);
if (v_isSharedCheck_138_ == 0)
{
v___x_131_ = v_inst_127_;
v_isShared_132_ = v_isSharedCheck_138_;
goto v_resetjp_130_;
}
else
{
lean_inc(v_sup_129_);
lean_inc(v_toPartialOrder_128_);
lean_dec(v_inst_127_);
v___x_131_ = lean_box(0);
v_isShared_132_ = v_isSharedCheck_138_;
goto v_resetjp_130_;
}
v_resetjp_130_:
{
lean_object* v___f_133_; lean_object* v___x_134_; lean_object* v___x_136_; 
v___f_133_ = lean_alloc_closure((void*)(lp_mathlib_Set_Ioc_semilatticeSup___redArg___lam__0), 3, 1);
lean_closure_set(v___f_133_, 0, v_sup_129_);
v___x_134_ = lp_mathlib_Subtype_preorder(lean_box(0), v_toPartialOrder_128_, lean_box(0));
lean_dec_ref(v_toPartialOrder_128_);
if (v_isShared_132_ == 0)
{
lean_ctor_set(v___x_131_, 1, v___f_133_);
lean_ctor_set(v___x_131_, 0, v___x_134_);
v___x_136_ = v___x_131_;
goto v_reusejp_135_;
}
else
{
lean_object* v_reuseFailAlloc_137_; 
v_reuseFailAlloc_137_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_137_, 0, v___x_134_);
lean_ctor_set(v_reuseFailAlloc_137_, 1, v___f_133_);
v___x_136_ = v_reuseFailAlloc_137_;
goto v_reusejp_135_;
}
v_reusejp_135_:
{
return v___x_136_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_semilatticeSup(lean_object* v_00_u03b1_139_, lean_object* v_a_140_, lean_object* v_inst_141_){
_start:
{
lean_object* v___x_142_; 
v___x_142_ = lp_mathlib_Set_Iic_semilatticeSup___redArg(v_inst_141_);
return v___x_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_semilatticeSup___boxed(lean_object* v_00_u03b1_143_, lean_object* v_a_144_, lean_object* v_inst_145_){
_start:
{
lean_object* v_res_146_; 
v_res_146_ = lp_mathlib_Set_Iic_semilatticeSup(v_00_u03b1_143_, v_a_144_, v_inst_145_);
lean_dec(v_a_144_);
return v_res_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_semilatticeInf___redArg(lean_object* v_inst_147_){
_start:
{
lean_object* v___x_148_; 
v___x_148_ = lp_mathlib_Subtype_semilatticeInf___redArg(v_inst_147_);
return v___x_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_semilatticeInf(lean_object* v_00_u03b1_149_, lean_object* v_a_150_, lean_object* v_inst_151_){
_start:
{
lean_object* v___x_152_; 
v___x_152_ = lp_mathlib_Subtype_semilatticeInf___redArg(v_inst_151_);
return v___x_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_semilatticeInf___boxed(lean_object* v_00_u03b1_153_, lean_object* v_a_154_, lean_object* v_inst_155_){
_start:
{
lean_object* v_res_156_; 
v_res_156_ = lp_mathlib_Set_Ici_semilatticeInf(v_00_u03b1_153_, v_a_154_, v_inst_155_);
lean_dec(v_a_154_);
return v_res_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_instLatticeElem___redArg___lam__0(lean_object* v_inf_157_, lean_object* v_x_158_, lean_object* v_y_159_){
_start:
{
lean_object* v___x_160_; 
v___x_160_ = lean_apply_2(v_inf_157_, v_x_158_, v_y_159_);
return v___x_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_instLatticeElem___redArg(lean_object* v_inst_161_){
_start:
{
lean_object* v_toSemilatticeSup_162_; lean_object* v_inf_163_; lean_object* v___x_165_; uint8_t v_isShared_166_; uint8_t v_isSharedCheck_172_; 
v_toSemilatticeSup_162_ = lean_ctor_get(v_inst_161_, 0);
v_inf_163_ = lean_ctor_get(v_inst_161_, 1);
v_isSharedCheck_172_ = !lean_is_exclusive(v_inst_161_);
if (v_isSharedCheck_172_ == 0)
{
v___x_165_ = v_inst_161_;
v_isShared_166_ = v_isSharedCheck_172_;
goto v_resetjp_164_;
}
else
{
lean_inc(v_inf_163_);
lean_inc(v_toSemilatticeSup_162_);
lean_dec(v_inst_161_);
v___x_165_ = lean_box(0);
v_isShared_166_ = v_isSharedCheck_172_;
goto v_resetjp_164_;
}
v_resetjp_164_:
{
lean_object* v___f_167_; lean_object* v___x_168_; lean_object* v___x_170_; 
v___f_167_ = lean_alloc_closure((void*)(lp_mathlib_Set_Iic_instLatticeElem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_167_, 0, v_inf_163_);
v___x_168_ = lp_mathlib_Set_Iic_semilatticeSup___redArg(v_toSemilatticeSup_162_);
if (v_isShared_166_ == 0)
{
lean_ctor_set(v___x_165_, 1, v___f_167_);
lean_ctor_set(v___x_165_, 0, v___x_168_);
v___x_170_ = v___x_165_;
goto v_reusejp_169_;
}
else
{
lean_object* v_reuseFailAlloc_171_; 
v_reuseFailAlloc_171_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_171_, 0, v___x_168_);
lean_ctor_set(v_reuseFailAlloc_171_, 1, v___f_167_);
v___x_170_ = v_reuseFailAlloc_171_;
goto v_reusejp_169_;
}
v_reusejp_169_:
{
return v___x_170_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_instLatticeElem(lean_object* v_00_u03b1_173_, lean_object* v_a_174_, lean_object* v_inst_175_){
_start:
{
lean_object* v___x_176_; 
v___x_176_ = lp_mathlib_Set_Iic_instLatticeElem___redArg(v_inst_175_);
return v___x_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_instLatticeElem___boxed(lean_object* v_00_u03b1_177_, lean_object* v_a_178_, lean_object* v_inst_179_){
_start:
{
lean_object* v_res_180_; 
v_res_180_ = lp_mathlib_Set_Iic_instLatticeElem(v_00_u03b1_177_, v_a_178_, v_inst_179_);
lean_dec(v_a_178_);
return v_res_180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_instLatticeElem___redArg___lam__0(lean_object* v_toSemilatticeSup_181_, lean_object* v_x_182_, lean_object* v_y_183_){
_start:
{
lean_object* v_sup_184_; lean_object* v___x_185_; 
v_sup_184_ = lean_ctor_get(v_toSemilatticeSup_181_, 1);
lean_inc(v_sup_184_);
lean_dec_ref(v_toSemilatticeSup_181_);
v___x_185_ = lean_apply_2(v_sup_184_, v_x_182_, v_y_183_);
return v___x_185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_instLatticeElem___redArg(lean_object* v_inst_186_){
_start:
{
lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v_toSemilatticeSup_189_; lean_object* v___f_190_; lean_object* v___x_191_; 
lean_inc_ref(v_inst_186_);
v___x_187_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_inst_186_);
v___x_188_ = lp_mathlib_Subtype_semilatticeInf___redArg(v___x_187_);
v_toSemilatticeSup_189_ = lean_ctor_get(v_inst_186_, 0);
lean_inc_ref(v_toSemilatticeSup_189_);
lean_dec_ref(v_inst_186_);
v___f_190_ = lean_alloc_closure((void*)(lp_mathlib_Set_Ici_instLatticeElem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_190_, 0, v_toSemilatticeSup_189_);
v___x_191_ = lp_mathlib_Lattice_mkDual___redArg(v___x_188_, v___f_190_);
return v___x_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_instLatticeElem(lean_object* v_00_u03b1_192_, lean_object* v_a_193_, lean_object* v_inst_194_){
_start:
{
lean_object* v___x_195_; 
v___x_195_ = lp_mathlib_Set_Ici_instLatticeElem___redArg(v_inst_194_);
return v___x_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_instLatticeElem___boxed(lean_object* v_00_u03b1_196_, lean_object* v_a_197_, lean_object* v_inst_198_){
_start:
{
lean_object* v_res_199_; 
v_res_199_ = lp_mathlib_Set_Ici_instLatticeElem(v_00_u03b1_196_, v_a_197_, v_inst_198_);
lean_dec(v_a_197_);
return v_res_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_instDistribLatticeElem___redArg(lean_object* v_inst_200_){
_start:
{
lean_object* v___x_201_; 
v___x_201_ = lp_mathlib_Set_Iic_instLatticeElem___redArg(v_inst_200_);
return v___x_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_instDistribLatticeElem(lean_object* v_00_u03b1_202_, lean_object* v_a_203_, lean_object* v_inst_204_){
_start:
{
lean_object* v___x_205_; 
v___x_205_ = lp_mathlib_Set_Iic_instLatticeElem___redArg(v_inst_204_);
return v___x_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_instDistribLatticeElem___boxed(lean_object* v_00_u03b1_206_, lean_object* v_a_207_, lean_object* v_inst_208_){
_start:
{
lean_object* v_res_209_; 
v_res_209_ = lp_mathlib_Set_Iic_instDistribLatticeElem(v_00_u03b1_206_, v_a_207_, v_inst_208_);
lean_dec(v_a_207_);
return v_res_209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_instDistribLatticeElem___redArg(lean_object* v_inst_210_){
_start:
{
lean_object* v___x_211_; 
v___x_211_ = lp_mathlib_Set_Ici_instLatticeElem___redArg(v_inst_210_);
return v___x_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_instDistribLatticeElem(lean_object* v_00_u03b1_212_, lean_object* v_a_213_, lean_object* v_inst_214_){
_start:
{
lean_object* v___x_215_; 
v___x_215_ = lp_mathlib_Set_Ici_instLatticeElem___redArg(v_inst_214_);
return v___x_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_instDistribLatticeElem___boxed(lean_object* v_00_u03b1_216_, lean_object* v_a_217_, lean_object* v_inst_218_){
_start:
{
lean_object* v_res_219_; 
v_res_219_ = lp_mathlib_Set_Ici_instDistribLatticeElem(v_00_u03b1_216_, v_a_217_, v_inst_218_);
lean_dec(v_a_217_);
return v_res_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_orderTop___redArg(lean_object* v_a_220_){
_start:
{
lean_inc(v_a_220_);
return v_a_220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_orderTop___redArg___boxed(lean_object* v_a_221_){
_start:
{
lean_object* v_res_222_; 
v_res_222_ = lp_mathlib_Set_Iic_orderTop___redArg(v_a_221_);
lean_dec(v_a_221_);
return v_res_222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_orderTop(lean_object* v_00_u03b1_223_, lean_object* v_a_224_, lean_object* v_inst_225_){
_start:
{
lean_inc(v_a_224_);
return v_a_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_orderTop___boxed(lean_object* v_00_u03b1_226_, lean_object* v_a_227_, lean_object* v_inst_228_){
_start:
{
lean_object* v_res_229_; 
v_res_229_ = lp_mathlib_Set_Iic_orderTop(v_00_u03b1_226_, v_a_227_, v_inst_228_);
lean_dec_ref(v_inst_228_);
lean_dec(v_a_227_);
return v_res_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_orderBot___redArg(lean_object* v_a_230_){
_start:
{
lean_inc(v_a_230_);
return v_a_230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_orderBot___redArg___boxed(lean_object* v_a_231_){
_start:
{
lean_object* v_res_232_; 
v_res_232_ = lp_mathlib_Set_Ici_orderBot___redArg(v_a_231_);
lean_dec(v_a_231_);
return v_res_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_orderBot(lean_object* v_00_u03b1_233_, lean_object* v_a_234_, lean_object* v_inst_235_){
_start:
{
lean_inc(v_a_234_);
return v_a_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_orderBot___boxed(lean_object* v_00_u03b1_236_, lean_object* v_a_237_, lean_object* v_inst_238_){
_start:
{
lean_object* v_res_239_; 
v_res_239_ = lp_mathlib_Set_Ici_orderBot(v_00_u03b1_236_, v_a_237_, v_inst_238_);
lean_dec_ref(v_inst_238_);
lean_dec(v_a_237_);
return v_res_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_orderBot___redArg(lean_object* v_inst_240_){
_start:
{
lean_inc(v_inst_240_);
return v_inst_240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_orderBot___redArg___boxed(lean_object* v_inst_241_){
_start:
{
lean_object* v_res_242_; 
v_res_242_ = lp_mathlib_Set_Iic_orderBot___redArg(v_inst_241_);
lean_dec(v_inst_241_);
return v_res_242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_orderBot(lean_object* v_00_u03b1_243_, lean_object* v_a_244_, lean_object* v_inst_245_, lean_object* v_inst_246_){
_start:
{
lean_inc(v_inst_246_);
return v_inst_246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_orderBot___boxed(lean_object* v_00_u03b1_247_, lean_object* v_a_248_, lean_object* v_inst_249_, lean_object* v_inst_250_){
_start:
{
lean_object* v_res_251_; 
v_res_251_ = lp_mathlib_Set_Iic_orderBot(v_00_u03b1_247_, v_a_248_, v_inst_249_, v_inst_250_);
lean_dec(v_inst_250_);
lean_dec_ref(v_inst_249_);
lean_dec(v_a_248_);
return v_res_251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_orderTop___redArg(lean_object* v_inst_252_){
_start:
{
lean_inc(v_inst_252_);
return v_inst_252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_orderTop___redArg___boxed(lean_object* v_inst_253_){
_start:
{
lean_object* v_res_254_; 
v_res_254_ = lp_mathlib_Set_Ici_orderTop___redArg(v_inst_253_);
lean_dec(v_inst_253_);
return v_res_254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_orderTop(lean_object* v_00_u03b1_255_, lean_object* v_a_256_, lean_object* v_inst_257_, lean_object* v_inst_258_){
_start:
{
lean_inc(v_inst_258_);
return v_inst_258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_orderTop___boxed(lean_object* v_00_u03b1_259_, lean_object* v_a_260_, lean_object* v_inst_261_, lean_object* v_inst_262_){
_start:
{
lean_object* v_res_263_; 
v_res_263_ = lp_mathlib_Set_Ici_orderTop(v_00_u03b1_259_, v_a_260_, v_inst_261_, v_inst_262_);
lean_dec(v_inst_262_);
lean_dec_ref(v_inst_261_);
lean_dec(v_a_260_);
return v_res_263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_instBoundedOrderElemOfOrderBot___redArg(lean_object* v_a_264_, lean_object* v_inst_265_){
_start:
{
lean_object* v___x_266_; 
v___x_266_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_266_, 0, v_a_264_);
lean_ctor_set(v___x_266_, 1, v_inst_265_);
return v___x_266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_instBoundedOrderElemOfOrderBot(lean_object* v_00_u03b1_267_, lean_object* v_a_268_, lean_object* v_inst_269_, lean_object* v_inst_270_){
_start:
{
lean_object* v___x_271_; 
v___x_271_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_271_, 0, v_a_268_);
lean_ctor_set(v___x_271_, 1, v_inst_270_);
return v___x_271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_instBoundedOrderElemOfOrderBot___boxed(lean_object* v_00_u03b1_272_, lean_object* v_a_273_, lean_object* v_inst_274_, lean_object* v_inst_275_){
_start:
{
lean_object* v_res_276_; 
v_res_276_ = lp_mathlib_Set_Iic_instBoundedOrderElemOfOrderBot(v_00_u03b1_272_, v_a_273_, v_inst_274_, v_inst_275_);
lean_dec_ref(v_inst_274_);
return v_res_276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_instBoundedOrderElemOfOrderTop___redArg(lean_object* v_a_277_, lean_object* v_inst_278_){
_start:
{
lean_object* v___x_279_; 
v___x_279_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_279_, 0, v_inst_278_);
lean_ctor_set(v___x_279_, 1, v_a_277_);
return v___x_279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_instBoundedOrderElemOfOrderTop(lean_object* v_00_u03b1_280_, lean_object* v_a_281_, lean_object* v_inst_282_, lean_object* v_inst_283_){
_start:
{
lean_object* v___x_284_; 
v___x_284_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_284_, 0, v_inst_283_);
lean_ctor_set(v___x_284_, 1, v_a_281_);
return v___x_284_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Ici_instBoundedOrderElemOfOrderTop___boxed(lean_object* v_00_u03b1_285_, lean_object* v_a_286_, lean_object* v_inst_287_, lean_object* v_inst_288_){
_start:
{
lean_object* v_res_289_; 
v_res_289_ = lp_mathlib_Set_Ici_instBoundedOrderElemOfOrderTop(v_00_u03b1_285_, v_a_286_, v_inst_287_, v_inst_288_);
lean_dec_ref(v_inst_287_);
return v_res_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_semilatticeInf___redArg(lean_object* v_inst_290_){
_start:
{
lean_object* v___x_291_; 
v___x_291_ = lp_mathlib_Subtype_semilatticeInf___redArg(v_inst_290_);
return v___x_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_semilatticeInf(lean_object* v_00_u03b1_292_, lean_object* v_a_293_, lean_object* v_b_294_, lean_object* v_inst_295_){
_start:
{
lean_object* v___x_296_; 
v___x_296_ = lp_mathlib_Subtype_semilatticeInf___redArg(v_inst_295_);
return v___x_296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_semilatticeInf___boxed(lean_object* v_00_u03b1_297_, lean_object* v_a_298_, lean_object* v_b_299_, lean_object* v_inst_300_){
_start:
{
lean_object* v_res_301_; 
v_res_301_ = lp_mathlib_Set_Icc_semilatticeInf(v_00_u03b1_297_, v_a_298_, v_b_299_, v_inst_300_);
lean_dec(v_b_299_);
lean_dec(v_a_298_);
return v_res_301_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_semilatticeSup___redArg(lean_object* v_inst_302_){
_start:
{
lean_object* v_toPartialOrder_303_; lean_object* v_sup_304_; lean_object* v___x_306_; uint8_t v_isShared_307_; uint8_t v_isSharedCheck_313_; 
v_toPartialOrder_303_ = lean_ctor_get(v_inst_302_, 0);
v_sup_304_ = lean_ctor_get(v_inst_302_, 1);
v_isSharedCheck_313_ = !lean_is_exclusive(v_inst_302_);
if (v_isSharedCheck_313_ == 0)
{
v___x_306_ = v_inst_302_;
v_isShared_307_ = v_isSharedCheck_313_;
goto v_resetjp_305_;
}
else
{
lean_inc(v_sup_304_);
lean_inc(v_toPartialOrder_303_);
lean_dec(v_inst_302_);
v___x_306_ = lean_box(0);
v_isShared_307_ = v_isSharedCheck_313_;
goto v_resetjp_305_;
}
v_resetjp_305_:
{
lean_object* v___f_308_; lean_object* v___x_309_; lean_object* v___x_311_; 
v___f_308_ = lean_alloc_closure((void*)(lp_mathlib_Set_Ioc_semilatticeSup___redArg___lam__0), 3, 1);
lean_closure_set(v___f_308_, 0, v_sup_304_);
v___x_309_ = lp_mathlib_Subtype_preorder(lean_box(0), v_toPartialOrder_303_, lean_box(0));
lean_dec_ref(v_toPartialOrder_303_);
if (v_isShared_307_ == 0)
{
lean_ctor_set(v___x_306_, 1, v___f_308_);
lean_ctor_set(v___x_306_, 0, v___x_309_);
v___x_311_ = v___x_306_;
goto v_reusejp_310_;
}
else
{
lean_object* v_reuseFailAlloc_312_; 
v_reuseFailAlloc_312_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_312_, 0, v___x_309_);
lean_ctor_set(v_reuseFailAlloc_312_, 1, v___f_308_);
v___x_311_ = v_reuseFailAlloc_312_;
goto v_reusejp_310_;
}
v_reusejp_310_:
{
return v___x_311_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_semilatticeSup(lean_object* v_00_u03b1_314_, lean_object* v_a_315_, lean_object* v_b_316_, lean_object* v_inst_317_){
_start:
{
lean_object* v___x_318_; 
v___x_318_ = lp_mathlib_Set_Icc_semilatticeSup___redArg(v_inst_317_);
return v___x_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_semilatticeSup___boxed(lean_object* v_00_u03b1_319_, lean_object* v_a_320_, lean_object* v_b_321_, lean_object* v_inst_322_){
_start:
{
lean_object* v_res_323_; 
v_res_323_ = lp_mathlib_Set_Icc_semilatticeSup(v_00_u03b1_319_, v_a_320_, v_b_321_, v_inst_322_);
lean_dec(v_b_321_);
lean_dec(v_a_320_);
return v_res_323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_lattice___redArg(lean_object* v_inst_324_){
_start:
{
lean_object* v_toSemilatticeSup_325_; lean_object* v_inf_326_; lean_object* v___x_328_; uint8_t v_isShared_329_; uint8_t v_isSharedCheck_335_; 
v_toSemilatticeSup_325_ = lean_ctor_get(v_inst_324_, 0);
v_inf_326_ = lean_ctor_get(v_inst_324_, 1);
v_isSharedCheck_335_ = !lean_is_exclusive(v_inst_324_);
if (v_isSharedCheck_335_ == 0)
{
v___x_328_ = v_inst_324_;
v_isShared_329_ = v_isSharedCheck_335_;
goto v_resetjp_327_;
}
else
{
lean_inc(v_inf_326_);
lean_inc(v_toSemilatticeSup_325_);
lean_dec(v_inst_324_);
v___x_328_ = lean_box(0);
v_isShared_329_ = v_isSharedCheck_335_;
goto v_resetjp_327_;
}
v_resetjp_327_:
{
lean_object* v___f_330_; lean_object* v___x_331_; lean_object* v___x_333_; 
v___f_330_ = lean_alloc_closure((void*)(lp_mathlib_Set_Iic_instLatticeElem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_330_, 0, v_inf_326_);
v___x_331_ = lp_mathlib_Set_Icc_semilatticeSup___redArg(v_toSemilatticeSup_325_);
if (v_isShared_329_ == 0)
{
lean_ctor_set(v___x_328_, 1, v___f_330_);
lean_ctor_set(v___x_328_, 0, v___x_331_);
v___x_333_ = v___x_328_;
goto v_reusejp_332_;
}
else
{
lean_object* v_reuseFailAlloc_334_; 
v_reuseFailAlloc_334_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_334_, 0, v___x_331_);
lean_ctor_set(v_reuseFailAlloc_334_, 1, v___f_330_);
v___x_333_ = v_reuseFailAlloc_334_;
goto v_reusejp_332_;
}
v_reusejp_332_:
{
return v___x_333_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_lattice(lean_object* v_00_u03b1_336_, lean_object* v_a_337_, lean_object* v_b_338_, lean_object* v_inst_339_){
_start:
{
lean_object* v___x_340_; 
v___x_340_ = lp_mathlib_Set_Icc_lattice___redArg(v_inst_339_);
return v___x_340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_lattice___boxed(lean_object* v_00_u03b1_341_, lean_object* v_a_342_, lean_object* v_b_343_, lean_object* v_inst_344_){
_start:
{
lean_object* v_res_345_; 
v_res_345_ = lp_mathlib_Set_Icc_lattice(v_00_u03b1_341_, v_a_342_, v_b_343_, v_inst_344_);
lean_dec(v_b_343_);
lean_dec(v_a_342_);
return v_res_345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_instOrderBotElemOfFactLe___redArg(lean_object* v_a_346_){
_start:
{
lean_inc(v_a_346_);
return v_a_346_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_instOrderBotElemOfFactLe___redArg___boxed(lean_object* v_a_347_){
_start:
{
lean_object* v_res_348_; 
v_res_348_ = lp_mathlib_Set_Icc_instOrderBotElemOfFactLe___redArg(v_a_347_);
lean_dec(v_a_347_);
return v_res_348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_instOrderBotElemOfFactLe(lean_object* v_00_u03b1_349_, lean_object* v_a_350_, lean_object* v_b_351_, lean_object* v_inst_352_, lean_object* v_inst_353_){
_start:
{
lean_inc(v_a_350_);
return v_a_350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_instOrderBotElemOfFactLe___boxed(lean_object* v_00_u03b1_354_, lean_object* v_a_355_, lean_object* v_b_356_, lean_object* v_inst_357_, lean_object* v_inst_358_){
_start:
{
lean_object* v_res_359_; 
v_res_359_ = lp_mathlib_Set_Icc_instOrderBotElemOfFactLe(v_00_u03b1_354_, v_a_355_, v_b_356_, v_inst_357_, v_inst_358_);
lean_dec_ref(v_inst_357_);
lean_dec(v_b_356_);
lean_dec(v_a_355_);
return v_res_359_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_instOrderTopElemOfFactLe___redArg(lean_object* v_a_360_){
_start:
{
lean_inc(v_a_360_);
return v_a_360_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_instOrderTopElemOfFactLe___redArg___boxed(lean_object* v_a_361_){
_start:
{
lean_object* v_res_362_; 
v_res_362_ = lp_mathlib_Set_Icc_instOrderTopElemOfFactLe___redArg(v_a_361_);
lean_dec(v_a_361_);
return v_res_362_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_instOrderTopElemOfFactLe(lean_object* v_00_u03b1_363_, lean_object* v_a_364_, lean_object* v_b_365_, lean_object* v_inst_366_, lean_object* v_inst_367_){
_start:
{
lean_inc(v_a_364_);
return v_a_364_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_instOrderTopElemOfFactLe___boxed(lean_object* v_00_u03b1_368_, lean_object* v_a_369_, lean_object* v_b_370_, lean_object* v_inst_371_, lean_object* v_inst_372_){
_start:
{
lean_object* v_res_373_; 
v_res_373_ = lp_mathlib_Set_Icc_instOrderTopElemOfFactLe(v_00_u03b1_368_, v_a_369_, v_b_370_, v_inst_371_, v_inst_372_);
lean_dec_ref(v_inst_371_);
lean_dec(v_b_370_);
lean_dec(v_a_369_);
return v_res_373_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_instBoundedOrderElemOfFactLe___redArg(lean_object* v_a_374_, lean_object* v_b_375_){
_start:
{
lean_object* v___x_376_; 
v___x_376_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_376_, 0, v_b_375_);
lean_ctor_set(v___x_376_, 1, v_a_374_);
return v___x_376_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_instBoundedOrderElemOfFactLe(lean_object* v_00_u03b1_377_, lean_object* v_a_378_, lean_object* v_b_379_, lean_object* v_inst_380_, lean_object* v_inst_381_){
_start:
{
lean_object* v___x_382_; 
v___x_382_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_382_, 0, v_b_379_);
lean_ctor_set(v___x_382_, 1, v_a_378_);
return v___x_382_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Icc_instBoundedOrderElemOfFactLe___boxed(lean_object* v_00_u03b1_383_, lean_object* v_a_384_, lean_object* v_b_385_, lean_object* v_inst_386_, lean_object* v_inst_387_){
_start:
{
lean_object* v_res_388_; 
v_res_388_ = lp_mathlib_Set_Icc_instBoundedOrderElemOfFactLe(v_00_u03b1_383_, v_a_384_, v_b_385_, v_inst_386_, v_inst_387_);
lean_dec_ref(v_inst_386_);
return v_res_388_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Bounds_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_LatticeIntervals(uint8_t builtin) {
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
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_LatticeIntervals(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Order_Bounds_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_LatticeIntervals(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Order_LatticeIntervals(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_LatticeIntervals(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_LatticeIntervals(builtin);
}
#ifdef __cplusplus
}
#endif
