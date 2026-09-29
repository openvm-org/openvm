// Lean compiler output
// Module: Mathlib.Order.FixedPoints
// Imports: public import Init public meta import Init public import Mathlib.Order.Hom.Order public import Mathlib.Order.BourbakiWitt public import Mathlib.Algebra.Group.End
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
lean_object* lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(lean_object*);
lean_object* lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(lean_object*);
lean_object* lp_mathlib_ChainCompletePartialOrder_instOfCompleteLattice___redArg(lean_object*);
lean_object* lp_mathlib_Subtype_preorder(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_OrderHom_const___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_lfp___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_lfp___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_lfp___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_lfp(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_gfp___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_gfp___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_gfp___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_gfp(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prevFixed___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prevFixed___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prevFixed___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prevFixed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_nextFixed___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_nextFixed___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_nextFixed___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_nextFixed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_fixedPoints_instBoundedOrderElemFixedPointsCoeOrderHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_fixedPoints_instBoundedOrderElemFixedPointsCoeOrderHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_fixedPoints_instSemilatticeSupElemFixedPointsCoeOrderHom___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_fixedPoints_instSemilatticeSupElemFixedPointsCoeOrderHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_fixedPoints_instSemilatticeSupElemFixedPointsCoeOrderHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_fixedPoints_instSemilatticeInfElemFixedPointsCoeOrderHom___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_fixedPoints_instSemilatticeInfElemFixedPointsCoeOrderHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_fixedPoints_instSemilatticeInfElemFixedPointsCoeOrderHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_fixedPoints_completeLattice___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_fixedPoints_completeLattice___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_fixedPoints_completeLattice___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_fixedPoints_completeLattice(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_lfp___redArg___lam__0(lean_object* v_toInfSet_1_, lean_object* v_f_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_apply_1(v_toInfSet_1_, lean_box(0));
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_lfp___redArg___lam__0___boxed(lean_object* v_toInfSet_4_, lean_object* v_f_5_){
_start:
{
lean_object* v_res_6_; 
v_res_6_ = lp_mathlib_OrderHom_lfp___redArg___lam__0(v_toInfSet_4_, v_f_5_);
lean_dec(v_f_5_);
return v_res_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_lfp___redArg(lean_object* v_inst_7_){
_start:
{
lean_object* v___x_8_; lean_object* v_toInfSet_9_; lean_object* v___f_10_; 
v___x_8_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_inst_7_);
v_toInfSet_9_ = lean_ctor_get(v___x_8_, 1);
lean_inc(v_toInfSet_9_);
lean_dec_ref(v___x_8_);
v___f_10_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_lfp___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_10_, 0, v_toInfSet_9_);
return v___f_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_lfp(lean_object* v_00_u03b1_11_, lean_object* v_inst_12_){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = lp_mathlib_OrderHom_lfp___redArg(v_inst_12_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_gfp___redArg___lam__0(lean_object* v_toSupSet_14_, lean_object* v_f_15_){
_start:
{
lean_object* v___x_16_; 
v___x_16_ = lean_apply_1(v_toSupSet_14_, lean_box(0));
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_gfp___redArg___lam__0___boxed(lean_object* v_toSupSet_17_, lean_object* v_f_18_){
_start:
{
lean_object* v_res_19_; 
v_res_19_ = lp_mathlib_OrderHom_gfp___redArg___lam__0(v_toSupSet_17_, v_f_18_);
lean_dec(v_f_18_);
return v_res_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_gfp___redArg(lean_object* v_inst_20_){
_start:
{
lean_object* v___x_21_; lean_object* v_toSupSet_22_; lean_object* v___f_23_; 
v___x_21_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_inst_20_);
v_toSupSet_22_ = lean_ctor_get(v___x_21_, 1);
lean_inc(v_toSupSet_22_);
lean_dec_ref(v___x_21_);
v___f_23_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_gfp___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_23_, 0, v_toSupSet_22_);
return v___f_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_gfp(lean_object* v_00_u03b1_24_, lean_object* v_inst_25_){
_start:
{
lean_object* v___x_26_; 
v___x_26_ = lp_mathlib_OrderHom_gfp___redArg(v_inst_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prevFixed___redArg___lam__0(lean_object* v_toLattice_27_, lean_object* v_x_28_, lean_object* v_f_29_, lean_object* v_a_30_){
_start:
{
lean_object* v_inf_31_; lean_object* v___x_32_; lean_object* v___x_33_; lean_object* v___x_34_; 
v_inf_31_ = lean_ctor_get(v_toLattice_27_, 1);
lean_inc(v_inf_31_);
lean_dec_ref(v_toLattice_27_);
v___x_32_ = lp_mathlib_OrderHom_const___lam__0(v_x_28_, v_a_30_);
v___x_33_ = lean_apply_1(v_f_29_, v_a_30_);
v___x_34_ = lean_apply_2(v_inf_31_, v___x_32_, v___x_33_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prevFixed___redArg___lam__0___boxed(lean_object* v_toLattice_35_, lean_object* v_x_36_, lean_object* v_f_37_, lean_object* v_a_38_){
_start:
{
lean_object* v_res_39_; 
v_res_39_ = lp_mathlib_OrderHom_prevFixed___redArg___lam__0(v_toLattice_35_, v_x_36_, v_f_37_, v_a_38_);
lean_dec(v_x_36_);
return v_res_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prevFixed___redArg(lean_object* v_inst_40_, lean_object* v_f_41_, lean_object* v_x_42_){
_start:
{
lean_object* v_toLattice_43_; lean_object* v___f_44_; lean_object* v___x_59__overap_45_; lean_object* v___x_46_; 
v_toLattice_43_ = lean_ctor_get(v_inst_40_, 0);
lean_inc_ref(v_toLattice_43_);
v___f_44_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_prevFixed___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_44_, 0, v_toLattice_43_);
lean_closure_set(v___f_44_, 1, v_x_42_);
lean_closure_set(v___f_44_, 2, v_f_41_);
v___x_59__overap_45_ = lp_mathlib_OrderHom_gfp___redArg(v_inst_40_);
v___x_46_ = lean_apply_1(v___x_59__overap_45_, v___f_44_);
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prevFixed(lean_object* v_00_u03b1_47_, lean_object* v_inst_48_, lean_object* v_f_49_, lean_object* v_x_50_, lean_object* v_hx_51_){
_start:
{
lean_object* v___x_52_; 
v___x_52_ = lp_mathlib_OrderHom_prevFixed___redArg(v_inst_48_, v_f_49_, v_x_50_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_nextFixed___redArg___lam__0(lean_object* v_toSemilatticeSup_53_, lean_object* v_x_54_, lean_object* v_f_55_, lean_object* v_a_56_){
_start:
{
lean_object* v_sup_57_; lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; 
v_sup_57_ = lean_ctor_get(v_toSemilatticeSup_53_, 1);
lean_inc(v_sup_57_);
lean_dec_ref(v_toSemilatticeSup_53_);
v___x_58_ = lp_mathlib_OrderHom_const___lam__0(v_x_54_, v_a_56_);
v___x_59_ = lean_apply_1(v_f_55_, v_a_56_);
v___x_60_ = lean_apply_2(v_sup_57_, v___x_58_, v___x_59_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_nextFixed___redArg___lam__0___boxed(lean_object* v_toSemilatticeSup_61_, lean_object* v_x_62_, lean_object* v_f_63_, lean_object* v_a_64_){
_start:
{
lean_object* v_res_65_; 
v_res_65_ = lp_mathlib_OrderHom_nextFixed___redArg___lam__0(v_toSemilatticeSup_61_, v_x_62_, v_f_63_, v_a_64_);
lean_dec(v_x_62_);
return v_res_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_nextFixed___redArg(lean_object* v_inst_66_, lean_object* v_f_67_, lean_object* v_x_68_){
_start:
{
lean_object* v_toLattice_69_; lean_object* v_toSemilatticeSup_70_; lean_object* v___f_71_; lean_object* v___x_72__overap_72_; lean_object* v___x_73_; 
v_toLattice_69_ = lean_ctor_get(v_inst_66_, 0);
v_toSemilatticeSup_70_ = lean_ctor_get(v_toLattice_69_, 0);
lean_inc_ref(v_toSemilatticeSup_70_);
v___f_71_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_nextFixed___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_71_, 0, v_toSemilatticeSup_70_);
lean_closure_set(v___f_71_, 1, v_x_68_);
lean_closure_set(v___f_71_, 2, v_f_67_);
v___x_72__overap_72_ = lp_mathlib_OrderHom_lfp___redArg(v_inst_66_);
v___x_73_ = lean_apply_1(v___x_72__overap_72_, v___f_71_);
return v___x_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_nextFixed(lean_object* v_00_u03b1_74_, lean_object* v_inst_75_, lean_object* v_f_76_, lean_object* v_x_77_, lean_object* v_hx_78_){
_start:
{
lean_object* v___x_79_; 
v___x_79_ = lp_mathlib_OrderHom_nextFixed___redArg(v_inst_75_, v_f_76_, v_x_77_);
return v___x_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_fixedPoints_instBoundedOrderElemFixedPointsCoeOrderHom___redArg(lean_object* v_inst_80_, lean_object* v_f_81_){
_start:
{
lean_object* v___x_26__overap_82_; lean_object* v___x_83_; lean_object* v___x_27__overap_84_; lean_object* v___x_85_; lean_object* v___x_86_; 
lean_inc_ref(v_inst_80_);
v___x_26__overap_82_ = lp_mathlib_OrderHom_gfp___redArg(v_inst_80_);
lean_inc(v_f_81_);
v___x_83_ = lean_apply_1(v___x_26__overap_82_, v_f_81_);
v___x_27__overap_84_ = lp_mathlib_OrderHom_lfp___redArg(v_inst_80_);
v___x_85_ = lean_apply_1(v___x_27__overap_84_, v_f_81_);
v___x_86_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_86_, 0, v___x_83_);
lean_ctor_set(v___x_86_, 1, v___x_85_);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_fixedPoints_instBoundedOrderElemFixedPointsCoeOrderHom(lean_object* v_00_u03b1_87_, lean_object* v_inst_88_, lean_object* v_f_89_){
_start:
{
lean_object* v___x_90_; 
v___x_90_ = lp_mathlib_fixedPoints_instBoundedOrderElemFixedPointsCoeOrderHom___redArg(v_inst_88_, v_f_89_);
return v___x_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_fixedPoints_instSemilatticeSupElemFixedPointsCoeOrderHom___redArg___lam__0(lean_object* v_toSemilatticeSup_91_, lean_object* v_inst_92_, lean_object* v_f_93_, lean_object* v_x_94_, lean_object* v_y_95_){
_start:
{
lean_object* v_sup_96_; lean_object* v___x_97_; lean_object* v___x_98_; 
v_sup_96_ = lean_ctor_get(v_toSemilatticeSup_91_, 1);
lean_inc(v_sup_96_);
lean_dec_ref(v_toSemilatticeSup_91_);
v___x_97_ = lean_apply_2(v_sup_96_, v_x_94_, v_y_95_);
v___x_98_ = lp_mathlib_OrderHom_nextFixed___redArg(v_inst_92_, v_f_93_, v___x_97_);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_fixedPoints_instSemilatticeSupElemFixedPointsCoeOrderHom___redArg(lean_object* v_inst_99_, lean_object* v_f_100_){
_start:
{
lean_object* v___x_101_; lean_object* v_toPartialOrder_102_; lean_object* v___x_103_; lean_object* v_toLattice_104_; lean_object* v_toSemilatticeSup_105_; lean_object* v___x_107_; uint8_t v_isShared_108_; uint8_t v_isSharedCheck_113_; 
lean_inc_ref(v_inst_99_);
v___x_101_ = lp_mathlib_ChainCompletePartialOrder_instOfCompleteLattice___redArg(v_inst_99_);
v_toPartialOrder_102_ = lean_ctor_get(v___x_101_, 0);
lean_inc_ref(v_toPartialOrder_102_);
lean_dec_ref(v___x_101_);
v___x_103_ = lp_mathlib_Subtype_preorder(lean_box(0), v_toPartialOrder_102_, lean_box(0));
lean_dec_ref(v_toPartialOrder_102_);
v_toLattice_104_ = lean_ctor_get(v_inst_99_, 0);
lean_inc_ref(v_toLattice_104_);
v_toSemilatticeSup_105_ = lean_ctor_get(v_toLattice_104_, 0);
v_isSharedCheck_113_ = !lean_is_exclusive(v_toLattice_104_);
if (v_isSharedCheck_113_ == 0)
{
lean_object* v_unused_114_; 
v_unused_114_ = lean_ctor_get(v_toLattice_104_, 1);
lean_dec(v_unused_114_);
v___x_107_ = v_toLattice_104_;
v_isShared_108_ = v_isSharedCheck_113_;
goto v_resetjp_106_;
}
else
{
lean_inc(v_toSemilatticeSup_105_);
lean_dec(v_toLattice_104_);
v___x_107_ = lean_box(0);
v_isShared_108_ = v_isSharedCheck_113_;
goto v_resetjp_106_;
}
v_resetjp_106_:
{
lean_object* v___f_109_; lean_object* v___x_111_; 
v___f_109_ = lean_alloc_closure((void*)(lp_mathlib_fixedPoints_instSemilatticeSupElemFixedPointsCoeOrderHom___redArg___lam__0), 5, 3);
lean_closure_set(v___f_109_, 0, v_toSemilatticeSup_105_);
lean_closure_set(v___f_109_, 1, v_inst_99_);
lean_closure_set(v___f_109_, 2, v_f_100_);
if (v_isShared_108_ == 0)
{
lean_ctor_set(v___x_107_, 1, v___f_109_);
lean_ctor_set(v___x_107_, 0, v___x_103_);
v___x_111_ = v___x_107_;
goto v_reusejp_110_;
}
else
{
lean_object* v_reuseFailAlloc_112_; 
v_reuseFailAlloc_112_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_112_, 0, v___x_103_);
lean_ctor_set(v_reuseFailAlloc_112_, 1, v___f_109_);
v___x_111_ = v_reuseFailAlloc_112_;
goto v_reusejp_110_;
}
v_reusejp_110_:
{
return v___x_111_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_fixedPoints_instSemilatticeSupElemFixedPointsCoeOrderHom(lean_object* v_00_u03b1_115_, lean_object* v_inst_116_, lean_object* v_f_117_){
_start:
{
lean_object* v___x_118_; 
v___x_118_ = lp_mathlib_fixedPoints_instSemilatticeSupElemFixedPointsCoeOrderHom___redArg(v_inst_116_, v_f_117_);
return v___x_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_fixedPoints_instSemilatticeInfElemFixedPointsCoeOrderHom___redArg___lam__0(lean_object* v_toLattice_119_, lean_object* v_inst_120_, lean_object* v_f_121_, lean_object* v_x_122_, lean_object* v_y_123_){
_start:
{
lean_object* v_inf_124_; lean_object* v___x_125_; lean_object* v___x_126_; 
v_inf_124_ = lean_ctor_get(v_toLattice_119_, 1);
lean_inc(v_inf_124_);
lean_dec_ref(v_toLattice_119_);
v___x_125_ = lean_apply_2(v_inf_124_, v_x_122_, v_y_123_);
v___x_126_ = lp_mathlib_OrderHom_prevFixed___redArg(v_inst_120_, v_f_121_, v___x_125_);
return v___x_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_fixedPoints_instSemilatticeInfElemFixedPointsCoeOrderHom___redArg(lean_object* v_inst_127_, lean_object* v_f_128_){
_start:
{
lean_object* v___x_129_; lean_object* v_toPartialOrder_130_; lean_object* v___x_132_; uint8_t v_isShared_133_; uint8_t v_isSharedCheck_140_; 
lean_inc_ref(v_inst_127_);
v___x_129_ = lp_mathlib_ChainCompletePartialOrder_instOfCompleteLattice___redArg(v_inst_127_);
v_toPartialOrder_130_ = lean_ctor_get(v___x_129_, 0);
v_isSharedCheck_140_ = !lean_is_exclusive(v___x_129_);
if (v_isSharedCheck_140_ == 0)
{
lean_object* v_unused_141_; 
v_unused_141_ = lean_ctor_get(v___x_129_, 1);
lean_dec(v_unused_141_);
v___x_132_ = v___x_129_;
v_isShared_133_ = v_isSharedCheck_140_;
goto v_resetjp_131_;
}
else
{
lean_inc(v_toPartialOrder_130_);
lean_dec(v___x_129_);
v___x_132_ = lean_box(0);
v_isShared_133_ = v_isSharedCheck_140_;
goto v_resetjp_131_;
}
v_resetjp_131_:
{
lean_object* v___x_134_; lean_object* v_toLattice_135_; lean_object* v___f_136_; lean_object* v___x_138_; 
v___x_134_ = lp_mathlib_Subtype_preorder(lean_box(0), v_toPartialOrder_130_, lean_box(0));
lean_dec_ref(v_toPartialOrder_130_);
v_toLattice_135_ = lean_ctor_get(v_inst_127_, 0);
lean_inc_ref(v_toLattice_135_);
v___f_136_ = lean_alloc_closure((void*)(lp_mathlib_fixedPoints_instSemilatticeInfElemFixedPointsCoeOrderHom___redArg___lam__0), 5, 3);
lean_closure_set(v___f_136_, 0, v_toLattice_135_);
lean_closure_set(v___f_136_, 1, v_inst_127_);
lean_closure_set(v___f_136_, 2, v_f_128_);
if (v_isShared_133_ == 0)
{
lean_ctor_set(v___x_132_, 1, v___f_136_);
lean_ctor_set(v___x_132_, 0, v___x_134_);
v___x_138_ = v___x_132_;
goto v_reusejp_137_;
}
else
{
lean_object* v_reuseFailAlloc_139_; 
v_reuseFailAlloc_139_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_139_, 0, v___x_134_);
lean_ctor_set(v_reuseFailAlloc_139_, 1, v___f_136_);
v___x_138_ = v_reuseFailAlloc_139_;
goto v_reusejp_137_;
}
v_reusejp_137_:
{
return v___x_138_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_fixedPoints_instSemilatticeInfElemFixedPointsCoeOrderHom(lean_object* v_00_u03b1_142_, lean_object* v_inst_143_, lean_object* v_f_144_){
_start:
{
lean_object* v___x_145_; 
v___x_145_ = lp_mathlib_fixedPoints_instSemilatticeInfElemFixedPointsCoeOrderHom___redArg(v_inst_143_, v_f_144_);
return v___x_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_fixedPoints_completeLattice___redArg___lam__1(lean_object* v_toSupSet_146_, lean_object* v_inst_147_, lean_object* v_f_148_, lean_object* v_s_149_){
_start:
{
lean_object* v___x_150_; lean_object* v___x_151_; 
v___x_150_ = lean_apply_1(v_toSupSet_146_, lean_box(0));
v___x_151_ = lp_mathlib_OrderHom_nextFixed___redArg(v_inst_147_, v_f_148_, v___x_150_);
return v___x_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_fixedPoints_completeLattice___redArg___lam__0(lean_object* v_toInfSet_152_, lean_object* v_inst_153_, lean_object* v_f_154_, lean_object* v_s_155_){
_start:
{
lean_object* v___x_156_; lean_object* v___x_157_; 
v___x_156_ = lean_apply_1(v_toInfSet_152_, lean_box(0));
v___x_157_ = lp_mathlib_OrderHom_prevFixed___redArg(v_inst_153_, v_f_154_, v___x_156_);
return v___x_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_fixedPoints_completeLattice___redArg(lean_object* v_inst_158_, lean_object* v_f_159_){
_start:
{
lean_object* v___x_160_; lean_object* v_toLattice_161_; lean_object* v___f_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v_toSupSet_165_; lean_object* v___x_166_; lean_object* v_toInfSet_167_; lean_object* v___f_168_; lean_object* v___f_169_; lean_object* v___x_170_; lean_object* v___x_171_; 
lean_inc_n(v_f_159_, 4);
lean_inc_ref_n(v_inst_158_, 6);
v___x_160_ = lp_mathlib_fixedPoints_instSemilatticeSupElemFixedPointsCoeOrderHom___redArg(v_inst_158_, v_f_159_);
v_toLattice_161_ = lean_ctor_get(v_inst_158_, 0);
lean_inc_ref(v_toLattice_161_);
v___f_162_ = lean_alloc_closure((void*)(lp_mathlib_fixedPoints_instSemilatticeInfElemFixedPointsCoeOrderHom___redArg___lam__0), 5, 3);
lean_closure_set(v___f_162_, 0, v_toLattice_161_);
lean_closure_set(v___f_162_, 1, v_inst_158_);
lean_closure_set(v___f_162_, 2, v_f_159_);
v___x_163_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_163_, 0, v___x_160_);
lean_ctor_set(v___x_163_, 1, v___f_162_);
v___x_164_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_inst_158_);
v_toSupSet_165_ = lean_ctor_get(v___x_164_, 1);
lean_inc(v_toSupSet_165_);
lean_dec_ref(v___x_164_);
v___x_166_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_inst_158_);
v_toInfSet_167_ = lean_ctor_get(v___x_166_, 1);
lean_inc(v_toInfSet_167_);
lean_dec_ref(v___x_166_);
v___f_168_ = lean_alloc_closure((void*)(lp_mathlib_fixedPoints_completeLattice___redArg___lam__1), 4, 3);
lean_closure_set(v___f_168_, 0, v_toSupSet_165_);
lean_closure_set(v___f_168_, 1, v_inst_158_);
lean_closure_set(v___f_168_, 2, v_f_159_);
v___f_169_ = lean_alloc_closure((void*)(lp_mathlib_fixedPoints_completeLattice___redArg___lam__0), 4, 3);
lean_closure_set(v___f_169_, 0, v_toInfSet_167_);
lean_closure_set(v___f_169_, 1, v_inst_158_);
lean_closure_set(v___f_169_, 2, v_f_159_);
v___x_170_ = lp_mathlib_fixedPoints_instBoundedOrderElemFixedPointsCoeOrderHom___redArg(v_inst_158_, v_f_159_);
v___x_171_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_171_, 0, v___x_163_);
lean_ctor_set(v___x_171_, 1, v___f_168_);
lean_ctor_set(v___x_171_, 2, v___f_169_);
lean_ctor_set(v___x_171_, 3, v___x_170_);
return v___x_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_fixedPoints_completeLattice(lean_object* v_00_u03b1_172_, lean_object* v_inst_173_, lean_object* v_f_174_){
_start:
{
lean_object* v___x_175_; 
v___x_175_ = lp_mathlib_fixedPoints_completeLattice___redArg(v_inst_173_, v_f_174_);
return v___x_175_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Hom_Order(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_BourbakiWitt(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_End(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_FixedPoints(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Hom_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_BourbakiWitt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_FixedPoints(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Order_Hom_Order(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_BourbakiWitt(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_End(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_FixedPoints(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Hom_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_BourbakiWitt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_FixedPoints(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_FixedPoints(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_FixedPoints(builtin);
}
#ifdef __cplusplus
}
#endif
