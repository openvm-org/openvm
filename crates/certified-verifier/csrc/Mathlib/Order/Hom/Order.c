// Lean compiler output
// Module: Mathlib.Order.Hom.Order
// Imports: public import Init public meta import Init public import Mathlib.Logic.Function.Iterate public import Mathlib.Order.GaloisConnection.Basic public import Mathlib.Order.Hom.Basic
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
lean_object* lp_mathlib_OrderHom_instPartialOrder(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_OrderHom_const___lam__0___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(lean_object*);
lean_object* lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instMax___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instMax___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instMax(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instMax___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instSemilatticeSup___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instSemilatticeSup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instSemilatticeSup___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instSemilatticeSup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instSemilatticeSup___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instMin___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instMin___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instMin(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instMin___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instSemilatticeInf___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instSemilatticeInf___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instSemilatticeInf___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instSemilatticeInf(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instSemilatticeInf___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_lattice___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_lattice___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_lattice(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_lattice___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instBotOfOrderBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instBotOfOrderBot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instBotOfOrderBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_orderBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_orderBot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_orderBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instTopOrderHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instTopOrderHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instTopOrderHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_orderTop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_orderTop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_orderTop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instInfSet___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instInfSet___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instInfSet___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instInfSet(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instInfSet___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instSupSet___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instSupSet___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instSupSet___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instSupSet(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instSupSet___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instCompleteLattice___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instCompleteLattice___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instCompleteLattice(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instCompleteLattice___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instMax___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_f_2_, lean_object* v_g_3_, lean_object* v___y_4_){
_start:
{
lean_object* v_sup_5_; lean_object* v___x_6_; lean_object* v___x_7_; lean_object* v___x_8_; 
v_sup_5_ = lean_ctor_get(v_inst_1_, 1);
lean_inc(v_sup_5_);
lean_dec_ref(v_inst_1_);
lean_inc(v___y_4_);
v___x_6_ = lean_apply_1(v_f_2_, v___y_4_);
v___x_7_ = lean_apply_1(v_g_3_, v___y_4_);
v___x_8_ = lean_apply_2(v_sup_5_, v___x_6_, v___x_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instMax___redArg(lean_object* v_inst_9_){
_start:
{
lean_object* v___f_10_; 
v___f_10_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_instMax___redArg___lam__0), 4, 1);
lean_closure_set(v___f_10_, 0, v_inst_9_);
return v___f_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instMax(lean_object* v_00_u03b1_11_, lean_object* v_00_u03b2_12_, lean_object* v_inst_13_, lean_object* v_inst_14_){
_start:
{
lean_object* v___f_15_; 
v___f_15_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_instMax___redArg___lam__0), 4, 1);
lean_closure_set(v___f_15_, 0, v_inst_14_);
return v___f_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instMax___boxed(lean_object* v_00_u03b1_16_, lean_object* v_00_u03b2_17_, lean_object* v_inst_18_, lean_object* v_inst_19_){
_start:
{
lean_object* v_res_20_; 
v_res_20_ = lp_mathlib_OrderHom_instMax(v_00_u03b1_16_, v_00_u03b2_17_, v_inst_18_, v_inst_19_);
lean_dec_ref(v_inst_18_);
return v_res_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instSemilatticeSup___redArg___lam__0(lean_object* v_sup_21_, lean_object* v_f_22_, lean_object* v_g_23_, lean_object* v___y_24_){
_start:
{
lean_object* v___x_25_; lean_object* v___x_26_; lean_object* v___x_27_; 
lean_inc(v___y_24_);
v___x_25_ = lean_apply_1(v_f_22_, v___y_24_);
v___x_26_ = lean_apply_1(v_g_23_, v___y_24_);
v___x_27_ = lean_apply_2(v_sup_21_, v___x_25_, v___x_26_);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instSemilatticeSup___redArg(lean_object* v_inst_28_, lean_object* v_inst_29_){
_start:
{
lean_object* v_toPartialOrder_30_; lean_object* v_sup_31_; lean_object* v___x_33_; uint8_t v_isShared_34_; uint8_t v_isSharedCheck_40_; 
v_toPartialOrder_30_ = lean_ctor_get(v_inst_29_, 0);
v_sup_31_ = lean_ctor_get(v_inst_29_, 1);
v_isSharedCheck_40_ = !lean_is_exclusive(v_inst_29_);
if (v_isSharedCheck_40_ == 0)
{
v___x_33_ = v_inst_29_;
v_isShared_34_ = v_isSharedCheck_40_;
goto v_resetjp_32_;
}
else
{
lean_inc(v_sup_31_);
lean_inc(v_toPartialOrder_30_);
lean_dec(v_inst_29_);
v___x_33_ = lean_box(0);
v_isShared_34_ = v_isSharedCheck_40_;
goto v_resetjp_32_;
}
v_resetjp_32_:
{
lean_object* v___f_35_; lean_object* v___x_36_; lean_object* v___x_38_; 
v___f_35_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_instSemilatticeSup___redArg___lam__0), 4, 1);
lean_closure_set(v___f_35_, 0, v_sup_31_);
v___x_36_ = lp_mathlib_OrderHom_instPartialOrder(lean_box(0), v_inst_28_, lean_box(0), v_toPartialOrder_30_);
lean_dec_ref(v_toPartialOrder_30_);
if (v_isShared_34_ == 0)
{
lean_ctor_set(v___x_33_, 1, v___f_35_);
lean_ctor_set(v___x_33_, 0, v___x_36_);
v___x_38_ = v___x_33_;
goto v_reusejp_37_;
}
else
{
lean_object* v_reuseFailAlloc_39_; 
v_reuseFailAlloc_39_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_39_, 0, v___x_36_);
lean_ctor_set(v_reuseFailAlloc_39_, 1, v___f_35_);
v___x_38_ = v_reuseFailAlloc_39_;
goto v_reusejp_37_;
}
v_reusejp_37_:
{
return v___x_38_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instSemilatticeSup___redArg___boxed(lean_object* v_inst_41_, lean_object* v_inst_42_){
_start:
{
lean_object* v_res_43_; 
v_res_43_ = lp_mathlib_OrderHom_instSemilatticeSup___redArg(v_inst_41_, v_inst_42_);
lean_dec_ref(v_inst_41_);
return v_res_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instSemilatticeSup(lean_object* v_00_u03b1_44_, lean_object* v_00_u03b2_45_, lean_object* v_inst_46_, lean_object* v_inst_47_){
_start:
{
lean_object* v___x_48_; 
v___x_48_ = lp_mathlib_OrderHom_instSemilatticeSup___redArg(v_inst_46_, v_inst_47_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instSemilatticeSup___boxed(lean_object* v_00_u03b1_49_, lean_object* v_00_u03b2_50_, lean_object* v_inst_51_, lean_object* v_inst_52_){
_start:
{
lean_object* v_res_53_; 
v_res_53_ = lp_mathlib_OrderHom_instSemilatticeSup(v_00_u03b1_49_, v_00_u03b2_50_, v_inst_51_, v_inst_52_);
lean_dec_ref(v_inst_51_);
return v_res_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instMin___redArg___lam__0(lean_object* v_inst_54_, lean_object* v_f_55_, lean_object* v_g_56_, lean_object* v___y_57_){
_start:
{
lean_object* v_inf_58_; lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; 
v_inf_58_ = lean_ctor_get(v_inst_54_, 1);
lean_inc(v_inf_58_);
lean_dec_ref(v_inst_54_);
lean_inc(v___y_57_);
v___x_59_ = lean_apply_1(v_f_55_, v___y_57_);
v___x_60_ = lean_apply_1(v_g_56_, v___y_57_);
v___x_61_ = lean_apply_2(v_inf_58_, v___x_59_, v___x_60_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instMin___redArg(lean_object* v_inst_62_){
_start:
{
lean_object* v___f_63_; 
v___f_63_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_instMin___redArg___lam__0), 4, 1);
lean_closure_set(v___f_63_, 0, v_inst_62_);
return v___f_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instMin(lean_object* v_00_u03b1_64_, lean_object* v_00_u03b2_65_, lean_object* v_inst_66_, lean_object* v_inst_67_){
_start:
{
lean_object* v___f_68_; 
v___f_68_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_instMin___redArg___lam__0), 4, 1);
lean_closure_set(v___f_68_, 0, v_inst_67_);
return v___f_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instMin___boxed(lean_object* v_00_u03b1_69_, lean_object* v_00_u03b2_70_, lean_object* v_inst_71_, lean_object* v_inst_72_){
_start:
{
lean_object* v_res_73_; 
v_res_73_ = lp_mathlib_OrderHom_instMin(v_00_u03b1_69_, v_00_u03b2_70_, v_inst_71_, v_inst_72_);
lean_dec_ref(v_inst_71_);
return v_res_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instSemilatticeInf___redArg___lam__0(lean_object* v_inf_74_, lean_object* v_x1_75_, lean_object* v_x2_76_, lean_object* v___y_77_){
_start:
{
lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; 
lean_inc(v___y_77_);
v___x_78_ = lean_apply_1(v_x1_75_, v___y_77_);
v___x_79_ = lean_apply_1(v_x2_76_, v___y_77_);
v___x_80_ = lean_apply_2(v_inf_74_, v___x_78_, v___x_79_);
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instSemilatticeInf___redArg(lean_object* v_inst_81_, lean_object* v_inst_82_){
_start:
{
lean_object* v_toPartialOrder_83_; lean_object* v_inf_84_; lean_object* v___x_86_; uint8_t v_isShared_87_; uint8_t v_isSharedCheck_93_; 
v_toPartialOrder_83_ = lean_ctor_get(v_inst_82_, 0);
v_inf_84_ = lean_ctor_get(v_inst_82_, 1);
v_isSharedCheck_93_ = !lean_is_exclusive(v_inst_82_);
if (v_isSharedCheck_93_ == 0)
{
v___x_86_ = v_inst_82_;
v_isShared_87_ = v_isSharedCheck_93_;
goto v_resetjp_85_;
}
else
{
lean_inc(v_inf_84_);
lean_inc(v_toPartialOrder_83_);
lean_dec(v_inst_82_);
v___x_86_ = lean_box(0);
v_isShared_87_ = v_isSharedCheck_93_;
goto v_resetjp_85_;
}
v_resetjp_85_:
{
lean_object* v___f_88_; lean_object* v___x_89_; lean_object* v___x_91_; 
v___f_88_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_instSemilatticeInf___redArg___lam__0), 4, 1);
lean_closure_set(v___f_88_, 0, v_inf_84_);
v___x_89_ = lp_mathlib_OrderHom_instPartialOrder(lean_box(0), v_inst_81_, lean_box(0), v_toPartialOrder_83_);
lean_dec_ref(v_toPartialOrder_83_);
if (v_isShared_87_ == 0)
{
lean_ctor_set(v___x_86_, 1, v___f_88_);
lean_ctor_set(v___x_86_, 0, v___x_89_);
v___x_91_ = v___x_86_;
goto v_reusejp_90_;
}
else
{
lean_object* v_reuseFailAlloc_92_; 
v_reuseFailAlloc_92_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_92_, 0, v___x_89_);
lean_ctor_set(v_reuseFailAlloc_92_, 1, v___f_88_);
v___x_91_ = v_reuseFailAlloc_92_;
goto v_reusejp_90_;
}
v_reusejp_90_:
{
return v___x_91_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instSemilatticeInf___redArg___boxed(lean_object* v_inst_94_, lean_object* v_inst_95_){
_start:
{
lean_object* v_res_96_; 
v_res_96_ = lp_mathlib_OrderHom_instSemilatticeInf___redArg(v_inst_94_, v_inst_95_);
lean_dec_ref(v_inst_94_);
return v_res_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instSemilatticeInf(lean_object* v_00_u03b1_97_, lean_object* v_00_u03b2_98_, lean_object* v_inst_99_, lean_object* v_inst_100_){
_start:
{
lean_object* v___x_101_; 
v___x_101_ = lp_mathlib_OrderHom_instSemilatticeInf___redArg(v_inst_99_, v_inst_100_);
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instSemilatticeInf___boxed(lean_object* v_00_u03b1_102_, lean_object* v_00_u03b2_103_, lean_object* v_inst_104_, lean_object* v_inst_105_){
_start:
{
lean_object* v_res_106_; 
v_res_106_ = lp_mathlib_OrderHom_instSemilatticeInf(v_00_u03b1_102_, v_00_u03b2_103_, v_inst_104_, v_inst_105_);
lean_dec_ref(v_inst_104_);
return v_res_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_lattice___redArg(lean_object* v_inst_107_, lean_object* v_inst_108_){
_start:
{
lean_object* v_toSemilatticeSup_109_; lean_object* v_inf_110_; lean_object* v___x_112_; uint8_t v_isShared_113_; uint8_t v_isSharedCheck_119_; 
v_toSemilatticeSup_109_ = lean_ctor_get(v_inst_108_, 0);
v_inf_110_ = lean_ctor_get(v_inst_108_, 1);
v_isSharedCheck_119_ = !lean_is_exclusive(v_inst_108_);
if (v_isSharedCheck_119_ == 0)
{
v___x_112_ = v_inst_108_;
v_isShared_113_ = v_isSharedCheck_119_;
goto v_resetjp_111_;
}
else
{
lean_inc(v_inf_110_);
lean_inc(v_toSemilatticeSup_109_);
lean_dec(v_inst_108_);
v___x_112_ = lean_box(0);
v_isShared_113_ = v_isSharedCheck_119_;
goto v_resetjp_111_;
}
v_resetjp_111_:
{
lean_object* v___f_114_; lean_object* v___x_115_; lean_object* v___x_117_; 
v___f_114_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_instSemilatticeInf___redArg___lam__0), 4, 1);
lean_closure_set(v___f_114_, 0, v_inf_110_);
v___x_115_ = lp_mathlib_OrderHom_instSemilatticeSup___redArg(v_inst_107_, v_toSemilatticeSup_109_);
if (v_isShared_113_ == 0)
{
lean_ctor_set(v___x_112_, 1, v___f_114_);
lean_ctor_set(v___x_112_, 0, v___x_115_);
v___x_117_ = v___x_112_;
goto v_reusejp_116_;
}
else
{
lean_object* v_reuseFailAlloc_118_; 
v_reuseFailAlloc_118_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_118_, 0, v___x_115_);
lean_ctor_set(v_reuseFailAlloc_118_, 1, v___f_114_);
v___x_117_ = v_reuseFailAlloc_118_;
goto v_reusejp_116_;
}
v_reusejp_116_:
{
return v___x_117_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_lattice___redArg___boxed(lean_object* v_inst_120_, lean_object* v_inst_121_){
_start:
{
lean_object* v_res_122_; 
v_res_122_ = lp_mathlib_OrderHom_lattice___redArg(v_inst_120_, v_inst_121_);
lean_dec_ref(v_inst_120_);
return v_res_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_lattice(lean_object* v_00_u03b1_123_, lean_object* v_00_u03b2_124_, lean_object* v_inst_125_, lean_object* v_inst_126_){
_start:
{
lean_object* v___x_127_; 
v___x_127_ = lp_mathlib_OrderHom_lattice___redArg(v_inst_125_, v_inst_126_);
return v___x_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_lattice___boxed(lean_object* v_00_u03b1_128_, lean_object* v_00_u03b2_129_, lean_object* v_inst_130_, lean_object* v_inst_131_){
_start:
{
lean_object* v_res_132_; 
v_res_132_ = lp_mathlib_OrderHom_lattice(v_00_u03b1_128_, v_00_u03b2_129_, v_inst_130_, v_inst_131_);
lean_dec_ref(v_inst_130_);
return v_res_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instBotOfOrderBot___redArg(lean_object* v_inst_133_){
_start:
{
lean_object* v___x_134_; 
v___x_134_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_const___lam__0___boxed), 2, 1);
lean_closure_set(v___x_134_, 0, v_inst_133_);
return v___x_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instBotOfOrderBot(lean_object* v_00_u03b1_135_, lean_object* v_00_u03b2_136_, lean_object* v_inst_137_, lean_object* v_inst_138_, lean_object* v_inst_139_){
_start:
{
lean_object* v___x_140_; 
v___x_140_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_const___lam__0___boxed), 2, 1);
lean_closure_set(v___x_140_, 0, v_inst_139_);
return v___x_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instBotOfOrderBot___boxed(lean_object* v_00_u03b1_141_, lean_object* v_00_u03b2_142_, lean_object* v_inst_143_, lean_object* v_inst_144_, lean_object* v_inst_145_){
_start:
{
lean_object* v_res_146_; 
v_res_146_ = lp_mathlib_OrderHom_instBotOfOrderBot(v_00_u03b1_141_, v_00_u03b2_142_, v_inst_143_, v_inst_144_, v_inst_145_);
lean_dec_ref(v_inst_144_);
lean_dec_ref(v_inst_143_);
return v_res_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_orderBot___redArg(lean_object* v_inst_147_){
_start:
{
lean_object* v___x_148_; 
v___x_148_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_const___lam__0___boxed), 2, 1);
lean_closure_set(v___x_148_, 0, v_inst_147_);
return v___x_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_orderBot(lean_object* v_00_u03b1_149_, lean_object* v_00_u03b2_150_, lean_object* v_inst_151_, lean_object* v_inst_152_, lean_object* v_inst_153_){
_start:
{
lean_object* v___x_154_; 
v___x_154_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_const___lam__0___boxed), 2, 1);
lean_closure_set(v___x_154_, 0, v_inst_153_);
return v___x_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_orderBot___boxed(lean_object* v_00_u03b1_155_, lean_object* v_00_u03b2_156_, lean_object* v_inst_157_, lean_object* v_inst_158_, lean_object* v_inst_159_){
_start:
{
lean_object* v_res_160_; 
v_res_160_ = lp_mathlib_OrderHom_orderBot(v_00_u03b1_155_, v_00_u03b2_156_, v_inst_157_, v_inst_158_, v_inst_159_);
lean_dec_ref(v_inst_158_);
lean_dec_ref(v_inst_157_);
return v_res_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instTopOrderHom___redArg(lean_object* v_inst_161_){
_start:
{
lean_object* v___x_162_; 
v___x_162_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_const___lam__0___boxed), 2, 1);
lean_closure_set(v___x_162_, 0, v_inst_161_);
return v___x_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instTopOrderHom(lean_object* v_00_u03b1_163_, lean_object* v_00_u03b2_164_, lean_object* v_inst_165_, lean_object* v_inst_166_, lean_object* v_inst_167_){
_start:
{
lean_object* v___x_168_; 
v___x_168_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_const___lam__0___boxed), 2, 1);
lean_closure_set(v___x_168_, 0, v_inst_167_);
return v___x_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instTopOrderHom___boxed(lean_object* v_00_u03b1_169_, lean_object* v_00_u03b2_170_, lean_object* v_inst_171_, lean_object* v_inst_172_, lean_object* v_inst_173_){
_start:
{
lean_object* v_res_174_; 
v_res_174_ = lp_mathlib_OrderHom_instTopOrderHom(v_00_u03b1_169_, v_00_u03b2_170_, v_inst_171_, v_inst_172_, v_inst_173_);
lean_dec_ref(v_inst_172_);
lean_dec_ref(v_inst_171_);
return v_res_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_orderTop___redArg(lean_object* v_inst_175_){
_start:
{
lean_object* v___x_176_; 
v___x_176_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_const___lam__0___boxed), 2, 1);
lean_closure_set(v___x_176_, 0, v_inst_175_);
return v___x_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_orderTop(lean_object* v_00_u03b1_177_, lean_object* v_00_u03b2_178_, lean_object* v_inst_179_, lean_object* v_inst_180_, lean_object* v_inst_181_){
_start:
{
lean_object* v___x_182_; 
v___x_182_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_const___lam__0___boxed), 2, 1);
lean_closure_set(v___x_182_, 0, v_inst_181_);
return v___x_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_orderTop___boxed(lean_object* v_00_u03b1_183_, lean_object* v_00_u03b2_184_, lean_object* v_inst_185_, lean_object* v_inst_186_, lean_object* v_inst_187_){
_start:
{
lean_object* v_res_188_; 
v_res_188_ = lp_mathlib_OrderHom_orderTop(v_00_u03b1_183_, v_00_u03b2_184_, v_inst_185_, v_inst_186_, v_inst_187_);
lean_dec_ref(v_inst_186_);
lean_dec_ref(v_inst_185_);
return v_res_188_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instInfSet___redArg___lam__0(lean_object* v_toInfSet_189_, lean_object* v_s_190_, lean_object* v___y_191_){
_start:
{
lean_object* v___x_192_; 
v___x_192_ = lean_apply_1(v_toInfSet_189_, lean_box(0));
return v___x_192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instInfSet___redArg___lam__0___boxed(lean_object* v_toInfSet_193_, lean_object* v_s_194_, lean_object* v___y_195_){
_start:
{
lean_object* v_res_196_; 
v_res_196_ = lp_mathlib_OrderHom_instInfSet___redArg___lam__0(v_toInfSet_193_, v_s_194_, v___y_195_);
lean_dec(v___y_195_);
return v_res_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instInfSet___redArg(lean_object* v_inst_197_){
_start:
{
lean_object* v___x_198_; lean_object* v_toInfSet_199_; lean_object* v___f_200_; 
v___x_198_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_inst_197_);
v_toInfSet_199_ = lean_ctor_get(v___x_198_, 1);
lean_inc(v_toInfSet_199_);
lean_dec_ref(v___x_198_);
v___f_200_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_instInfSet___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_200_, 0, v_toInfSet_199_);
return v___f_200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instInfSet(lean_object* v_00_u03b1_201_, lean_object* v_00_u03b2_202_, lean_object* v_inst_203_, lean_object* v_inst_204_){
_start:
{
lean_object* v___x_205_; 
v___x_205_ = lp_mathlib_OrderHom_instInfSet___redArg(v_inst_204_);
return v___x_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instInfSet___boxed(lean_object* v_00_u03b1_206_, lean_object* v_00_u03b2_207_, lean_object* v_inst_208_, lean_object* v_inst_209_){
_start:
{
lean_object* v_res_210_; 
v_res_210_ = lp_mathlib_OrderHom_instInfSet(v_00_u03b1_206_, v_00_u03b2_207_, v_inst_208_, v_inst_209_);
lean_dec_ref(v_inst_208_);
return v_res_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instSupSet___redArg___lam__0(lean_object* v_toSupSet_211_, lean_object* v_s_212_, lean_object* v___y_213_){
_start:
{
lean_object* v___x_214_; 
v___x_214_ = lean_apply_1(v_toSupSet_211_, lean_box(0));
return v___x_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instSupSet___redArg___lam__0___boxed(lean_object* v_toSupSet_215_, lean_object* v_s_216_, lean_object* v___y_217_){
_start:
{
lean_object* v_res_218_; 
v_res_218_ = lp_mathlib_OrderHom_instSupSet___redArg___lam__0(v_toSupSet_215_, v_s_216_, v___y_217_);
lean_dec(v___y_217_);
return v_res_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instSupSet___redArg(lean_object* v_inst_219_){
_start:
{
lean_object* v___x_220_; lean_object* v_toSupSet_221_; lean_object* v___f_222_; 
v___x_220_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_inst_219_);
v_toSupSet_221_ = lean_ctor_get(v___x_220_, 1);
lean_inc(v_toSupSet_221_);
lean_dec_ref(v___x_220_);
v___f_222_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_instSupSet___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_222_, 0, v_toSupSet_221_);
return v___f_222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instSupSet(lean_object* v_00_u03b1_223_, lean_object* v_00_u03b2_224_, lean_object* v_inst_225_, lean_object* v_inst_226_){
_start:
{
lean_object* v___x_227_; 
v___x_227_ = lp_mathlib_OrderHom_instSupSet___redArg(v_inst_226_);
return v___x_227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instSupSet___boxed(lean_object* v_00_u03b1_228_, lean_object* v_00_u03b2_229_, lean_object* v_inst_230_, lean_object* v_inst_231_){
_start:
{
lean_object* v_res_232_; 
v_res_232_ = lp_mathlib_OrderHom_instSupSet(v_00_u03b1_228_, v_00_u03b2_229_, v_inst_230_, v_inst_231_);
lean_dec_ref(v_inst_230_);
return v_res_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instCompleteLattice___redArg(lean_object* v_inst_233_, lean_object* v_inst_234_){
_start:
{
lean_object* v_toLattice_235_; lean_object* v_toBoundedOrder_236_; lean_object* v___x_237_; lean_object* v_toOrderTop_238_; lean_object* v_toOrderBot_239_; lean_object* v___x_241_; uint8_t v_isShared_242_; uint8_t v_isSharedCheck_251_; 
v_toLattice_235_ = lean_ctor_get(v_inst_234_, 0);
v_toBoundedOrder_236_ = lean_ctor_get(v_inst_234_, 3);
lean_inc_ref(v_toBoundedOrder_236_);
lean_inc_ref(v_toLattice_235_);
v___x_237_ = lp_mathlib_OrderHom_lattice___redArg(v_inst_233_, v_toLattice_235_);
v_toOrderTop_238_ = lean_ctor_get(v_toBoundedOrder_236_, 0);
v_toOrderBot_239_ = lean_ctor_get(v_toBoundedOrder_236_, 1);
v_isSharedCheck_251_ = !lean_is_exclusive(v_toBoundedOrder_236_);
if (v_isSharedCheck_251_ == 0)
{
v___x_241_ = v_toBoundedOrder_236_;
v_isShared_242_ = v_isSharedCheck_251_;
goto v_resetjp_240_;
}
else
{
lean_inc(v_toOrderBot_239_);
lean_inc(v_toOrderTop_238_);
lean_dec(v_toBoundedOrder_236_);
v___x_241_ = lean_box(0);
v_isShared_242_ = v_isSharedCheck_251_;
goto v_resetjp_240_;
}
v_resetjp_240_:
{
lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_248_; 
v___x_243_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_const___lam__0___boxed), 2, 1);
lean_closure_set(v___x_243_, 0, v_toOrderTop_238_);
v___x_244_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_const___lam__0___boxed), 2, 1);
lean_closure_set(v___x_244_, 0, v_toOrderBot_239_);
lean_inc_ref(v_inst_234_);
v___x_245_ = lp_mathlib_OrderHom_instSupSet___redArg(v_inst_234_);
v___x_246_ = lp_mathlib_OrderHom_instInfSet___redArg(v_inst_234_);
if (v_isShared_242_ == 0)
{
lean_ctor_set(v___x_241_, 1, v___x_244_);
lean_ctor_set(v___x_241_, 0, v___x_243_);
v___x_248_ = v___x_241_;
goto v_reusejp_247_;
}
else
{
lean_object* v_reuseFailAlloc_250_; 
v_reuseFailAlloc_250_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_250_, 0, v___x_243_);
lean_ctor_set(v_reuseFailAlloc_250_, 1, v___x_244_);
v___x_248_ = v_reuseFailAlloc_250_;
goto v_reusejp_247_;
}
v_reusejp_247_:
{
lean_object* v___x_249_; 
v___x_249_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_249_, 0, v___x_237_);
lean_ctor_set(v___x_249_, 1, v___x_245_);
lean_ctor_set(v___x_249_, 2, v___x_246_);
lean_ctor_set(v___x_249_, 3, v___x_248_);
return v___x_249_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instCompleteLattice___redArg___boxed(lean_object* v_inst_252_, lean_object* v_inst_253_){
_start:
{
lean_object* v_res_254_; 
v_res_254_ = lp_mathlib_OrderHom_instCompleteLattice___redArg(v_inst_252_, v_inst_253_);
lean_dec_ref(v_inst_252_);
return v_res_254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instCompleteLattice(lean_object* v_00_u03b1_255_, lean_object* v_00_u03b2_256_, lean_object* v_inst_257_, lean_object* v_inst_258_){
_start:
{
lean_object* v___x_259_; 
v___x_259_ = lp_mathlib_OrderHom_instCompleteLattice___redArg(v_inst_257_, v_inst_258_);
return v___x_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instCompleteLattice___boxed(lean_object* v_00_u03b1_260_, lean_object* v_00_u03b2_261_, lean_object* v_inst_262_, lean_object* v_inst_263_){
_start:
{
lean_object* v_res_264_; 
v_res_264_ = lp_mathlib_OrderHom_instCompleteLattice(v_00_u03b1_260_, v_00_u03b2_261_, v_inst_262_, v_inst_263_);
lean_dec_ref(v_inst_262_);
return v_res_264_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Function_Iterate(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_GaloisConnection_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Hom_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Hom_Order(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Function_Iterate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_GaloisConnection_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Hom_Order(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Logic_Function_Iterate(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_GaloisConnection_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Hom_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Hom_Order(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Function_Iterate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_GaloisConnection_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Hom_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Hom_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Hom_Order(builtin);
}
#ifdef __cplusplus
}
#endif
