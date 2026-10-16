// Lean compiler output
// Module: Mathlib.Order.CompleteLattice.Lemmas
// Imports: public import Init public meta import Init public import Mathlib.Data.Bool.Set public import Mathlib.Data.Nat.Set public import Mathlib.Order.CompleteLattice.Basic
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
lean_object* lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(lean_object*);
lean_object* lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(lean_object*);
lean_object* lp_mathlib_PUnit_instLinearOrder___lam__2___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_decidableEqOfDecidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_PUnit_instBooleanAlgebra;
extern lean_object* lp_mathlib_PUnit_instBiheytingAlgebra;
lean_object* lp_mathlib_BiheytingAlgebra_toCoheytingAlgebra___redArg(lean_object*);
lean_object* lp_mathlib_PUnit_instLinearOrder___lam__4___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_decidableLTOfDecidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_supSet___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_supSet___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_supSet(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_infSet___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_infSet(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instCompleteLattice___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instCompleteLattice___redArg___lam__1(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_ULift_instCompleteLattice___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_ULift_instCompleteLattice___redArg___closed__0 = (const lean_object*)&lp_mathlib_ULift_instCompleteLattice___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_ULift_instCompleteLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instCompleteLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instCompleteLinearOrder___lam__0(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_PUnit_instCompleteLinearOrder___lam__1(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instCompleteLinearOrder___lam__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_PUnit_instCompleteLinearOrder___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_PUnit_instCompleteLinearOrder___closed__0;
static const lean_closure_object lp_mathlib_PUnit_instCompleteLinearOrder___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_PUnit_instCompleteLinearOrder___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_PUnit_instCompleteLinearOrder___closed__1 = (const lean_object*)&lp_mathlib_PUnit_instCompleteLinearOrder___closed__1_value;
static const lean_closure_object lp_mathlib_PUnit_instCompleteLinearOrder___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_PUnit_instLinearOrder___lam__2___boxed, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))} };
static const lean_object* lp_mathlib_PUnit_instCompleteLinearOrder___closed__2 = (const lean_object*)&lp_mathlib_PUnit_instCompleteLinearOrder___closed__2_value;
static const lean_closure_object lp_mathlib_PUnit_instCompleteLinearOrder___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_PUnit_instLinearOrder___lam__4___boxed, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_PUnit_instCompleteLinearOrder___closed__2_value)} };
static const lean_object* lp_mathlib_PUnit_instCompleteLinearOrder___closed__3 = (const lean_object*)&lp_mathlib_PUnit_instCompleteLinearOrder___closed__3_value;
static const lean_closure_object lp_mathlib_PUnit_instCompleteLinearOrder___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_PUnit_instCompleteLinearOrder___lam__1___boxed, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))} };
static const lean_object* lp_mathlib_PUnit_instCompleteLinearOrder___closed__4 = (const lean_object*)&lp_mathlib_PUnit_instCompleteLinearOrder___closed__4_value;
static const lean_ctor_object lp_mathlib_PUnit_instCompleteLinearOrder___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_PUnit_instCompleteLinearOrder___closed__5 = (const lean_object*)&lp_mathlib_PUnit_instCompleteLinearOrder___closed__5_value;
static const lean_closure_object lp_mathlib_PUnit_instCompleteLinearOrder___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_decidableEqOfDecidableLE___boxed, .m_arity = 5, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PUnit_instCompleteLinearOrder___closed__5_value),((lean_object*)&lp_mathlib_PUnit_instCompleteLinearOrder___closed__4_value)} };
static const lean_object* lp_mathlib_PUnit_instCompleteLinearOrder___closed__6 = (const lean_object*)&lp_mathlib_PUnit_instCompleteLinearOrder___closed__6_value;
static const lean_closure_object lp_mathlib_PUnit_instCompleteLinearOrder___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_decidableLTOfDecidableLE___boxed, .m_arity = 5, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PUnit_instCompleteLinearOrder___closed__5_value),((lean_object*)&lp_mathlib_PUnit_instCompleteLinearOrder___closed__4_value)} };
static const lean_object* lp_mathlib_PUnit_instCompleteLinearOrder___closed__7 = (const lean_object*)&lp_mathlib_PUnit_instCompleteLinearOrder___closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instCompleteLinearOrder;
LEAN_EXPORT lean_object* lp_mathlib_ULift_supSet___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_s_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_apply_1(v_inst_1_, lean_box(0));
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_supSet___redArg(lean_object* v_inst_4_){
_start:
{
lean_object* v___f_5_; 
v___f_5_ = lean_alloc_closure((void*)(lp_mathlib_ULift_supSet___redArg___lam__0), 2, 1);
lean_closure_set(v___f_5_, 0, v_inst_4_);
return v___f_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_supSet(lean_object* v_00_u03b1_6_, lean_object* v_inst_7_){
_start:
{
lean_object* v___f_8_; 
v___f_8_ = lean_alloc_closure((void*)(lp_mathlib_ULift_supSet___redArg___lam__0), 2, 1);
lean_closure_set(v___f_8_, 0, v_inst_7_);
return v___f_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_infSet___redArg(lean_object* v_inst_9_){
_start:
{
lean_object* v___f_10_; 
v___f_10_ = lean_alloc_closure((void*)(lp_mathlib_ULift_supSet___redArg___lam__0), 2, 1);
lean_closure_set(v___f_10_, 0, v_inst_9_);
return v___f_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_infSet(lean_object* v_00_u03b1_11_, lean_object* v_inst_12_){
_start:
{
lean_object* v___f_13_; 
v___f_13_ = lean_alloc_closure((void*)(lp_mathlib_ULift_supSet___redArg___lam__0), 2, 1);
lean_closure_set(v___f_13_, 0, v_inst_12_);
return v___f_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instCompleteLattice___redArg___lam__0(lean_object* v_inf_14_, lean_object* v_a_15_, lean_object* v_b_16_){
_start:
{
lean_object* v___x_17_; 
v___x_17_ = lean_apply_2(v_inf_14_, v_a_15_, v_b_16_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instCompleteLattice___redArg___lam__1(lean_object* v_toSemilatticeSup_18_, lean_object* v_a_19_, lean_object* v_b_20_){
_start:
{
lean_object* v_sup_21_; lean_object* v___x_22_; 
v_sup_21_ = lean_ctor_get(v_toSemilatticeSup_18_, 1);
lean_inc(v_sup_21_);
lean_dec_ref(v_toSemilatticeSup_18_);
v___x_22_ = lean_apply_2(v_sup_21_, v_a_19_, v_b_20_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instCompleteLattice___redArg(lean_object* v_inst_26_){
_start:
{
lean_object* v_toLattice_27_; lean_object* v_toBoundedOrder_28_; lean_object* v_toSemilatticeSup_29_; lean_object* v_inf_30_; lean_object* v___x_32_; uint8_t v_isShared_33_; uint8_t v_isSharedCheck_64_; 
v_toLattice_27_ = lean_ctor_get(v_inst_26_, 0);
lean_inc_ref(v_toLattice_27_);
v_toBoundedOrder_28_ = lean_ctor_get(v_inst_26_, 3);
lean_inc_ref(v_toBoundedOrder_28_);
v_toSemilatticeSup_29_ = lean_ctor_get(v_toLattice_27_, 0);
v_inf_30_ = lean_ctor_get(v_toLattice_27_, 1);
v_isSharedCheck_64_ = !lean_is_exclusive(v_toLattice_27_);
if (v_isSharedCheck_64_ == 0)
{
v___x_32_ = v_toLattice_27_;
v_isShared_33_ = v_isSharedCheck_64_;
goto v_resetjp_31_;
}
else
{
lean_inc(v_inf_30_);
lean_inc(v_toSemilatticeSup_29_);
lean_dec(v_toLattice_27_);
v___x_32_ = lean_box(0);
v_isShared_33_ = v_isSharedCheck_64_;
goto v_resetjp_31_;
}
v_resetjp_31_:
{
lean_object* v___x_34_; lean_object* v___x_35_; lean_object* v_toSupSet_36_; lean_object* v_toInfSet_37_; lean_object* v___x_39_; uint8_t v_isShared_40_; uint8_t v_isSharedCheck_62_; 
lean_inc_ref(v_inst_26_);
v___x_34_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_inst_26_);
v___x_35_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_inst_26_);
v_toSupSet_36_ = lean_ctor_get(v___x_35_, 1);
lean_inc(v_toSupSet_36_);
lean_dec_ref(v___x_35_);
v_toInfSet_37_ = lean_ctor_get(v___x_34_, 1);
v_isSharedCheck_62_ = !lean_is_exclusive(v___x_34_);
if (v_isSharedCheck_62_ == 0)
{
lean_object* v_unused_63_; 
v_unused_63_ = lean_ctor_get(v___x_34_, 0);
lean_dec(v_unused_63_);
v___x_39_ = v___x_34_;
v_isShared_40_ = v_isSharedCheck_62_;
goto v_resetjp_38_;
}
else
{
lean_inc(v_toInfSet_37_);
lean_dec(v___x_34_);
v___x_39_ = lean_box(0);
v_isShared_40_ = v_isSharedCheck_62_;
goto v_resetjp_38_;
}
v_resetjp_38_:
{
lean_object* v_toOrderTop_41_; lean_object* v_toOrderBot_42_; lean_object* v___x_44_; uint8_t v_isShared_45_; uint8_t v_isSharedCheck_61_; 
v_toOrderTop_41_ = lean_ctor_get(v_toBoundedOrder_28_, 0);
v_toOrderBot_42_ = lean_ctor_get(v_toBoundedOrder_28_, 1);
v_isSharedCheck_61_ = !lean_is_exclusive(v_toBoundedOrder_28_);
if (v_isSharedCheck_61_ == 0)
{
v___x_44_ = v_toBoundedOrder_28_;
v_isShared_45_ = v_isSharedCheck_61_;
goto v_resetjp_43_;
}
else
{
lean_inc(v_toOrderBot_42_);
lean_inc(v_toOrderTop_41_);
lean_dec(v_toBoundedOrder_28_);
v___x_44_ = lean_box(0);
v_isShared_45_ = v_isSharedCheck_61_;
goto v_resetjp_43_;
}
v_resetjp_43_:
{
lean_object* v___f_46_; lean_object* v___f_47_; lean_object* v___f_48_; lean_object* v___f_49_; lean_object* v___x_50_; lean_object* v___x_52_; 
v___f_46_ = lean_alloc_closure((void*)(lp_mathlib_ULift_instCompleteLattice___redArg___lam__0), 3, 1);
lean_closure_set(v___f_46_, 0, v_inf_30_);
v___f_47_ = lean_alloc_closure((void*)(lp_mathlib_ULift_instCompleteLattice___redArg___lam__1), 3, 1);
lean_closure_set(v___f_47_, 0, v_toSemilatticeSup_29_);
v___f_48_ = lean_alloc_closure((void*)(lp_mathlib_ULift_supSet___redArg___lam__0), 2, 1);
lean_closure_set(v___f_48_, 0, v_toSupSet_36_);
v___f_49_ = lean_alloc_closure((void*)(lp_mathlib_ULift_supSet___redArg___lam__0), 2, 1);
lean_closure_set(v___f_49_, 0, v_toInfSet_37_);
v___x_50_ = ((lean_object*)(lp_mathlib_ULift_instCompleteLattice___redArg___closed__0));
if (v_isShared_40_ == 0)
{
lean_ctor_set(v___x_39_, 1, v___f_47_);
lean_ctor_set(v___x_39_, 0, v___x_50_);
v___x_52_ = v___x_39_;
goto v_reusejp_51_;
}
else
{
lean_object* v_reuseFailAlloc_60_; 
v_reuseFailAlloc_60_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_60_, 0, v___x_50_);
lean_ctor_set(v_reuseFailAlloc_60_, 1, v___f_47_);
v___x_52_ = v_reuseFailAlloc_60_;
goto v_reusejp_51_;
}
v_reusejp_51_:
{
lean_object* v___x_54_; 
if (v_isShared_33_ == 0)
{
lean_ctor_set(v___x_32_, 1, v___f_46_);
lean_ctor_set(v___x_32_, 0, v___x_52_);
v___x_54_ = v___x_32_;
goto v_reusejp_53_;
}
else
{
lean_object* v_reuseFailAlloc_59_; 
v_reuseFailAlloc_59_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_59_, 0, v___x_52_);
lean_ctor_set(v_reuseFailAlloc_59_, 1, v___f_46_);
v___x_54_ = v_reuseFailAlloc_59_;
goto v_reusejp_53_;
}
v_reusejp_53_:
{
lean_object* v___x_56_; 
if (v_isShared_45_ == 0)
{
v___x_56_ = v___x_44_;
goto v_reusejp_55_;
}
else
{
lean_object* v_reuseFailAlloc_58_; 
v_reuseFailAlloc_58_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_58_, 0, v_toOrderTop_41_);
lean_ctor_set(v_reuseFailAlloc_58_, 1, v_toOrderBot_42_);
v___x_56_ = v_reuseFailAlloc_58_;
goto v_reusejp_55_;
}
v_reusejp_55_:
{
lean_object* v___x_57_; 
v___x_57_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_57_, 0, v___x_54_);
lean_ctor_set(v___x_57_, 1, v___f_48_);
lean_ctor_set(v___x_57_, 2, v___f_49_);
lean_ctor_set(v___x_57_, 3, v___x_56_);
return v___x_57_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instCompleteLattice(lean_object* v_00_u03b1_65_, lean_object* v_inst_66_){
_start:
{
lean_object* v___x_67_; 
v___x_67_ = lp_mathlib_ULift_instCompleteLattice___redArg(v_inst_66_);
return v___x_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instCompleteLinearOrder___lam__0(lean_object* v_x_68_){
_start:
{
lean_object* v___x_69_; 
v___x_69_ = lean_box(0);
return v___x_69_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_PUnit_instCompleteLinearOrder___lam__1(uint8_t v___x_70_, lean_object* v_x_71_, lean_object* v_x_72_){
_start:
{
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instCompleteLinearOrder___lam__1___boxed(lean_object* v___x_73_, lean_object* v_x_74_, lean_object* v_x_75_){
_start:
{
uint8_t v___x_181__boxed_76_; uint8_t v_res_77_; lean_object* v_r_78_; 
v___x_181__boxed_76_ = lean_unbox(v___x_73_);
v_res_77_ = lp_mathlib_PUnit_instCompleteLinearOrder___lam__1(v___x_181__boxed_76_, v_x_74_, v_x_75_);
v_r_78_ = lean_box(v_res_77_);
return v_r_78_;
}
}
static lean_object* _init_lp_mathlib_PUnit_instCompleteLinearOrder___closed__0(void){
_start:
{
lean_object* v___x_79_; lean_object* v___x_80_; 
v___x_79_ = lp_mathlib_PUnit_instBiheytingAlgebra;
v___x_80_ = lp_mathlib_BiheytingAlgebra_toCoheytingAlgebra___redArg(v___x_79_);
return v___x_80_;
}
}
static lean_object* _init_lp_mathlib_PUnit_instCompleteLinearOrder(void){
_start:
{
lean_object* v___x_99_; lean_object* v_toDistribLattice_100_; lean_object* v_toCompl_101_; lean_object* v_toSDiff_102_; lean_object* v_toHImp_103_; lean_object* v_toTop_104_; lean_object* v_toBot_105_; lean_object* v___x_106_; lean_object* v_toHNot_107_; lean_object* v___f_108_; lean_object* v___f_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___f_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; 
v___x_99_ = lp_mathlib_PUnit_instBooleanAlgebra;
v_toDistribLattice_100_ = lean_ctor_get(v___x_99_, 0);
v_toCompl_101_ = lean_ctor_get(v___x_99_, 1);
v_toSDiff_102_ = lean_ctor_get(v___x_99_, 2);
v_toHImp_103_ = lean_ctor_get(v___x_99_, 3);
v_toTop_104_ = lean_ctor_get(v___x_99_, 4);
v_toBot_105_ = lean_ctor_get(v___x_99_, 5);
v___x_106_ = lean_obj_once(&lp_mathlib_PUnit_instCompleteLinearOrder___closed__0, &lp_mathlib_PUnit_instCompleteLinearOrder___closed__0_once, _init_lp_mathlib_PUnit_instCompleteLinearOrder___closed__0);
v_toHNot_107_ = lean_ctor_get(v___x_106_, 2);
v___f_108_ = ((lean_object*)(lp_mathlib_PUnit_instCompleteLinearOrder___closed__1));
v___f_109_ = ((lean_object*)(lp_mathlib_PUnit_instCompleteLinearOrder___closed__3));
lean_inc(v_toBot_105_);
lean_inc(v_toTop_104_);
v___x_110_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_110_, 0, v_toTop_104_);
lean_ctor_set(v___x_110_, 1, v_toBot_105_);
lean_inc_ref(v_toDistribLattice_100_);
v___x_111_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_111_, 0, v_toDistribLattice_100_);
lean_ctor_set(v___x_111_, 1, v___f_108_);
lean_ctor_set(v___x_111_, 2, v___f_108_);
lean_ctor_set(v___x_111_, 3, v___x_110_);
v___f_112_ = ((lean_object*)(lp_mathlib_PUnit_instCompleteLinearOrder___closed__4));
v___x_113_ = ((lean_object*)(lp_mathlib_PUnit_instCompleteLinearOrder___closed__6));
v___x_114_ = ((lean_object*)(lp_mathlib_PUnit_instCompleteLinearOrder___closed__7));
lean_inc(v_toHNot_107_);
lean_inc(v_toSDiff_102_);
lean_inc(v_toCompl_101_);
lean_inc(v_toHImp_103_);
v___x_115_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v___x_115_, 0, v___x_111_);
lean_ctor_set(v___x_115_, 1, v_toHImp_103_);
lean_ctor_set(v___x_115_, 2, v_toCompl_101_);
lean_ctor_set(v___x_115_, 3, v_toSDiff_102_);
lean_ctor_set(v___x_115_, 4, v_toHNot_107_);
lean_ctor_set(v___x_115_, 5, v___f_109_);
lean_ctor_set(v___x_115_, 6, v___f_112_);
lean_ctor_set(v___x_115_, 7, v___x_113_);
lean_ctor_set(v___x_115_, 8, v___x_114_);
return v___x_115_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Bool_Set(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Set(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_CompleteLattice_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_CompleteLattice_Lemmas(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Bool_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_CompleteLattice_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_PUnit_instCompleteLinearOrder = _init_lp_mathlib_PUnit_instCompleteLinearOrder();
lean_mark_persistent(lp_mathlib_PUnit_instCompleteLinearOrder);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_CompleteLattice_Lemmas(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Bool_Set(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Set(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_CompleteLattice_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_CompleteLattice_Lemmas(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Bool_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_CompleteLattice_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_CompleteLattice_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_CompleteLattice_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_CompleteLattice_Lemmas(builtin);
}
#ifdef __cplusplus
}
#endif
