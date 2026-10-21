// Lean compiler output
// Module: Plausible.Random
// Imports: public import Init public meta import Init
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
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Fin_ofNat___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_randNat___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Prod_map___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* lean_int_add(lean_object*, lean_object*);
lean_object* lean_int_sub(lean_object*, lean_object*);
lean_object* lean_nat_abs(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
extern lean_object* l_IO_stdGenRef;
lean_object* l_ST_Prim_Ref_get___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ST_Prim_Ref_set___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_mkStdGen(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_instMonadLiftTRandGTOfMonadLift___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_instMonadLiftTRandGTOfMonadLift___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_instMonadLiftTRandGTOfMonadLift(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_RandT_up___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_RandT_up___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_RandT_up___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_RandT_up___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_RandT_up___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_RandT_up(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_RandT_up___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_RandT_down___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_RandT_down___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_RandT_down___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_RandT_down(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_RandT_down___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_next___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_next___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_next___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_next(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_split___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_split___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_split___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_split(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_range___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_range___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_range(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_up___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_up(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_up___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_down___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_down(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_down___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_rand___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_rand(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randBound___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randBound(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randFin___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randFin___redArg___lam__0___boxed(lean_object*);
static const lean_closure_object lp_plausible_Plausible_Random_randFin___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible_Plausible_Random_randFin___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_Random_randFin___redArg___closed__0 = (const lean_object*)&lp_plausible_Plausible_Random_randFin___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randFin___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randFin___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randFin(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randFin___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instFinSucc___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instFinSucc___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instFinSucc___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instFinSucc(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randBool___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randBool___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randBool___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randBool(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBool___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBool___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBool(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomNat___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomNat___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomNat___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomNat___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomNat___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomNat(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomInt___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomInt___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomInt___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomInt___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomInt___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomInt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomFin___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomFin___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomFin___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomFin___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomFin(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomFin___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomBitVec___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomBitVec(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomBitVec___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_runRand___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_runRand___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_runRand___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_plausible_Plausible_runRand___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_runRand___redArg___closed__0;
LEAN_EXPORT lean_object* lp_plausible_Plausible_runRand___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_runRand(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_runRandWith___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_runRandWith___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_runRandWith___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_runRandWith(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_runRandWith___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_instMonadLiftTRandGTOfMonadLift___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_00_u03b1_2_, lean_object* v_x_3_, lean_object* v_s_4_){
_start:
{
lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_5_ = lean_apply_1(v_x_3_, v_s_4_);
v___x_6_ = lean_apply_2(v_inst_1_, lean_box(0), v___x_5_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_instMonadLiftTRandGTOfMonadLift___redArg(lean_object* v_inst_7_){
_start:
{
lean_object* v___f_8_; 
v___f_8_ = lean_alloc_closure((void*)(lp_plausible_Plausible_instMonadLiftTRandGTOfMonadLift___redArg___lam__0), 4, 1);
lean_closure_set(v___f_8_, 0, v_inst_7_);
return v___f_8_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_instMonadLiftTRandGTOfMonadLift(lean_object* v_m_9_, lean_object* v_n_10_, lean_object* v_g_11_, lean_object* v_inst_12_){
_start:
{
lean_object* v___f_13_; 
v___f_13_ = lean_alloc_closure((void*)(lp_plausible_Plausible_instMonadLiftTRandGTOfMonadLift___redArg___lam__0), 4, 1);
lean_closure_set(v___f_13_, 0, v_inst_12_);
return v___f_13_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_RandT_up___redArg___lam__0(lean_object* v_fst_14_, lean_object* v_toPure_15_, lean_object* v_____x_16_){
_start:
{
lean_object* v_snd_17_; lean_object* v___x_19_; uint8_t v_isShared_20_; uint8_t v_isSharedCheck_25_; 
v_snd_17_ = lean_ctor_get(v_____x_16_, 1);
v_isSharedCheck_25_ = !lean_is_exclusive(v_____x_16_);
if (v_isSharedCheck_25_ == 0)
{
lean_object* v_unused_26_; 
v_unused_26_ = lean_ctor_get(v_____x_16_, 0);
lean_dec(v_unused_26_);
v___x_19_ = v_____x_16_;
v_isShared_20_ = v_isSharedCheck_25_;
goto v_resetjp_18_;
}
else
{
lean_inc(v_snd_17_);
lean_dec(v_____x_16_);
v___x_19_ = lean_box(0);
v_isShared_20_ = v_isSharedCheck_25_;
goto v_resetjp_18_;
}
v_resetjp_18_:
{
lean_object* v___x_22_; 
if (v_isShared_20_ == 0)
{
lean_ctor_set(v___x_19_, 0, v_fst_14_);
v___x_22_ = v___x_19_;
goto v_reusejp_21_;
}
else
{
lean_object* v_reuseFailAlloc_24_; 
v_reuseFailAlloc_24_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_24_, 0, v_fst_14_);
lean_ctor_set(v_reuseFailAlloc_24_, 1, v_snd_17_);
v___x_22_ = v_reuseFailAlloc_24_;
goto v_reusejp_21_;
}
v_reusejp_21_:
{
lean_object* v___x_23_; 
v___x_23_ = lean_apply_2(v_toPure_15_, lean_box(0), v___x_22_);
return v___x_23_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_RandT_up___redArg___lam__1(lean_object* v_toPure_27_, lean_object* v_toBind_28_, lean_object* v_____x_29_){
_start:
{
lean_object* v_fst_30_; lean_object* v_fst_31_; lean_object* v_snd_32_; lean_object* v___x_34_; uint8_t v_isShared_35_; uint8_t v_isSharedCheck_43_; 
v_fst_30_ = lean_ctor_get(v_____x_29_, 0);
lean_inc(v_fst_30_);
lean_dec_ref(v_____x_29_);
v_fst_31_ = lean_ctor_get(v_fst_30_, 0);
v_snd_32_ = lean_ctor_get(v_fst_30_, 1);
v_isSharedCheck_43_ = !lean_is_exclusive(v_fst_30_);
if (v_isSharedCheck_43_ == 0)
{
v___x_34_ = v_fst_30_;
v_isShared_35_ = v_isSharedCheck_43_;
goto v_resetjp_33_;
}
else
{
lean_inc(v_snd_32_);
lean_inc(v_fst_31_);
lean_dec(v_fst_30_);
v___x_34_ = lean_box(0);
v_isShared_35_ = v_isSharedCheck_43_;
goto v_resetjp_33_;
}
v_resetjp_33_:
{
lean_object* v___f_36_; lean_object* v___x_37_; lean_object* v___x_39_; 
lean_inc(v_toPure_27_);
v___f_36_ = lean_alloc_closure((void*)(lp_plausible_Plausible_RandT_up___redArg___lam__0), 3, 2);
lean_closure_set(v___f_36_, 0, v_fst_31_);
lean_closure_set(v___f_36_, 1, v_toPure_27_);
v___x_37_ = lean_box(0);
if (v_isShared_35_ == 0)
{
lean_ctor_set(v___x_34_, 0, v___x_37_);
v___x_39_ = v___x_34_;
goto v_reusejp_38_;
}
else
{
lean_object* v_reuseFailAlloc_42_; 
v_reuseFailAlloc_42_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_42_, 0, v___x_37_);
lean_ctor_set(v_reuseFailAlloc_42_, 1, v_snd_32_);
v___x_39_ = v_reuseFailAlloc_42_;
goto v_reusejp_38_;
}
v_reusejp_38_:
{
lean_object* v___x_40_; lean_object* v___x_41_; 
v___x_40_ = lean_apply_2(v_toPure_27_, lean_box(0), v___x_39_);
v___x_41_ = lean_apply_4(v_toBind_28_, lean_box(0), lean_box(0), v___x_40_, v___f_36_);
return v___x_41_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_RandT_up___redArg___lam__2(lean_object* v_snd_44_, lean_object* v_toPure_45_, lean_object* v_a_46_){
_start:
{
lean_object* v___x_47_; lean_object* v___x_48_; 
v___x_47_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_47_, 0, v_a_46_);
lean_ctor_set(v___x_47_, 1, v_snd_44_);
v___x_48_ = lean_apply_2(v_toPure_45_, lean_box(0), v___x_47_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_RandT_up___redArg___lam__3(lean_object* v_x_49_, lean_object* v_m__up_50_, lean_object* v_toPure_51_, lean_object* v_toBind_52_, lean_object* v___f_53_, lean_object* v_____x_54_){
_start:
{
lean_object* v_fst_55_; lean_object* v_snd_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___f_59_; lean_object* v___x_60_; lean_object* v___x_61_; 
v_fst_55_ = lean_ctor_get(v_____x_54_, 0);
lean_inc(v_fst_55_);
v_snd_56_ = lean_ctor_get(v_____x_54_, 1);
lean_inc(v_snd_56_);
lean_dec_ref(v_____x_54_);
v___x_57_ = lean_apply_1(v_x_49_, v_fst_55_);
v___x_58_ = lean_apply_2(v_m__up_50_, lean_box(0), v___x_57_);
v___f_59_ = lean_alloc_closure((void*)(lp_plausible_Plausible_RandT_up___redArg___lam__2), 3, 2);
lean_closure_set(v___f_59_, 0, v_snd_56_);
lean_closure_set(v___f_59_, 1, v_toPure_51_);
lean_inc(v_toBind_52_);
v___x_60_ = lean_apply_4(v_toBind_52_, lean_box(0), lean_box(0), v___x_58_, v___f_59_);
v___x_61_ = lean_apply_4(v_toBind_52_, lean_box(0), lean_box(0), v___x_60_, v___f_53_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_RandT_up___redArg(lean_object* v_inst_62_, lean_object* v_m__up_63_, lean_object* v_x_64_, lean_object* v_a_65_){
_start:
{
lean_object* v_toApplicative_66_; lean_object* v_toBind_67_; lean_object* v___x_69_; uint8_t v_isShared_70_; uint8_t v_isSharedCheck_79_; 
v_toApplicative_66_ = lean_ctor_get(v_inst_62_, 0);
v_toBind_67_ = lean_ctor_get(v_inst_62_, 1);
v_isSharedCheck_79_ = !lean_is_exclusive(v_inst_62_);
if (v_isSharedCheck_79_ == 0)
{
v___x_69_ = v_inst_62_;
v_isShared_70_ = v_isSharedCheck_79_;
goto v_resetjp_68_;
}
else
{
lean_inc(v_toBind_67_);
lean_inc(v_toApplicative_66_);
lean_dec(v_inst_62_);
v___x_69_ = lean_box(0);
v_isShared_70_ = v_isSharedCheck_79_;
goto v_resetjp_68_;
}
v_resetjp_68_:
{
lean_object* v_toPure_71_; lean_object* v___f_72_; lean_object* v___f_73_; lean_object* v___x_75_; 
v_toPure_71_ = lean_ctor_get(v_toApplicative_66_, 1);
lean_inc_n(v_toPure_71_, 3);
lean_dec_ref(v_toApplicative_66_);
lean_inc_n(v_toBind_67_, 2);
v___f_72_ = lean_alloc_closure((void*)(lp_plausible_Plausible_RandT_up___redArg___lam__1), 3, 2);
lean_closure_set(v___f_72_, 0, v_toPure_71_);
lean_closure_set(v___f_72_, 1, v_toBind_67_);
v___f_73_ = lean_alloc_closure((void*)(lp_plausible_Plausible_RandT_up___redArg___lam__3), 6, 5);
lean_closure_set(v___f_73_, 0, v_x_64_);
lean_closure_set(v___f_73_, 1, v_m__up_63_);
lean_closure_set(v___f_73_, 2, v_toPure_71_);
lean_closure_set(v___f_73_, 3, v_toBind_67_);
lean_closure_set(v___f_73_, 4, v___f_72_);
lean_inc(v_a_65_);
if (v_isShared_70_ == 0)
{
lean_ctor_set(v___x_69_, 1, v_a_65_);
lean_ctor_set(v___x_69_, 0, v_a_65_);
v___x_75_ = v___x_69_;
goto v_reusejp_74_;
}
else
{
lean_object* v_reuseFailAlloc_78_; 
v_reuseFailAlloc_78_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_78_, 0, v_a_65_);
lean_ctor_set(v_reuseFailAlloc_78_, 1, v_a_65_);
v___x_75_ = v_reuseFailAlloc_78_;
goto v_reusejp_74_;
}
v_reusejp_74_:
{
lean_object* v___x_76_; lean_object* v___x_77_; 
v___x_76_ = lean_apply_2(v_toPure_71_, lean_box(0), v___x_75_);
v___x_77_ = lean_apply_4(v_toBind_67_, lean_box(0), lean_box(0), v___x_76_, v___f_73_);
return v___x_77_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_RandT_up(lean_object* v_00_u03b1_80_, lean_object* v_m_81_, lean_object* v_m_x27_82_, lean_object* v_g_83_, lean_object* v_inst_84_, lean_object* v_inst_85_, lean_object* v_inst_86_, lean_object* v_m__up_87_, lean_object* v_x_88_, lean_object* v_a_89_){
_start:
{
lean_object* v_toApplicative_90_; lean_object* v_toBind_91_; lean_object* v___x_93_; uint8_t v_isShared_94_; uint8_t v_isSharedCheck_103_; 
v_toApplicative_90_ = lean_ctor_get(v_inst_86_, 0);
v_toBind_91_ = lean_ctor_get(v_inst_86_, 1);
v_isSharedCheck_103_ = !lean_is_exclusive(v_inst_86_);
if (v_isSharedCheck_103_ == 0)
{
v___x_93_ = v_inst_86_;
v_isShared_94_ = v_isSharedCheck_103_;
goto v_resetjp_92_;
}
else
{
lean_inc(v_toBind_91_);
lean_inc(v_toApplicative_90_);
lean_dec(v_inst_86_);
v___x_93_ = lean_box(0);
v_isShared_94_ = v_isSharedCheck_103_;
goto v_resetjp_92_;
}
v_resetjp_92_:
{
lean_object* v_toPure_95_; lean_object* v___f_96_; lean_object* v___f_97_; lean_object* v___x_99_; 
v_toPure_95_ = lean_ctor_get(v_toApplicative_90_, 1);
lean_inc_n(v_toPure_95_, 3);
lean_dec_ref(v_toApplicative_90_);
lean_inc_n(v_toBind_91_, 2);
v___f_96_ = lean_alloc_closure((void*)(lp_plausible_Plausible_RandT_up___redArg___lam__1), 3, 2);
lean_closure_set(v___f_96_, 0, v_toPure_95_);
lean_closure_set(v___f_96_, 1, v_toBind_91_);
v___f_97_ = lean_alloc_closure((void*)(lp_plausible_Plausible_RandT_up___redArg___lam__3), 6, 5);
lean_closure_set(v___f_97_, 0, v_x_88_);
lean_closure_set(v___f_97_, 1, v_m__up_87_);
lean_closure_set(v___f_97_, 2, v_toPure_95_);
lean_closure_set(v___f_97_, 3, v_toBind_91_);
lean_closure_set(v___f_97_, 4, v___f_96_);
lean_inc(v_a_89_);
if (v_isShared_94_ == 0)
{
lean_ctor_set(v___x_93_, 1, v_a_89_);
lean_ctor_set(v___x_93_, 0, v_a_89_);
v___x_99_ = v___x_93_;
goto v_reusejp_98_;
}
else
{
lean_object* v_reuseFailAlloc_102_; 
v_reuseFailAlloc_102_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_102_, 0, v_a_89_);
lean_ctor_set(v_reuseFailAlloc_102_, 1, v_a_89_);
v___x_99_ = v_reuseFailAlloc_102_;
goto v_reusejp_98_;
}
v_reusejp_98_:
{
lean_object* v___x_100_; lean_object* v___x_101_; 
v___x_100_ = lean_apply_2(v_toPure_95_, lean_box(0), v___x_99_);
v___x_101_ = lean_apply_4(v_toBind_91_, lean_box(0), lean_box(0), v___x_100_, v___f_97_);
return v___x_101_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_RandT_up___boxed(lean_object* v_00_u03b1_104_, lean_object* v_m_105_, lean_object* v_m_x27_106_, lean_object* v_g_107_, lean_object* v_inst_108_, lean_object* v_inst_109_, lean_object* v_inst_110_, lean_object* v_m__up_111_, lean_object* v_x_112_, lean_object* v_a_113_){
_start:
{
lean_object* v_res_114_; 
v_res_114_ = lp_plausible_Plausible_RandT_up(v_00_u03b1_104_, v_m_105_, v_m_x27_106_, v_g_107_, v_inst_108_, v_inst_109_, v_inst_110_, v_m__up_111_, v_x_112_, v_a_113_);
lean_dec_ref(v_inst_109_);
lean_dec_ref(v_inst_108_);
return v_res_114_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_RandT_down___redArg___lam__2(lean_object* v_toPure_115_, lean_object* v_____x_116_){
_start:
{
lean_object* v_fst_117_; lean_object* v_snd_118_; lean_object* v___x_120_; uint8_t v_isShared_121_; uint8_t v_isSharedCheck_126_; 
v_fst_117_ = lean_ctor_get(v_____x_116_, 0);
v_snd_118_ = lean_ctor_get(v_____x_116_, 1);
v_isSharedCheck_126_ = !lean_is_exclusive(v_____x_116_);
if (v_isSharedCheck_126_ == 0)
{
v___x_120_ = v_____x_116_;
v_isShared_121_ = v_isSharedCheck_126_;
goto v_resetjp_119_;
}
else
{
lean_inc(v_snd_118_);
lean_inc(v_fst_117_);
lean_dec(v_____x_116_);
v___x_120_ = lean_box(0);
v_isShared_121_ = v_isSharedCheck_126_;
goto v_resetjp_119_;
}
v_resetjp_119_:
{
lean_object* v___x_123_; 
if (v_isShared_121_ == 0)
{
v___x_123_ = v___x_120_;
goto v_reusejp_122_;
}
else
{
lean_object* v_reuseFailAlloc_125_; 
v_reuseFailAlloc_125_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_125_, 0, v_fst_117_);
lean_ctor_set(v_reuseFailAlloc_125_, 1, v_snd_118_);
v___x_123_ = v_reuseFailAlloc_125_;
goto v_reusejp_122_;
}
v_reusejp_122_:
{
lean_object* v___x_124_; 
v___x_124_ = lean_apply_2(v_toPure_115_, lean_box(0), v___x_123_);
return v___x_124_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_RandT_down___redArg___lam__1(lean_object* v_x_127_, lean_object* v_toBind_128_, lean_object* v___f_129_, lean_object* v_m__down_130_, lean_object* v_toPure_131_, lean_object* v_toBind_132_, lean_object* v___f_133_, lean_object* v_____x_134_){
_start:
{
lean_object* v_fst_135_; lean_object* v_snd_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___f_140_; lean_object* v___x_141_; lean_object* v___x_142_; 
v_fst_135_ = lean_ctor_get(v_____x_134_, 0);
lean_inc(v_fst_135_);
v_snd_136_ = lean_ctor_get(v_____x_134_, 1);
lean_inc(v_snd_136_);
lean_dec_ref(v_____x_134_);
v___x_137_ = lean_apply_1(v_x_127_, v_fst_135_);
v___x_138_ = lean_apply_4(v_toBind_128_, lean_box(0), lean_box(0), v___x_137_, v___f_129_);
v___x_139_ = lean_apply_2(v_m__down_130_, lean_box(0), v___x_138_);
v___f_140_ = lean_alloc_closure((void*)(lp_plausible_Plausible_RandT_up___redArg___lam__2), 3, 2);
lean_closure_set(v___f_140_, 0, v_snd_136_);
lean_closure_set(v___f_140_, 1, v_toPure_131_);
lean_inc(v_toBind_132_);
v___x_141_ = lean_apply_4(v_toBind_132_, lean_box(0), lean_box(0), v___x_139_, v___f_140_);
v___x_142_ = lean_apply_4(v_toBind_132_, lean_box(0), lean_box(0), v___x_141_, v___f_133_);
return v___x_142_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_RandT_down___redArg(lean_object* v_inst_143_, lean_object* v_inst_144_, lean_object* v_m__down_145_, lean_object* v_x_146_, lean_object* v_a_147_){
_start:
{
lean_object* v_toApplicative_148_; lean_object* v_toApplicative_149_; lean_object* v_toBind_150_; lean_object* v_toPure_151_; lean_object* v_toBind_152_; lean_object* v___x_154_; uint8_t v_isShared_155_; uint8_t v_isSharedCheck_165_; 
v_toApplicative_148_ = lean_ctor_get(v_inst_143_, 0);
lean_inc_ref(v_toApplicative_148_);
v_toApplicative_149_ = lean_ctor_get(v_inst_144_, 0);
lean_inc_ref(v_toApplicative_149_);
v_toBind_150_ = lean_ctor_get(v_inst_143_, 1);
lean_inc(v_toBind_150_);
lean_dec_ref(v_inst_143_);
v_toPure_151_ = lean_ctor_get(v_toApplicative_148_, 1);
lean_inc(v_toPure_151_);
lean_dec_ref(v_toApplicative_148_);
v_toBind_152_ = lean_ctor_get(v_inst_144_, 1);
v_isSharedCheck_165_ = !lean_is_exclusive(v_inst_144_);
if (v_isSharedCheck_165_ == 0)
{
lean_object* v_unused_166_; 
v_unused_166_ = lean_ctor_get(v_inst_144_, 0);
lean_dec(v_unused_166_);
v___x_154_ = v_inst_144_;
v_isShared_155_ = v_isSharedCheck_165_;
goto v_resetjp_153_;
}
else
{
lean_inc(v_toBind_152_);
lean_dec(v_inst_144_);
v___x_154_ = lean_box(0);
v_isShared_155_ = v_isSharedCheck_165_;
goto v_resetjp_153_;
}
v_resetjp_153_:
{
lean_object* v_toPure_156_; lean_object* v___f_157_; lean_object* v___f_158_; lean_object* v___f_159_; lean_object* v___x_161_; 
v_toPure_156_ = lean_ctor_get(v_toApplicative_149_, 1);
lean_inc_n(v_toPure_156_, 3);
lean_dec_ref(v_toApplicative_149_);
lean_inc_n(v_toBind_152_, 2);
v___f_157_ = lean_alloc_closure((void*)(lp_plausible_Plausible_RandT_up___redArg___lam__1), 3, 2);
lean_closure_set(v___f_157_, 0, v_toPure_156_);
lean_closure_set(v___f_157_, 1, v_toBind_152_);
v___f_158_ = lean_alloc_closure((void*)(lp_plausible_Plausible_RandT_down___redArg___lam__2), 2, 1);
lean_closure_set(v___f_158_, 0, v_toPure_151_);
v___f_159_ = lean_alloc_closure((void*)(lp_plausible_Plausible_RandT_down___redArg___lam__1), 8, 7);
lean_closure_set(v___f_159_, 0, v_x_146_);
lean_closure_set(v___f_159_, 1, v_toBind_150_);
lean_closure_set(v___f_159_, 2, v___f_158_);
lean_closure_set(v___f_159_, 3, v_m__down_145_);
lean_closure_set(v___f_159_, 4, v_toPure_156_);
lean_closure_set(v___f_159_, 5, v_toBind_152_);
lean_closure_set(v___f_159_, 6, v___f_157_);
lean_inc(v_a_147_);
if (v_isShared_155_ == 0)
{
lean_ctor_set(v___x_154_, 1, v_a_147_);
lean_ctor_set(v___x_154_, 0, v_a_147_);
v___x_161_ = v___x_154_;
goto v_reusejp_160_;
}
else
{
lean_object* v_reuseFailAlloc_164_; 
v_reuseFailAlloc_164_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_164_, 0, v_a_147_);
lean_ctor_set(v_reuseFailAlloc_164_, 1, v_a_147_);
v___x_161_ = v_reuseFailAlloc_164_;
goto v_reusejp_160_;
}
v_reusejp_160_:
{
lean_object* v___x_162_; lean_object* v___x_163_; 
v___x_162_ = lean_apply_2(v_toPure_156_, lean_box(0), v___x_161_);
v___x_163_ = lean_apply_4(v_toBind_152_, lean_box(0), lean_box(0), v___x_162_, v___f_159_);
return v___x_163_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_RandT_down(lean_object* v_00_u03b1_167_, lean_object* v_m_168_, lean_object* v_m_x27_169_, lean_object* v_g_170_, lean_object* v_inst_171_, lean_object* v_inst_172_, lean_object* v_inst_173_, lean_object* v_m__down_174_, lean_object* v_x_175_, lean_object* v_a_176_){
_start:
{
lean_object* v_toApplicative_177_; lean_object* v_toApplicative_178_; lean_object* v_toBind_179_; lean_object* v_toPure_180_; lean_object* v_toBind_181_; lean_object* v___x_183_; uint8_t v_isShared_184_; uint8_t v_isSharedCheck_194_; 
v_toApplicative_177_ = lean_ctor_get(v_inst_172_, 0);
lean_inc_ref(v_toApplicative_177_);
v_toApplicative_178_ = lean_ctor_get(v_inst_173_, 0);
lean_inc_ref(v_toApplicative_178_);
v_toBind_179_ = lean_ctor_get(v_inst_172_, 1);
lean_inc(v_toBind_179_);
lean_dec_ref(v_inst_172_);
v_toPure_180_ = lean_ctor_get(v_toApplicative_177_, 1);
lean_inc(v_toPure_180_);
lean_dec_ref(v_toApplicative_177_);
v_toBind_181_ = lean_ctor_get(v_inst_173_, 1);
v_isSharedCheck_194_ = !lean_is_exclusive(v_inst_173_);
if (v_isSharedCheck_194_ == 0)
{
lean_object* v_unused_195_; 
v_unused_195_ = lean_ctor_get(v_inst_173_, 0);
lean_dec(v_unused_195_);
v___x_183_ = v_inst_173_;
v_isShared_184_ = v_isSharedCheck_194_;
goto v_resetjp_182_;
}
else
{
lean_inc(v_toBind_181_);
lean_dec(v_inst_173_);
v___x_183_ = lean_box(0);
v_isShared_184_ = v_isSharedCheck_194_;
goto v_resetjp_182_;
}
v_resetjp_182_:
{
lean_object* v_toPure_185_; lean_object* v___f_186_; lean_object* v___f_187_; lean_object* v___f_188_; lean_object* v___x_190_; 
v_toPure_185_ = lean_ctor_get(v_toApplicative_178_, 1);
lean_inc_n(v_toPure_185_, 3);
lean_dec_ref(v_toApplicative_178_);
lean_inc_n(v_toBind_181_, 2);
v___f_186_ = lean_alloc_closure((void*)(lp_plausible_Plausible_RandT_up___redArg___lam__1), 3, 2);
lean_closure_set(v___f_186_, 0, v_toPure_185_);
lean_closure_set(v___f_186_, 1, v_toBind_181_);
v___f_187_ = lean_alloc_closure((void*)(lp_plausible_Plausible_RandT_down___redArg___lam__2), 2, 1);
lean_closure_set(v___f_187_, 0, v_toPure_180_);
v___f_188_ = lean_alloc_closure((void*)(lp_plausible_Plausible_RandT_down___redArg___lam__1), 8, 7);
lean_closure_set(v___f_188_, 0, v_x_175_);
lean_closure_set(v___f_188_, 1, v_toBind_179_);
lean_closure_set(v___f_188_, 2, v___f_187_);
lean_closure_set(v___f_188_, 3, v_m__down_174_);
lean_closure_set(v___f_188_, 4, v_toPure_185_);
lean_closure_set(v___f_188_, 5, v_toBind_181_);
lean_closure_set(v___f_188_, 6, v___f_186_);
lean_inc(v_a_176_);
if (v_isShared_184_ == 0)
{
lean_ctor_set(v___x_183_, 1, v_a_176_);
lean_ctor_set(v___x_183_, 0, v_a_176_);
v___x_190_ = v___x_183_;
goto v_reusejp_189_;
}
else
{
lean_object* v_reuseFailAlloc_193_; 
v_reuseFailAlloc_193_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_193_, 0, v_a_176_);
lean_ctor_set(v_reuseFailAlloc_193_, 1, v_a_176_);
v___x_190_ = v_reuseFailAlloc_193_;
goto v_reusejp_189_;
}
v_reusejp_189_:
{
lean_object* v___x_191_; lean_object* v___x_192_; 
v___x_191_ = lean_apply_2(v_toPure_185_, lean_box(0), v___x_190_);
v___x_192_ = lean_apply_4(v_toBind_181_, lean_box(0), lean_box(0), v___x_191_, v___f_188_);
return v___x_192_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_RandT_down___boxed(lean_object* v_00_u03b1_196_, lean_object* v_m_197_, lean_object* v_m_x27_198_, lean_object* v_g_199_, lean_object* v_inst_200_, lean_object* v_inst_201_, lean_object* v_inst_202_, lean_object* v_m__down_203_, lean_object* v_x_204_, lean_object* v_a_205_){
_start:
{
lean_object* v_res_206_; 
v_res_206_ = lp_plausible_Plausible_RandT_down(v_00_u03b1_196_, v_m_197_, v_m_x27_198_, v_g_199_, v_inst_200_, v_inst_201_, v_inst_202_, v_m__down_203_, v_x_204_, v_a_205_);
lean_dec_ref(v_inst_200_);
return v_res_206_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_next___redArg___lam__0(lean_object* v_fst_207_, lean_object* v_toPure_208_, lean_object* v_____x_209_){
_start:
{
lean_object* v_snd_210_; lean_object* v___x_212_; uint8_t v_isShared_213_; uint8_t v_isSharedCheck_218_; 
v_snd_210_ = lean_ctor_get(v_____x_209_, 1);
v_isSharedCheck_218_ = !lean_is_exclusive(v_____x_209_);
if (v_isSharedCheck_218_ == 0)
{
lean_object* v_unused_219_; 
v_unused_219_ = lean_ctor_get(v_____x_209_, 0);
lean_dec(v_unused_219_);
v___x_212_ = v_____x_209_;
v_isShared_213_ = v_isSharedCheck_218_;
goto v_resetjp_211_;
}
else
{
lean_inc(v_snd_210_);
lean_dec(v_____x_209_);
v___x_212_ = lean_box(0);
v_isShared_213_ = v_isSharedCheck_218_;
goto v_resetjp_211_;
}
v_resetjp_211_:
{
lean_object* v___x_215_; 
if (v_isShared_213_ == 0)
{
lean_ctor_set(v___x_212_, 0, v_fst_207_);
v___x_215_ = v___x_212_;
goto v_reusejp_214_;
}
else
{
lean_object* v_reuseFailAlloc_217_; 
v_reuseFailAlloc_217_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_217_, 0, v_fst_207_);
lean_ctor_set(v_reuseFailAlloc_217_, 1, v_snd_210_);
v___x_215_ = v_reuseFailAlloc_217_;
goto v_reusejp_214_;
}
v_reusejp_214_:
{
lean_object* v___x_216_; 
v___x_216_ = lean_apply_2(v_toPure_208_, lean_box(0), v___x_215_);
return v___x_216_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_next___redArg___lam__1(lean_object* v_inst_220_, lean_object* v_toPure_221_, lean_object* v_toBind_222_, lean_object* v_____x_223_){
_start:
{
lean_object* v_fst_224_; lean_object* v_next_225_; lean_object* v___x_226_; lean_object* v_fst_227_; lean_object* v_snd_228_; lean_object* v___x_230_; uint8_t v_isShared_231_; uint8_t v_isSharedCheck_239_; 
v_fst_224_ = lean_ctor_get(v_____x_223_, 0);
lean_inc(v_fst_224_);
lean_dec_ref(v_____x_223_);
v_next_225_ = lean_ctor_get(v_inst_220_, 1);
lean_inc_ref(v_next_225_);
lean_dec_ref(v_inst_220_);
v___x_226_ = lean_apply_1(v_next_225_, v_fst_224_);
v_fst_227_ = lean_ctor_get(v___x_226_, 0);
v_snd_228_ = lean_ctor_get(v___x_226_, 1);
v_isSharedCheck_239_ = !lean_is_exclusive(v___x_226_);
if (v_isSharedCheck_239_ == 0)
{
v___x_230_ = v___x_226_;
v_isShared_231_ = v_isSharedCheck_239_;
goto v_resetjp_229_;
}
else
{
lean_inc(v_snd_228_);
lean_inc(v_fst_227_);
lean_dec(v___x_226_);
v___x_230_ = lean_box(0);
v_isShared_231_ = v_isSharedCheck_239_;
goto v_resetjp_229_;
}
v_resetjp_229_:
{
lean_object* v___f_232_; lean_object* v___x_233_; lean_object* v___x_235_; 
lean_inc(v_toPure_221_);
v___f_232_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Rand_next___redArg___lam__0), 3, 2);
lean_closure_set(v___f_232_, 0, v_fst_227_);
lean_closure_set(v___f_232_, 1, v_toPure_221_);
v___x_233_ = lean_box(0);
if (v_isShared_231_ == 0)
{
lean_ctor_set(v___x_230_, 0, v___x_233_);
v___x_235_ = v___x_230_;
goto v_reusejp_234_;
}
else
{
lean_object* v_reuseFailAlloc_238_; 
v_reuseFailAlloc_238_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_238_, 0, v___x_233_);
lean_ctor_set(v_reuseFailAlloc_238_, 1, v_snd_228_);
v___x_235_ = v_reuseFailAlloc_238_;
goto v_reusejp_234_;
}
v_reusejp_234_:
{
lean_object* v___x_236_; lean_object* v___x_237_; 
v___x_236_ = lean_apply_2(v_toPure_221_, lean_box(0), v___x_235_);
v___x_237_ = lean_apply_4(v_toBind_222_, lean_box(0), lean_box(0), v___x_236_, v___f_232_);
return v___x_237_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_next___redArg(lean_object* v_inst_240_, lean_object* v_inst_241_, lean_object* v_a_242_){
_start:
{
lean_object* v_toApplicative_243_; lean_object* v_toBind_244_; lean_object* v___x_246_; uint8_t v_isShared_247_; uint8_t v_isSharedCheck_255_; 
v_toApplicative_243_ = lean_ctor_get(v_inst_241_, 0);
v_toBind_244_ = lean_ctor_get(v_inst_241_, 1);
v_isSharedCheck_255_ = !lean_is_exclusive(v_inst_241_);
if (v_isSharedCheck_255_ == 0)
{
v___x_246_ = v_inst_241_;
v_isShared_247_ = v_isSharedCheck_255_;
goto v_resetjp_245_;
}
else
{
lean_inc(v_toBind_244_);
lean_inc(v_toApplicative_243_);
lean_dec(v_inst_241_);
v___x_246_ = lean_box(0);
v_isShared_247_ = v_isSharedCheck_255_;
goto v_resetjp_245_;
}
v_resetjp_245_:
{
lean_object* v_toPure_248_; lean_object* v___f_249_; lean_object* v___x_251_; 
v_toPure_248_ = lean_ctor_get(v_toApplicative_243_, 1);
lean_inc_n(v_toPure_248_, 2);
lean_dec_ref(v_toApplicative_243_);
lean_inc(v_toBind_244_);
v___f_249_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Rand_next___redArg___lam__1), 4, 3);
lean_closure_set(v___f_249_, 0, v_inst_240_);
lean_closure_set(v___f_249_, 1, v_toPure_248_);
lean_closure_set(v___f_249_, 2, v_toBind_244_);
lean_inc(v_a_242_);
if (v_isShared_247_ == 0)
{
lean_ctor_set(v___x_246_, 1, v_a_242_);
lean_ctor_set(v___x_246_, 0, v_a_242_);
v___x_251_ = v___x_246_;
goto v_reusejp_250_;
}
else
{
lean_object* v_reuseFailAlloc_254_; 
v_reuseFailAlloc_254_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_254_, 0, v_a_242_);
lean_ctor_set(v_reuseFailAlloc_254_, 1, v_a_242_);
v___x_251_ = v_reuseFailAlloc_254_;
goto v_reusejp_250_;
}
v_reusejp_250_:
{
lean_object* v___x_252_; lean_object* v___x_253_; 
v___x_252_ = lean_apply_2(v_toPure_248_, lean_box(0), v___x_251_);
v___x_253_ = lean_apply_4(v_toBind_244_, lean_box(0), lean_box(0), v___x_252_, v___f_249_);
return v___x_253_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_next(lean_object* v_g_256_, lean_object* v_m_257_, lean_object* v_inst_258_, lean_object* v_inst_259_, lean_object* v_a_260_){
_start:
{
lean_object* v___x_261_; 
v___x_261_ = lp_plausible_Plausible_Rand_next___redArg(v_inst_258_, v_inst_259_, v_a_260_);
return v___x_261_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_split___redArg___lam__0(lean_object* v_snd_262_, lean_object* v_toPure_263_, lean_object* v_____x_264_){
_start:
{
lean_object* v_snd_265_; lean_object* v___x_267_; uint8_t v_isShared_268_; uint8_t v_isSharedCheck_273_; 
v_snd_265_ = lean_ctor_get(v_____x_264_, 1);
v_isSharedCheck_273_ = !lean_is_exclusive(v_____x_264_);
if (v_isSharedCheck_273_ == 0)
{
lean_object* v_unused_274_; 
v_unused_274_ = lean_ctor_get(v_____x_264_, 0);
lean_dec(v_unused_274_);
v___x_267_ = v_____x_264_;
v_isShared_268_ = v_isSharedCheck_273_;
goto v_resetjp_266_;
}
else
{
lean_inc(v_snd_265_);
lean_dec(v_____x_264_);
v___x_267_ = lean_box(0);
v_isShared_268_ = v_isSharedCheck_273_;
goto v_resetjp_266_;
}
v_resetjp_266_:
{
lean_object* v___x_270_; 
if (v_isShared_268_ == 0)
{
lean_ctor_set(v___x_267_, 0, v_snd_262_);
v___x_270_ = v___x_267_;
goto v_reusejp_269_;
}
else
{
lean_object* v_reuseFailAlloc_272_; 
v_reuseFailAlloc_272_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_272_, 0, v_snd_262_);
lean_ctor_set(v_reuseFailAlloc_272_, 1, v_snd_265_);
v___x_270_ = v_reuseFailAlloc_272_;
goto v_reusejp_269_;
}
v_reusejp_269_:
{
lean_object* v___x_271_; 
v___x_271_ = lean_apply_2(v_toPure_263_, lean_box(0), v___x_270_);
return v___x_271_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_split___redArg___lam__1(lean_object* v_inst_275_, lean_object* v_toPure_276_, lean_object* v_toBind_277_, lean_object* v_____x_278_){
_start:
{
lean_object* v_fst_279_; lean_object* v_split_280_; lean_object* v___x_281_; lean_object* v_fst_282_; lean_object* v_snd_283_; lean_object* v___x_285_; uint8_t v_isShared_286_; uint8_t v_isSharedCheck_294_; 
v_fst_279_ = lean_ctor_get(v_____x_278_, 0);
lean_inc(v_fst_279_);
lean_dec_ref(v_____x_278_);
v_split_280_ = lean_ctor_get(v_inst_275_, 2);
lean_inc_ref(v_split_280_);
lean_dec_ref(v_inst_275_);
v___x_281_ = lean_apply_1(v_split_280_, v_fst_279_);
v_fst_282_ = lean_ctor_get(v___x_281_, 0);
v_snd_283_ = lean_ctor_get(v___x_281_, 1);
v_isSharedCheck_294_ = !lean_is_exclusive(v___x_281_);
if (v_isSharedCheck_294_ == 0)
{
v___x_285_ = v___x_281_;
v_isShared_286_ = v_isSharedCheck_294_;
goto v_resetjp_284_;
}
else
{
lean_inc(v_snd_283_);
lean_inc(v_fst_282_);
lean_dec(v___x_281_);
v___x_285_ = lean_box(0);
v_isShared_286_ = v_isSharedCheck_294_;
goto v_resetjp_284_;
}
v_resetjp_284_:
{
lean_object* v___f_287_; lean_object* v___x_288_; lean_object* v___x_290_; 
lean_inc(v_toPure_276_);
v___f_287_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Rand_split___redArg___lam__0), 3, 2);
lean_closure_set(v___f_287_, 0, v_snd_283_);
lean_closure_set(v___f_287_, 1, v_toPure_276_);
v___x_288_ = lean_box(0);
if (v_isShared_286_ == 0)
{
lean_ctor_set(v___x_285_, 1, v_fst_282_);
lean_ctor_set(v___x_285_, 0, v___x_288_);
v___x_290_ = v___x_285_;
goto v_reusejp_289_;
}
else
{
lean_object* v_reuseFailAlloc_293_; 
v_reuseFailAlloc_293_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_293_, 0, v___x_288_);
lean_ctor_set(v_reuseFailAlloc_293_, 1, v_fst_282_);
v___x_290_ = v_reuseFailAlloc_293_;
goto v_reusejp_289_;
}
v_reusejp_289_:
{
lean_object* v___x_291_; lean_object* v___x_292_; 
v___x_291_ = lean_apply_2(v_toPure_276_, lean_box(0), v___x_290_);
v___x_292_ = lean_apply_4(v_toBind_277_, lean_box(0), lean_box(0), v___x_291_, v___f_287_);
return v___x_292_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_split___redArg(lean_object* v_inst_295_, lean_object* v_inst_296_, lean_object* v_a_297_){
_start:
{
lean_object* v_toApplicative_298_; lean_object* v_toBind_299_; lean_object* v___x_301_; uint8_t v_isShared_302_; uint8_t v_isSharedCheck_310_; 
v_toApplicative_298_ = lean_ctor_get(v_inst_296_, 0);
v_toBind_299_ = lean_ctor_get(v_inst_296_, 1);
v_isSharedCheck_310_ = !lean_is_exclusive(v_inst_296_);
if (v_isSharedCheck_310_ == 0)
{
v___x_301_ = v_inst_296_;
v_isShared_302_ = v_isSharedCheck_310_;
goto v_resetjp_300_;
}
else
{
lean_inc(v_toBind_299_);
lean_inc(v_toApplicative_298_);
lean_dec(v_inst_296_);
v___x_301_ = lean_box(0);
v_isShared_302_ = v_isSharedCheck_310_;
goto v_resetjp_300_;
}
v_resetjp_300_:
{
lean_object* v_toPure_303_; lean_object* v___f_304_; lean_object* v___x_306_; 
v_toPure_303_ = lean_ctor_get(v_toApplicative_298_, 1);
lean_inc_n(v_toPure_303_, 2);
lean_dec_ref(v_toApplicative_298_);
lean_inc(v_toBind_299_);
v___f_304_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Rand_split___redArg___lam__1), 4, 3);
lean_closure_set(v___f_304_, 0, v_inst_295_);
lean_closure_set(v___f_304_, 1, v_toPure_303_);
lean_closure_set(v___f_304_, 2, v_toBind_299_);
lean_inc(v_a_297_);
if (v_isShared_302_ == 0)
{
lean_ctor_set(v___x_301_, 1, v_a_297_);
lean_ctor_set(v___x_301_, 0, v_a_297_);
v___x_306_ = v___x_301_;
goto v_reusejp_305_;
}
else
{
lean_object* v_reuseFailAlloc_309_; 
v_reuseFailAlloc_309_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_309_, 0, v_a_297_);
lean_ctor_set(v_reuseFailAlloc_309_, 1, v_a_297_);
v___x_306_ = v_reuseFailAlloc_309_;
goto v_reusejp_305_;
}
v_reusejp_305_:
{
lean_object* v___x_307_; lean_object* v___x_308_; 
v___x_307_ = lean_apply_2(v_toPure_303_, lean_box(0), v___x_306_);
v___x_308_ = lean_apply_4(v_toBind_299_, lean_box(0), lean_box(0), v___x_307_, v___f_304_);
return v___x_308_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_split(lean_object* v_m_311_, lean_object* v_g_312_, lean_object* v_inst_313_, lean_object* v_inst_314_, lean_object* v_a_315_){
_start:
{
lean_object* v___x_316_; 
v___x_316_ = lp_plausible_Plausible_Rand_split___redArg(v_inst_313_, v_inst_314_, v_a_315_);
return v___x_316_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_range___redArg___lam__0(lean_object* v_inst_317_, lean_object* v_toPure_318_, lean_object* v_____x_319_){
_start:
{
lean_object* v_fst_320_; lean_object* v_snd_321_; lean_object* v___x_323_; uint8_t v_isShared_324_; uint8_t v_isSharedCheck_331_; 
v_fst_320_ = lean_ctor_get(v_____x_319_, 0);
v_snd_321_ = lean_ctor_get(v_____x_319_, 1);
v_isSharedCheck_331_ = !lean_is_exclusive(v_____x_319_);
if (v_isSharedCheck_331_ == 0)
{
v___x_323_ = v_____x_319_;
v_isShared_324_ = v_isSharedCheck_331_;
goto v_resetjp_322_;
}
else
{
lean_inc(v_snd_321_);
lean_inc(v_fst_320_);
lean_dec(v_____x_319_);
v___x_323_ = lean_box(0);
v_isShared_324_ = v_isSharedCheck_331_;
goto v_resetjp_322_;
}
v_resetjp_322_:
{
lean_object* v_range_325_; lean_object* v___x_326_; lean_object* v___x_328_; 
v_range_325_ = lean_ctor_get(v_inst_317_, 0);
lean_inc_ref(v_range_325_);
lean_dec_ref(v_inst_317_);
v___x_326_ = lean_apply_1(v_range_325_, v_fst_320_);
if (v_isShared_324_ == 0)
{
lean_ctor_set(v___x_323_, 0, v___x_326_);
v___x_328_ = v___x_323_;
goto v_reusejp_327_;
}
else
{
lean_object* v_reuseFailAlloc_330_; 
v_reuseFailAlloc_330_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_330_, 0, v___x_326_);
lean_ctor_set(v_reuseFailAlloc_330_, 1, v_snd_321_);
v___x_328_ = v_reuseFailAlloc_330_;
goto v_reusejp_327_;
}
v_reusejp_327_:
{
lean_object* v___x_329_; 
v___x_329_ = lean_apply_2(v_toPure_318_, lean_box(0), v___x_328_);
return v___x_329_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_range___redArg(lean_object* v_inst_332_, lean_object* v_inst_333_, lean_object* v_a_334_){
_start:
{
lean_object* v_toApplicative_335_; lean_object* v_toBind_336_; lean_object* v___x_338_; uint8_t v_isShared_339_; uint8_t v_isSharedCheck_347_; 
v_toApplicative_335_ = lean_ctor_get(v_inst_333_, 0);
v_toBind_336_ = lean_ctor_get(v_inst_333_, 1);
v_isSharedCheck_347_ = !lean_is_exclusive(v_inst_333_);
if (v_isSharedCheck_347_ == 0)
{
v___x_338_ = v_inst_333_;
v_isShared_339_ = v_isSharedCheck_347_;
goto v_resetjp_337_;
}
else
{
lean_inc(v_toBind_336_);
lean_inc(v_toApplicative_335_);
lean_dec(v_inst_333_);
v___x_338_ = lean_box(0);
v_isShared_339_ = v_isSharedCheck_347_;
goto v_resetjp_337_;
}
v_resetjp_337_:
{
lean_object* v_toPure_340_; lean_object* v___f_341_; lean_object* v___x_343_; 
v_toPure_340_ = lean_ctor_get(v_toApplicative_335_, 1);
lean_inc_n(v_toPure_340_, 2);
lean_dec_ref(v_toApplicative_335_);
v___f_341_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Rand_range___redArg___lam__0), 3, 2);
lean_closure_set(v___f_341_, 0, v_inst_332_);
lean_closure_set(v___f_341_, 1, v_toPure_340_);
lean_inc(v_a_334_);
if (v_isShared_339_ == 0)
{
lean_ctor_set(v___x_338_, 1, v_a_334_);
lean_ctor_set(v___x_338_, 0, v_a_334_);
v___x_343_ = v___x_338_;
goto v_reusejp_342_;
}
else
{
lean_object* v_reuseFailAlloc_346_; 
v_reuseFailAlloc_346_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_346_, 0, v_a_334_);
lean_ctor_set(v_reuseFailAlloc_346_, 1, v_a_334_);
v___x_343_ = v_reuseFailAlloc_346_;
goto v_reusejp_342_;
}
v_reusejp_342_:
{
lean_object* v___x_344_; lean_object* v___x_345_; 
v___x_344_ = lean_apply_2(v_toPure_340_, lean_box(0), v___x_343_);
v___x_345_ = lean_apply_4(v_toBind_336_, lean_box(0), lean_box(0), v___x_344_, v___f_341_);
return v___x_345_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_range(lean_object* v_m_348_, lean_object* v_g_349_, lean_object* v_inst_350_, lean_object* v_inst_351_, lean_object* v_a_352_){
_start:
{
lean_object* v___x_353_; 
v___x_353_ = lp_plausible_Plausible_Rand_range___redArg(v_inst_350_, v_inst_351_, v_a_352_);
return v___x_353_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_up___redArg(lean_object* v_x_354_, lean_object* v_a_355_){
_start:
{
lean_object* v___x_356_; lean_object* v_fst_357_; lean_object* v_snd_358_; lean_object* v___x_360_; uint8_t v_isShared_361_; uint8_t v_isSharedCheck_365_; 
v___x_356_ = lean_apply_1(v_x_354_, v_a_355_);
v_fst_357_ = lean_ctor_get(v___x_356_, 0);
v_snd_358_ = lean_ctor_get(v___x_356_, 1);
v_isSharedCheck_365_ = !lean_is_exclusive(v___x_356_);
if (v_isSharedCheck_365_ == 0)
{
v___x_360_ = v___x_356_;
v_isShared_361_ = v_isSharedCheck_365_;
goto v_resetjp_359_;
}
else
{
lean_inc(v_snd_358_);
lean_inc(v_fst_357_);
lean_dec(v___x_356_);
v___x_360_ = lean_box(0);
v_isShared_361_ = v_isSharedCheck_365_;
goto v_resetjp_359_;
}
v_resetjp_359_:
{
lean_object* v___x_363_; 
if (v_isShared_361_ == 0)
{
v___x_363_ = v___x_360_;
goto v_reusejp_362_;
}
else
{
lean_object* v_reuseFailAlloc_364_; 
v_reuseFailAlloc_364_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_364_, 0, v_fst_357_);
lean_ctor_set(v_reuseFailAlloc_364_, 1, v_snd_358_);
v___x_363_ = v_reuseFailAlloc_364_;
goto v_reusejp_362_;
}
v_reusejp_362_:
{
return v___x_363_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_up(lean_object* v_00_u03b1_366_, lean_object* v_g_367_, lean_object* v_inst_368_, lean_object* v_x_369_, lean_object* v_a_370_){
_start:
{
lean_object* v___x_371_; lean_object* v_fst_372_; lean_object* v_snd_373_; lean_object* v___x_375_; uint8_t v_isShared_376_; uint8_t v_isSharedCheck_380_; 
v___x_371_ = lean_apply_1(v_x_369_, v_a_370_);
v_fst_372_ = lean_ctor_get(v___x_371_, 0);
v_snd_373_ = lean_ctor_get(v___x_371_, 1);
v_isSharedCheck_380_ = !lean_is_exclusive(v___x_371_);
if (v_isSharedCheck_380_ == 0)
{
v___x_375_ = v___x_371_;
v_isShared_376_ = v_isSharedCheck_380_;
goto v_resetjp_374_;
}
else
{
lean_inc(v_snd_373_);
lean_inc(v_fst_372_);
lean_dec(v___x_371_);
v___x_375_ = lean_box(0);
v_isShared_376_ = v_isSharedCheck_380_;
goto v_resetjp_374_;
}
v_resetjp_374_:
{
lean_object* v___x_378_; 
if (v_isShared_376_ == 0)
{
v___x_378_ = v___x_375_;
goto v_reusejp_377_;
}
else
{
lean_object* v_reuseFailAlloc_379_; 
v_reuseFailAlloc_379_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_379_, 0, v_fst_372_);
lean_ctor_set(v_reuseFailAlloc_379_, 1, v_snd_373_);
v___x_378_ = v_reuseFailAlloc_379_;
goto v_reusejp_377_;
}
v_reusejp_377_:
{
return v___x_378_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_up___boxed(lean_object* v_00_u03b1_381_, lean_object* v_g_382_, lean_object* v_inst_383_, lean_object* v_x_384_, lean_object* v_a_385_){
_start:
{
lean_object* v_res_386_; 
v_res_386_ = lp_plausible_Plausible_Rand_up(v_00_u03b1_381_, v_g_382_, v_inst_383_, v_x_384_, v_a_385_);
lean_dec_ref(v_inst_383_);
return v_res_386_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_down___redArg(lean_object* v_x_387_, lean_object* v_a_388_){
_start:
{
lean_object* v___x_389_; lean_object* v_fst_390_; lean_object* v_snd_391_; lean_object* v___x_393_; uint8_t v_isShared_394_; uint8_t v_isSharedCheck_398_; 
v___x_389_ = lean_apply_1(v_x_387_, v_a_388_);
v_fst_390_ = lean_ctor_get(v___x_389_, 0);
v_snd_391_ = lean_ctor_get(v___x_389_, 1);
v_isSharedCheck_398_ = !lean_is_exclusive(v___x_389_);
if (v_isSharedCheck_398_ == 0)
{
v___x_393_ = v___x_389_;
v_isShared_394_ = v_isSharedCheck_398_;
goto v_resetjp_392_;
}
else
{
lean_inc(v_snd_391_);
lean_inc(v_fst_390_);
lean_dec(v___x_389_);
v___x_393_ = lean_box(0);
v_isShared_394_ = v_isSharedCheck_398_;
goto v_resetjp_392_;
}
v_resetjp_392_:
{
lean_object* v___x_396_; 
if (v_isShared_394_ == 0)
{
v___x_396_ = v___x_393_;
goto v_reusejp_395_;
}
else
{
lean_object* v_reuseFailAlloc_397_; 
v_reuseFailAlloc_397_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_397_, 0, v_fst_390_);
lean_ctor_set(v_reuseFailAlloc_397_, 1, v_snd_391_);
v___x_396_ = v_reuseFailAlloc_397_;
goto v_reusejp_395_;
}
v_reusejp_395_:
{
return v___x_396_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_down(lean_object* v_00_u03b1_399_, lean_object* v_g_400_, lean_object* v_inst_401_, lean_object* v_x_402_, lean_object* v_a_403_){
_start:
{
lean_object* v___x_404_; lean_object* v_fst_405_; lean_object* v_snd_406_; lean_object* v___x_408_; uint8_t v_isShared_409_; uint8_t v_isSharedCheck_413_; 
v___x_404_ = lean_apply_1(v_x_402_, v_a_403_);
v_fst_405_ = lean_ctor_get(v___x_404_, 0);
v_snd_406_ = lean_ctor_get(v___x_404_, 1);
v_isSharedCheck_413_ = !lean_is_exclusive(v___x_404_);
if (v_isSharedCheck_413_ == 0)
{
v___x_408_ = v___x_404_;
v_isShared_409_ = v_isSharedCheck_413_;
goto v_resetjp_407_;
}
else
{
lean_inc(v_snd_406_);
lean_inc(v_fst_405_);
lean_dec(v___x_404_);
v___x_408_ = lean_box(0);
v_isShared_409_ = v_isSharedCheck_413_;
goto v_resetjp_407_;
}
v_resetjp_407_:
{
lean_object* v___x_411_; 
if (v_isShared_409_ == 0)
{
v___x_411_ = v___x_408_;
goto v_reusejp_410_;
}
else
{
lean_object* v_reuseFailAlloc_412_; 
v_reuseFailAlloc_412_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_412_, 0, v_fst_405_);
lean_ctor_set(v_reuseFailAlloc_412_, 1, v_snd_406_);
v___x_411_ = v_reuseFailAlloc_412_;
goto v_reusejp_410_;
}
v_reusejp_410_:
{
return v___x_411_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_down___boxed(lean_object* v_00_u03b1_414_, lean_object* v_g_415_, lean_object* v_inst_416_, lean_object* v_x_417_, lean_object* v_a_418_){
_start:
{
lean_object* v_res_419_; 
v_res_419_ = lp_plausible_Plausible_Rand_down(v_00_u03b1_414_, v_g_415_, v_inst_416_, v_x_417_, v_a_418_);
lean_dec_ref(v_inst_416_);
return v_res_419_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_rand___redArg(lean_object* v_inst_420_, lean_object* v_inst_421_, lean_object* v_a_422_){
_start:
{
lean_object* v___x_423_; 
v___x_423_ = lean_apply_3(v_inst_420_, lean_box(0), v_inst_421_, v_a_422_);
return v___x_423_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_rand(lean_object* v_m_424_, lean_object* v_g_425_, lean_object* v_00_u03b1_426_, lean_object* v_inst_427_, lean_object* v_inst_428_, lean_object* v_a_429_){
_start:
{
lean_object* v___x_430_; 
v___x_430_ = lean_apply_3(v_inst_427_, lean_box(0), v_inst_428_, v_a_429_);
return v___x_430_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randBound___redArg(lean_object* v_inst_431_, lean_object* v_lo_432_, lean_object* v_hi_433_, lean_object* v_inst_434_, lean_object* v_a_435_){
_start:
{
lean_object* v___x_436_; 
v___x_436_ = lean_apply_6(v_inst_431_, lean_box(0), v_lo_432_, v_hi_433_, lean_box(0), v_inst_434_, v_a_435_);
return v___x_436_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randBound(lean_object* v_m_437_, lean_object* v_g_438_, lean_object* v_00_u03b1_439_, lean_object* v_inst_440_, lean_object* v_inst_441_, lean_object* v_lo_442_, lean_object* v_hi_443_, lean_object* v_h_444_, lean_object* v_inst_445_, lean_object* v_a_446_){
_start:
{
lean_object* v___x_447_; 
v___x_447_ = lean_apply_6(v_inst_441_, lean_box(0), v_lo_442_, v_hi_443_, lean_box(0), v_inst_445_, v_a_446_);
return v___x_447_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randFin___redArg___lam__0(lean_object* v_down_448_){
_start:
{
lean_inc(v_down_448_);
return v_down_448_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randFin___redArg___lam__0___boxed(lean_object* v_down_449_){
_start:
{
lean_object* v_res_450_; 
v_res_450_ = lp_plausible_Plausible_Random_randFin___redArg___lam__0(v_down_449_);
lean_dec(v_down_449_);
return v_res_450_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randFin___redArg(lean_object* v_inst_452_, lean_object* v_n_453_, lean_object* v_inst_454_, lean_object* v_x_455_){
_start:
{
lean_object* v_toApplicative_456_; lean_object* v_toPure_457_; lean_object* v___f_458_; lean_object* v___x_459_; lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_462_; lean_object* v___x_463_; lean_object* v___x_464_; lean_object* v___x_465_; 
v_toApplicative_456_ = lean_ctor_get(v_inst_452_, 0);
lean_inc_ref(v_toApplicative_456_);
lean_dec_ref(v_inst_452_);
v_toPure_457_ = lean_ctor_get(v_toApplicative_456_, 1);
lean_inc(v_toPure_457_);
lean_dec_ref(v_toApplicative_456_);
v___f_458_ = ((lean_object*)(lp_plausible_Plausible_Random_randFin___redArg___closed__0));
v___x_459_ = lean_unsigned_to_nat(1u);
v___x_460_ = lean_nat_add(v_n_453_, v___x_459_);
v___x_461_ = lean_alloc_closure((void*)(l_Fin_ofNat___boxed), 3, 2);
lean_closure_set(v___x_461_, 0, v___x_460_);
lean_closure_set(v___x_461_, 1, lean_box(0));
v___x_462_ = lean_unsigned_to_nat(0u);
v___x_463_ = l_randNat___redArg(v_inst_454_, v_x_455_, v___x_462_, v_n_453_);
v___x_464_ = l_Prod_map___redArg(v___x_461_, v___f_458_, v___x_463_);
v___x_465_ = lean_apply_2(v_toPure_457_, lean_box(0), v___x_464_);
return v___x_465_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randFin___redArg___boxed(lean_object* v_inst_466_, lean_object* v_n_467_, lean_object* v_inst_468_, lean_object* v_x_469_){
_start:
{
lean_object* v_res_470_; 
v_res_470_ = lp_plausible_Plausible_Random_randFin___redArg(v_inst_466_, v_n_467_, v_inst_468_, v_x_469_);
lean_dec(v_n_467_);
return v_res_470_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randFin(lean_object* v_m_471_, lean_object* v_inst_472_, lean_object* v_g_473_, lean_object* v_n_474_, lean_object* v_inst_475_, lean_object* v_x_476_){
_start:
{
lean_object* v___x_477_; 
v___x_477_ = lp_plausible_Plausible_Random_randFin___redArg(v_inst_472_, v_n_474_, v_inst_475_, v_x_476_);
return v___x_477_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randFin___boxed(lean_object* v_m_478_, lean_object* v_inst_479_, lean_object* v_g_480_, lean_object* v_n_481_, lean_object* v_inst_482_, lean_object* v_x_483_){
_start:
{
lean_object* v_res_484_; 
v_res_484_ = lp_plausible_Plausible_Random_randFin(v_m_478_, v_inst_479_, v_g_480_, v_n_481_, v_inst_482_, v_x_483_);
lean_dec(v_n_481_);
return v_res_484_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instFinSucc___redArg___lam__0(lean_object* v_inst_485_, lean_object* v_n_486_, lean_object* v_g_487_, lean_object* v_inst_488_, lean_object* v___y_489_){
_start:
{
lean_object* v___x_490_; 
v___x_490_ = lp_plausible_Plausible_Random_randFin___redArg(v_inst_485_, v_n_486_, v_inst_488_, v___y_489_);
return v___x_490_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instFinSucc___redArg___lam__0___boxed(lean_object* v_inst_491_, lean_object* v_n_492_, lean_object* v_g_493_, lean_object* v_inst_494_, lean_object* v___y_495_){
_start:
{
lean_object* v_res_496_; 
v_res_496_ = lp_plausible_Plausible_Random_instFinSucc___redArg___lam__0(v_inst_491_, v_n_492_, v_g_493_, v_inst_494_, v___y_495_);
lean_dec(v_n_492_);
return v_res_496_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instFinSucc___redArg(lean_object* v_inst_497_, lean_object* v_n_498_){
_start:
{
lean_object* v___f_499_; 
v___f_499_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Random_instFinSucc___redArg___lam__0___boxed), 5, 2);
lean_closure_set(v___f_499_, 0, v_inst_497_);
lean_closure_set(v___f_499_, 1, v_n_498_);
return v___f_499_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instFinSucc(lean_object* v_m_500_, lean_object* v_inst_501_, lean_object* v_n_502_){
_start:
{
lean_object* v___f_503_; 
v___f_503_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Random_instFinSucc___redArg___lam__0___boxed), 5, 2);
lean_closure_set(v___f_503_, 0, v_inst_501_);
lean_closure_set(v___f_503_, 1, v_n_502_);
return v___f_503_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randBool___redArg___lam__0(lean_object* v_toApplicative_504_, lean_object* v___x_505_, lean_object* v_____x_506_){
_start:
{
lean_object* v_fst_507_; lean_object* v_snd_508_; lean_object* v___x_510_; uint8_t v_isShared_511_; uint8_t v_isSharedCheck_521_; 
v_fst_507_ = lean_ctor_get(v_____x_506_, 0);
v_snd_508_ = lean_ctor_get(v_____x_506_, 1);
v_isSharedCheck_521_ = !lean_is_exclusive(v_____x_506_);
if (v_isSharedCheck_521_ == 0)
{
v___x_510_ = v_____x_506_;
v_isShared_511_ = v_isSharedCheck_521_;
goto v_resetjp_509_;
}
else
{
lean_inc(v_snd_508_);
lean_inc(v_fst_507_);
lean_dec(v_____x_506_);
v___x_510_ = lean_box(0);
v_isShared_511_ = v_isSharedCheck_521_;
goto v_resetjp_509_;
}
v_resetjp_509_:
{
lean_object* v_toPure_512_; lean_object* v___x_513_; lean_object* v___x_514_; uint8_t v___x_515_; lean_object* v___x_516_; lean_object* v___x_518_; 
v_toPure_512_ = lean_ctor_get(v_toApplicative_504_, 1);
lean_inc(v_toPure_512_);
lean_dec_ref(v_toApplicative_504_);
v___x_513_ = lean_unsigned_to_nat(2u);
v___x_514_ = lean_nat_mod(v___x_505_, v___x_513_);
v___x_515_ = lean_nat_dec_eq(v_fst_507_, v___x_514_);
lean_dec(v___x_514_);
lean_dec(v_fst_507_);
v___x_516_ = lean_box(v___x_515_);
if (v_isShared_511_ == 0)
{
lean_ctor_set(v___x_510_, 0, v___x_516_);
v___x_518_ = v___x_510_;
goto v_reusejp_517_;
}
else
{
lean_object* v_reuseFailAlloc_520_; 
v_reuseFailAlloc_520_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_520_, 0, v___x_516_);
lean_ctor_set(v_reuseFailAlloc_520_, 1, v_snd_508_);
v___x_518_ = v_reuseFailAlloc_520_;
goto v_reusejp_517_;
}
v_reusejp_517_:
{
lean_object* v___x_519_; 
v___x_519_ = lean_apply_2(v_toPure_512_, lean_box(0), v___x_518_);
return v___x_519_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randBool___redArg___lam__0___boxed(lean_object* v_toApplicative_522_, lean_object* v___x_523_, lean_object* v_____x_524_){
_start:
{
lean_object* v_res_525_; 
v_res_525_ = lp_plausible_Plausible_Random_randBool___redArg___lam__0(v_toApplicative_522_, v___x_523_, v_____x_524_);
lean_dec(v___x_523_);
return v_res_525_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randBool___redArg(lean_object* v_inst_526_, lean_object* v_inst_527_, lean_object* v_a_528_){
_start:
{
lean_object* v_toApplicative_529_; lean_object* v_toBind_530_; lean_object* v___x_531_; lean_object* v___f_532_; lean_object* v___x_533_; lean_object* v___x_534_; 
v_toApplicative_529_ = lean_ctor_get(v_inst_526_, 0);
v_toBind_530_ = lean_ctor_get(v_inst_526_, 1);
lean_inc(v_toBind_530_);
v___x_531_ = lean_unsigned_to_nat(1u);
lean_inc_ref(v_toApplicative_529_);
v___f_532_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Random_randBool___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_532_, 0, v_toApplicative_529_);
lean_closure_set(v___f_532_, 1, v___x_531_);
v___x_533_ = lp_plausible_Plausible_Random_randFin___redArg(v_inst_526_, v___x_531_, v_inst_527_, v_a_528_);
v___x_534_ = lean_apply_4(v_toBind_530_, lean_box(0), lean_box(0), v___x_533_, v___f_532_);
return v___x_534_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randBool(lean_object* v_m_535_, lean_object* v_inst_536_, lean_object* v_g_537_, lean_object* v_inst_538_, lean_object* v_a_539_){
_start:
{
lean_object* v___x_540_; 
v___x_540_ = lp_plausible_Plausible_Random_randBool___redArg(v_inst_536_, v_inst_538_, v_a_539_);
return v___x_540_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBool___redArg___lam__0(lean_object* v_inst_541_, lean_object* v_g_542_, lean_object* v_inst_543_, lean_object* v___y_544_){
_start:
{
lean_object* v___x_545_; 
v___x_545_ = lp_plausible_Plausible_Random_randBool___redArg(v_inst_541_, v_inst_543_, v___y_544_);
return v___x_545_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBool___redArg(lean_object* v_inst_546_){
_start:
{
lean_object* v___f_547_; 
v___f_547_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Random_instBool___redArg___lam__0), 4, 1);
lean_closure_set(v___f_547_, 0, v_inst_546_);
return v___f_547_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBool(lean_object* v_m_548_, lean_object* v_inst_549_){
_start:
{
lean_object* v___f_550_; 
v___f_550_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Random_instBool___redArg___lam__0), 4, 1);
lean_closure_set(v___f_550_, 0, v_inst_549_);
return v___f_550_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomNat___redArg___lam__0(lean_object* v_toApplicative_551_, lean_object* v_lo_552_, lean_object* v_____x_553_){
_start:
{
lean_object* v_fst_554_; lean_object* v_snd_555_; lean_object* v___x_557_; uint8_t v_isShared_558_; uint8_t v_isSharedCheck_565_; 
v_fst_554_ = lean_ctor_get(v_____x_553_, 0);
v_snd_555_ = lean_ctor_get(v_____x_553_, 1);
v_isSharedCheck_565_ = !lean_is_exclusive(v_____x_553_);
if (v_isSharedCheck_565_ == 0)
{
v___x_557_ = v_____x_553_;
v_isShared_558_ = v_isSharedCheck_565_;
goto v_resetjp_556_;
}
else
{
lean_inc(v_snd_555_);
lean_inc(v_fst_554_);
lean_dec(v_____x_553_);
v___x_557_ = lean_box(0);
v_isShared_558_ = v_isSharedCheck_565_;
goto v_resetjp_556_;
}
v_resetjp_556_:
{
lean_object* v_toPure_559_; lean_object* v___x_560_; lean_object* v___x_562_; 
v_toPure_559_ = lean_ctor_get(v_toApplicative_551_, 1);
lean_inc(v_toPure_559_);
lean_dec_ref(v_toApplicative_551_);
v___x_560_ = lean_nat_add(v_lo_552_, v_fst_554_);
lean_dec(v_fst_554_);
if (v_isShared_558_ == 0)
{
lean_ctor_set(v___x_557_, 0, v___x_560_);
v___x_562_ = v___x_557_;
goto v_reusejp_561_;
}
else
{
lean_object* v_reuseFailAlloc_564_; 
v_reuseFailAlloc_564_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_564_, 0, v___x_560_);
lean_ctor_set(v_reuseFailAlloc_564_, 1, v_snd_555_);
v___x_562_ = v_reuseFailAlloc_564_;
goto v_reusejp_561_;
}
v_reusejp_561_:
{
lean_object* v___x_563_; 
v___x_563_ = lean_apply_2(v_toPure_559_, lean_box(0), v___x_562_);
return v___x_563_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomNat___redArg___lam__0___boxed(lean_object* v_toApplicative_566_, lean_object* v_lo_567_, lean_object* v_____x_568_){
_start:
{
lean_object* v_res_569_; 
v_res_569_ = lp_plausible_Plausible_Random_instBoundedRandomNat___redArg___lam__0(v_toApplicative_566_, v_lo_567_, v_____x_568_);
lean_dec(v_lo_567_);
return v_res_569_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomNat___redArg___lam__1(lean_object* v_inst_570_, lean_object* v_g_571_, lean_object* v_lo_572_, lean_object* v_hi_573_, lean_object* v_h_574_, lean_object* v_x_575_, lean_object* v___y_576_){
_start:
{
lean_object* v_toApplicative_577_; lean_object* v_toBind_578_; lean_object* v___f_579_; lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; 
v_toApplicative_577_ = lean_ctor_get(v_inst_570_, 0);
v_toBind_578_ = lean_ctor_get(v_inst_570_, 1);
lean_inc(v_toBind_578_);
lean_inc(v_lo_572_);
lean_inc_ref(v_toApplicative_577_);
v___f_579_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Random_instBoundedRandomNat___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_579_, 0, v_toApplicative_577_);
lean_closure_set(v___f_579_, 1, v_lo_572_);
v___x_580_ = lean_nat_sub(v_hi_573_, v_lo_572_);
lean_dec(v_lo_572_);
v___x_581_ = lp_plausible_Plausible_Random_randFin___redArg(v_inst_570_, v___x_580_, v_x_575_, v___y_576_);
lean_dec(v___x_580_);
v___x_582_ = lean_apply_4(v_toBind_578_, lean_box(0), lean_box(0), v___x_581_, v___f_579_);
return v___x_582_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomNat___redArg___lam__1___boxed(lean_object* v_inst_583_, lean_object* v_g_584_, lean_object* v_lo_585_, lean_object* v_hi_586_, lean_object* v_h_587_, lean_object* v_x_588_, lean_object* v___y_589_){
_start:
{
lean_object* v_res_590_; 
v_res_590_ = lp_plausible_Plausible_Random_instBoundedRandomNat___redArg___lam__1(v_inst_583_, v_g_584_, v_lo_585_, v_hi_586_, v_h_587_, v_x_588_, v___y_589_);
lean_dec(v_hi_586_);
return v_res_590_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomNat___redArg(lean_object* v_inst_591_){
_start:
{
lean_object* v___f_592_; 
v___f_592_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Random_instBoundedRandomNat___redArg___lam__1___boxed), 7, 1);
lean_closure_set(v___f_592_, 0, v_inst_591_);
return v___f_592_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomNat(lean_object* v_m_593_, lean_object* v_inst_594_){
_start:
{
lean_object* v___f_595_; 
v___f_595_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Random_instBoundedRandomNat___redArg___lam__1___boxed), 7, 1);
lean_closure_set(v___f_595_, 0, v_inst_594_);
return v___f_595_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomInt___redArg___lam__0(lean_object* v_toApplicative_596_, lean_object* v_lo_597_, lean_object* v_____x_598_){
_start:
{
lean_object* v_fst_599_; lean_object* v_snd_600_; lean_object* v___x_602_; uint8_t v_isShared_603_; uint8_t v_isSharedCheck_611_; 
v_fst_599_ = lean_ctor_get(v_____x_598_, 0);
v_snd_600_ = lean_ctor_get(v_____x_598_, 1);
v_isSharedCheck_611_ = !lean_is_exclusive(v_____x_598_);
if (v_isSharedCheck_611_ == 0)
{
v___x_602_ = v_____x_598_;
v_isShared_603_ = v_isSharedCheck_611_;
goto v_resetjp_601_;
}
else
{
lean_inc(v_snd_600_);
lean_inc(v_fst_599_);
lean_dec(v_____x_598_);
v___x_602_ = lean_box(0);
v_isShared_603_ = v_isSharedCheck_611_;
goto v_resetjp_601_;
}
v_resetjp_601_:
{
lean_object* v_toPure_604_; lean_object* v___x_605_; lean_object* v___x_606_; lean_object* v___x_608_; 
v_toPure_604_ = lean_ctor_get(v_toApplicative_596_, 1);
lean_inc(v_toPure_604_);
lean_dec_ref(v_toApplicative_596_);
v___x_605_ = lean_nat_to_int(v_fst_599_);
v___x_606_ = lean_int_add(v___x_605_, v_lo_597_);
lean_dec(v___x_605_);
if (v_isShared_603_ == 0)
{
lean_ctor_set(v___x_602_, 0, v___x_606_);
v___x_608_ = v___x_602_;
goto v_reusejp_607_;
}
else
{
lean_object* v_reuseFailAlloc_610_; 
v_reuseFailAlloc_610_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_610_, 0, v___x_606_);
lean_ctor_set(v_reuseFailAlloc_610_, 1, v_snd_600_);
v___x_608_ = v_reuseFailAlloc_610_;
goto v_reusejp_607_;
}
v_reusejp_607_:
{
lean_object* v___x_609_; 
v___x_609_ = lean_apply_2(v_toPure_604_, lean_box(0), v___x_608_);
return v___x_609_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomInt___redArg___lam__0___boxed(lean_object* v_toApplicative_612_, lean_object* v_lo_613_, lean_object* v_____x_614_){
_start:
{
lean_object* v_res_615_; 
v_res_615_ = lp_plausible_Plausible_Random_instBoundedRandomInt___redArg___lam__0(v_toApplicative_612_, v_lo_613_, v_____x_614_);
lean_dec(v_lo_613_);
return v_res_615_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomInt___redArg___lam__1(lean_object* v_inst_616_, lean_object* v_g_617_, lean_object* v_lo_618_, lean_object* v_hi_619_, lean_object* v_h_620_, lean_object* v_x_621_, lean_object* v___y_622_){
_start:
{
lean_object* v_toApplicative_623_; lean_object* v_toBind_624_; lean_object* v___f_625_; lean_object* v___x_626_; lean_object* v___x_627_; lean_object* v___x_628_; lean_object* v___x_629_; lean_object* v___x_630_; 
v_toApplicative_623_ = lean_ctor_get(v_inst_616_, 0);
v_toBind_624_ = lean_ctor_get(v_inst_616_, 1);
lean_inc(v_toBind_624_);
lean_inc(v_lo_618_);
lean_inc_ref(v_toApplicative_623_);
v___f_625_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Random_instBoundedRandomInt___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_625_, 0, v_toApplicative_623_);
lean_closure_set(v___f_625_, 1, v_lo_618_);
v___x_626_ = lean_unsigned_to_nat(0u);
v___x_627_ = lean_int_sub(v_hi_619_, v_lo_618_);
lean_dec(v_lo_618_);
v___x_628_ = lean_nat_abs(v___x_627_);
lean_dec(v___x_627_);
v___x_629_ = lp_plausible_Plausible_Random_instBoundedRandomNat___redArg___lam__1(v_inst_616_, lean_box(0), v___x_626_, v___x_628_, lean_box(0), v_x_621_, v___y_622_);
lean_dec(v___x_628_);
v___x_630_ = lean_apply_4(v_toBind_624_, lean_box(0), lean_box(0), v___x_629_, v___f_625_);
return v___x_630_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomInt___redArg___lam__1___boxed(lean_object* v_inst_631_, lean_object* v_g_632_, lean_object* v_lo_633_, lean_object* v_hi_634_, lean_object* v_h_635_, lean_object* v_x_636_, lean_object* v___y_637_){
_start:
{
lean_object* v_res_638_; 
v_res_638_ = lp_plausible_Plausible_Random_instBoundedRandomInt___redArg___lam__1(v_inst_631_, v_g_632_, v_lo_633_, v_hi_634_, v_h_635_, v_x_636_, v___y_637_);
lean_dec(v_hi_634_);
return v_res_638_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomInt___redArg(lean_object* v_inst_639_){
_start:
{
lean_object* v___f_640_; 
v___f_640_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Random_instBoundedRandomInt___redArg___lam__1___boxed), 7, 1);
lean_closure_set(v___f_640_, 0, v_inst_639_);
return v___f_640_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomInt(lean_object* v_m_641_, lean_object* v_inst_642_){
_start:
{
lean_object* v___f_643_; 
v___f_643_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Random_instBoundedRandomInt___redArg___lam__1___boxed), 7, 1);
lean_closure_set(v___f_643_, 0, v_inst_642_);
return v___f_643_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomFin___redArg___lam__0(lean_object* v_inst_644_, lean_object* v_____x_645_){
_start:
{
lean_object* v_toApplicative_646_; lean_object* v_fst_647_; lean_object* v_snd_648_; lean_object* v___x_650_; uint8_t v_isShared_651_; uint8_t v_isSharedCheck_657_; 
v_toApplicative_646_ = lean_ctor_get(v_inst_644_, 0);
lean_inc_ref(v_toApplicative_646_);
lean_dec_ref(v_inst_644_);
v_fst_647_ = lean_ctor_get(v_____x_645_, 0);
v_snd_648_ = lean_ctor_get(v_____x_645_, 1);
v_isSharedCheck_657_ = !lean_is_exclusive(v_____x_645_);
if (v_isSharedCheck_657_ == 0)
{
v___x_650_ = v_____x_645_;
v_isShared_651_ = v_isSharedCheck_657_;
goto v_resetjp_649_;
}
else
{
lean_inc(v_snd_648_);
lean_inc(v_fst_647_);
lean_dec(v_____x_645_);
v___x_650_ = lean_box(0);
v_isShared_651_ = v_isSharedCheck_657_;
goto v_resetjp_649_;
}
v_resetjp_649_:
{
lean_object* v_toPure_652_; lean_object* v___x_654_; 
v_toPure_652_ = lean_ctor_get(v_toApplicative_646_, 1);
lean_inc(v_toPure_652_);
lean_dec_ref(v_toApplicative_646_);
if (v_isShared_651_ == 0)
{
v___x_654_ = v___x_650_;
goto v_reusejp_653_;
}
else
{
lean_object* v_reuseFailAlloc_656_; 
v_reuseFailAlloc_656_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_656_, 0, v_fst_647_);
lean_ctor_set(v_reuseFailAlloc_656_, 1, v_snd_648_);
v___x_654_ = v_reuseFailAlloc_656_;
goto v_reusejp_653_;
}
v_reusejp_653_:
{
lean_object* v___x_655_; 
v___x_655_ = lean_apply_2(v_toPure_652_, lean_box(0), v___x_654_);
return v___x_655_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomFin___redArg___lam__1(lean_object* v_inst_658_, lean_object* v___f_659_, lean_object* v_g_660_, lean_object* v_lo_661_, lean_object* v_hi_662_, lean_object* v_h_663_, lean_object* v_x_664_, lean_object* v___y_665_){
_start:
{
lean_object* v_toBind_666_; lean_object* v___x_667_; lean_object* v___x_668_; 
v_toBind_666_ = lean_ctor_get(v_inst_658_, 1);
lean_inc(v_toBind_666_);
v___x_667_ = lp_plausible_Plausible_Random_instBoundedRandomNat___redArg___lam__1(v_inst_658_, lean_box(0), v_lo_661_, v_hi_662_, lean_box(0), v_x_664_, v___y_665_);
v___x_668_ = lean_apply_4(v_toBind_666_, lean_box(0), lean_box(0), v___x_667_, v___f_659_);
return v___x_668_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomFin___redArg___lam__1___boxed(lean_object* v_inst_669_, lean_object* v___f_670_, lean_object* v_g_671_, lean_object* v_lo_672_, lean_object* v_hi_673_, lean_object* v_h_674_, lean_object* v_x_675_, lean_object* v___y_676_){
_start:
{
lean_object* v_res_677_; 
v_res_677_ = lp_plausible_Plausible_Random_instBoundedRandomFin___redArg___lam__1(v_inst_669_, v___f_670_, v_g_671_, v_lo_672_, v_hi_673_, v_h_674_, v_x_675_, v___y_676_);
lean_dec(v_hi_673_);
return v_res_677_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomFin___redArg(lean_object* v_inst_678_){
_start:
{
lean_object* v___f_679_; lean_object* v___f_680_; 
lean_inc_ref(v_inst_678_);
v___f_679_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Random_instBoundedRandomFin___redArg___lam__0), 2, 1);
lean_closure_set(v___f_679_, 0, v_inst_678_);
v___f_680_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Random_instBoundedRandomFin___redArg___lam__1___boxed), 8, 2);
lean_closure_set(v___f_680_, 0, v_inst_678_);
lean_closure_set(v___f_680_, 1, v___f_679_);
return v___f_680_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomFin(lean_object* v_m_681_, lean_object* v_inst_682_, lean_object* v_n_683_){
_start:
{
lean_object* v___x_684_; 
v___x_684_ = lp_plausible_Plausible_Random_instBoundedRandomFin___redArg(v_inst_682_);
return v___x_684_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomFin___boxed(lean_object* v_m_685_, lean_object* v_inst_686_, lean_object* v_n_687_){
_start:
{
lean_object* v_res_688_; 
v_res_688_ = lp_plausible_Plausible_Random_instBoundedRandomFin(v_m_685_, v_inst_686_, v_n_687_);
lean_dec(v_n_687_);
return v_res_688_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomBitVec___redArg(lean_object* v_inst_689_){
_start:
{
lean_object* v___f_690_; lean_object* v___f_691_; 
lean_inc_ref(v_inst_689_);
v___f_690_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Random_instBoundedRandomFin___redArg___lam__0), 2, 1);
lean_closure_set(v___f_690_, 0, v_inst_689_);
v___f_691_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Random_instBoundedRandomFin___redArg___lam__1___boxed), 8, 2);
lean_closure_set(v___f_691_, 0, v_inst_689_);
lean_closure_set(v___f_691_, 1, v___f_690_);
return v___f_691_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomBitVec(lean_object* v_m_692_, lean_object* v_inst_693_, lean_object* v_n_694_){
_start:
{
lean_object* v___x_695_; 
v___x_695_ = lp_plausible_Plausible_Random_instBoundedRandomBitVec___redArg(v_inst_693_);
return v___x_695_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_instBoundedRandomBitVec___boxed(lean_object* v_m_696_, lean_object* v_inst_697_, lean_object* v_n_698_){
_start:
{
lean_object* v_res_699_; 
v_res_699_ = lp_plausible_Plausible_Random_instBoundedRandomBitVec(v_m_696_, v_inst_697_, v_n_698_);
lean_dec(v_n_698_);
return v_res_699_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_runRand___redArg___lam__0(lean_object* v_toPure_700_, lean_object* v_fst_701_, lean_object* v_____x_702_){
_start:
{
lean_object* v___x_703_; 
v___x_703_ = lean_apply_2(v_toPure_700_, lean_box(0), v_fst_701_);
return v___x_703_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_runRand___redArg___lam__1(lean_object* v_toPure_704_, lean_object* v___x_705_, lean_object* v_inst_706_, lean_object* v_toBind_707_, lean_object* v_____x_708_){
_start:
{
lean_object* v_fst_709_; lean_object* v_snd_710_; lean_object* v___f_711_; lean_object* v___x_712_; lean_object* v___x_713_; lean_object* v___x_714_; 
v_fst_709_ = lean_ctor_get(v_____x_708_, 0);
lean_inc(v_fst_709_);
v_snd_710_ = lean_ctor_get(v_____x_708_, 1);
lean_inc(v_snd_710_);
lean_dec_ref(v_____x_708_);
v___f_711_ = lean_alloc_closure((void*)(lp_plausible_Plausible_runRand___redArg___lam__0), 3, 2);
lean_closure_set(v___f_711_, 0, v_toPure_704_);
lean_closure_set(v___f_711_, 1, v_fst_709_);
v___x_712_ = lean_alloc_closure((void*)(l_ST_Prim_Ref_set___boxed), 5, 4);
lean_closure_set(v___x_712_, 0, lean_box(0));
lean_closure_set(v___x_712_, 1, lean_box(0));
lean_closure_set(v___x_712_, 2, v___x_705_);
lean_closure_set(v___x_712_, 3, v_snd_710_);
v___x_713_ = lean_apply_2(v_inst_706_, lean_box(0), v___x_712_);
v___x_714_ = lean_apply_4(v_toBind_707_, lean_box(0), lean_box(0), v___x_713_, v___f_711_);
return v___x_714_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_runRand___redArg___lam__2(lean_object* v_cmd_715_, lean_object* v_toBind_716_, lean_object* v___f_717_, lean_object* v_stdGen_718_){
_start:
{
lean_object* v___x_719_; lean_object* v___x_720_; 
v___x_719_ = lean_apply_1(v_cmd_715_, v_stdGen_718_);
v___x_720_ = lean_apply_4(v_toBind_716_, lean_box(0), lean_box(0), v___x_719_, v___f_717_);
return v___x_720_;
}
}
static lean_object* _init_lp_plausible_Plausible_runRand___redArg___closed__0(void){
_start:
{
lean_object* v___x_721_; lean_object* v___x_722_; 
v___x_721_ = l_IO_stdGenRef;
v___x_722_ = lean_alloc_closure((void*)(l_ST_Prim_Ref_get___boxed), 4, 3);
lean_closure_set(v___x_722_, 0, lean_box(0));
lean_closure_set(v___x_722_, 1, lean_box(0));
lean_closure_set(v___x_722_, 2, v___x_721_);
return v___x_722_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_runRand___redArg(lean_object* v_inst_723_, lean_object* v_inst_724_, lean_object* v_cmd_725_){
_start:
{
lean_object* v_toApplicative_726_; lean_object* v_toBind_727_; lean_object* v_toPure_728_; lean_object* v___x_729_; lean_object* v___x_730_; lean_object* v___x_731_; lean_object* v___f_732_; lean_object* v___f_733_; lean_object* v___x_734_; 
v_toApplicative_726_ = lean_ctor_get(v_inst_723_, 0);
lean_inc_ref(v_toApplicative_726_);
v_toBind_727_ = lean_ctor_get(v_inst_723_, 1);
lean_inc_n(v_toBind_727_, 3);
lean_dec_ref(v_inst_723_);
v_toPure_728_ = lean_ctor_get(v_toApplicative_726_, 1);
lean_inc(v_toPure_728_);
lean_dec_ref(v_toApplicative_726_);
v___x_729_ = l_IO_stdGenRef;
v___x_730_ = lean_obj_once(&lp_plausible_Plausible_runRand___redArg___closed__0, &lp_plausible_Plausible_runRand___redArg___closed__0_once, _init_lp_plausible_Plausible_runRand___redArg___closed__0);
lean_inc(v_inst_724_);
v___x_731_ = lean_apply_2(v_inst_724_, lean_box(0), v___x_730_);
v___f_732_ = lean_alloc_closure((void*)(lp_plausible_Plausible_runRand___redArg___lam__1), 5, 4);
lean_closure_set(v___f_732_, 0, v_toPure_728_);
lean_closure_set(v___f_732_, 1, v___x_729_);
lean_closure_set(v___f_732_, 2, v_inst_724_);
lean_closure_set(v___f_732_, 3, v_toBind_727_);
v___f_733_ = lean_alloc_closure((void*)(lp_plausible_Plausible_runRand___redArg___lam__2), 4, 3);
lean_closure_set(v___f_733_, 0, v_cmd_725_);
lean_closure_set(v___f_733_, 1, v_toBind_727_);
lean_closure_set(v___f_733_, 2, v___f_732_);
v___x_734_ = lean_apply_4(v_toBind_727_, lean_box(0), lean_box(0), v___x_731_, v___f_733_);
return v___x_734_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_runRand(lean_object* v_m_735_, lean_object* v_inst_736_, lean_object* v_inst_737_, lean_object* v_00_u03b1_738_, lean_object* v_cmd_739_){
_start:
{
lean_object* v___x_740_; 
v___x_740_ = lp_plausible_Plausible_runRand___redArg(v_inst_736_, v_inst_737_, v_cmd_739_);
return v___x_740_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_runRandWith___redArg___lam__0(lean_object* v_toPure_741_, lean_object* v_____do__lift_742_){
_start:
{
lean_object* v_fst_743_; lean_object* v___x_744_; 
v_fst_743_ = lean_ctor_get(v_____do__lift_742_, 0);
lean_inc(v_fst_743_);
lean_dec_ref(v_____do__lift_742_);
v___x_744_ = lean_apply_2(v_toPure_741_, lean_box(0), v_fst_743_);
return v___x_744_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_runRandWith___redArg(lean_object* v_inst_745_, lean_object* v_seed_746_, lean_object* v_cmd_747_){
_start:
{
lean_object* v_toApplicative_748_; lean_object* v_toBind_749_; lean_object* v_toPure_750_; lean_object* v___x_751_; lean_object* v___x_752_; lean_object* v___f_753_; lean_object* v___x_754_; 
v_toApplicative_748_ = lean_ctor_get(v_inst_745_, 0);
lean_inc_ref(v_toApplicative_748_);
v_toBind_749_ = lean_ctor_get(v_inst_745_, 1);
lean_inc(v_toBind_749_);
lean_dec_ref(v_inst_745_);
v_toPure_750_ = lean_ctor_get(v_toApplicative_748_, 1);
lean_inc(v_toPure_750_);
lean_dec_ref(v_toApplicative_748_);
v___x_751_ = l_mkStdGen(v_seed_746_);
v___x_752_ = lean_apply_1(v_cmd_747_, v___x_751_);
v___f_753_ = lean_alloc_closure((void*)(lp_plausible_Plausible_runRandWith___redArg___lam__0), 2, 1);
lean_closure_set(v___f_753_, 0, v_toPure_750_);
v___x_754_ = lean_apply_4(v_toBind_749_, lean_box(0), lean_box(0), v___x_752_, v___f_753_);
return v___x_754_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_runRandWith___redArg___boxed(lean_object* v_inst_755_, lean_object* v_seed_756_, lean_object* v_cmd_757_){
_start:
{
lean_object* v_res_758_; 
v_res_758_ = lp_plausible_Plausible_runRandWith___redArg(v_inst_755_, v_seed_756_, v_cmd_757_);
lean_dec(v_seed_756_);
return v_res_758_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_runRandWith(lean_object* v_m_759_, lean_object* v_inst_760_, lean_object* v_00_u03b1_761_, lean_object* v_seed_762_, lean_object* v_cmd_763_){
_start:
{
lean_object* v___x_764_; 
v___x_764_ = lp_plausible_Plausible_runRandWith___redArg(v_inst_760_, v_seed_762_, v_cmd_763_);
return v___x_764_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_runRandWith___boxed(lean_object* v_m_765_, lean_object* v_inst_766_, lean_object* v_00_u03b1_767_, lean_object* v_seed_768_, lean_object* v_cmd_769_){
_start:
{
lean_object* v_res_770_; 
v_res_770_ = lp_plausible_Plausible_runRandWith(v_m_765_, v_inst_766_, v_00_u03b1_767_, v_seed_768_, v_cmd_769_);
lean_dec(v_seed_768_);
return v_res_770_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_plausible_Plausible_Random(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_plausible_Plausible_Random(uint8_t builtin) {
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
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_plausible_Plausible_Random(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_plausible_Plausible_Random(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_plausible_Plausible_Random(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_plausible_Plausible_Random(builtin);
}
#ifdef __cplusplus
}
#endif
