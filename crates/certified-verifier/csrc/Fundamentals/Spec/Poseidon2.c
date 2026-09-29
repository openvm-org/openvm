// Lean compiler output
// Module: Fundamentals.Spec.Poseidon2
// Imports: public import Init public meta import Init public import Fundamentals.Spec.BabyBear public import Fundamentals.Spec.LawfulFieldOps public import Fundamentals.Spec.Poseidon2.Generic public import Fundamentals.Spec.Poseidon2.Sponge
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
extern lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat;
extern lean_object* lp_swirl_x2dfv_Fundamentals_BabyBear_fbbFieldOps;
lean_object* l_List_reverse___redArg(lean_object*);
extern lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat;
lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateOfNats___redArg(lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_divByTwoPow___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRounds___redArg(lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_permute___redArg(lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalRound___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_initialExternalRounds___redArg(lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_applySBoxToAll___redArg(lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalLinearLayer___redArg(lean_object*, lean_object*);
extern lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat;
lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_finalExternalRounds___redArg(lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_sbox___redArg(lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRound___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalLinearLayer___redArg(lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_addRoundConstants___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_applyMat4___redArg(lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_compressWithCapacity___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_WIDTH;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_RATE;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_sbox(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_addRoundConstants(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_applySBoxToAll(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_applyMat4(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_externalLinearLayer(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_divByTwoPow(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_internalLinearLayer(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_externalRound(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_internalRound(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_Poseidon2_babyBearRc16ExternalInitial_spec__0(lean_object*, lean_object*);
static lean_once_cell_t lp_swirl_x2dfv_Fundamentals_Poseidon2_babyBearRc16ExternalInitial___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_babyBearRc16ExternalInitial___closed__0;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_babyBearRc16ExternalInitial;
static lean_once_cell_t lp_swirl_x2dfv_Fundamentals_Poseidon2_babyBearRc16ExternalFinal___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_babyBearRc16ExternalFinal___closed__0;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_babyBearRc16ExternalFinal;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_Poseidon2_babyBearRc16Internal_spec__0(lean_object*, lean_object*);
static lean_once_cell_t lp_swirl_x2dfv_Fundamentals_Poseidon2_babyBearRc16Internal___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_babyBearRc16Internal___closed__0;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_babyBearRc16Internal;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_initialExternalRounds(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_internalRounds(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_finalExternalRounds(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_permute(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_compressWithCapacity(lean_object*, lean_object*);
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_Poseidon2_WIDTH(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lean_unsigned_to_nat(16u);
return v___x_1_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_Poseidon2_RATE(void){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(8u);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_sbox(lean_object* v_x_3_){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; 
v___x_4_ = lp_swirl_x2dfv_Fundamentals_BabyBear_fbbFieldOps;
v___x_5_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_sbox___redArg(v___x_4_, v_x_3_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_addRoundConstants(lean_object* v_rc_6_, lean_object* v_state_7_){
_start:
{
lean_object* v___x_8_; lean_object* v___x_9_; 
v___x_8_ = lp_swirl_x2dfv_Fundamentals_BabyBear_fbbFieldOps;
v___x_9_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_addRoundConstants___redArg(v___x_8_, v_rc_6_, v_state_7_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_applySBoxToAll(lean_object* v_state_10_){
_start:
{
lean_object* v___x_11_; lean_object* v___x_12_; 
v___x_11_ = lp_swirl_x2dfv_Fundamentals_BabyBear_fbbFieldOps;
v___x_12_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_applySBoxToAll___redArg(v___x_11_, v_state_10_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_applyMat4(lean_object* v_x_13_){
_start:
{
lean_object* v___x_14_; lean_object* v___x_15_; 
v___x_14_ = lp_swirl_x2dfv_Fundamentals_BabyBear_fbbFieldOps;
v___x_15_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_applyMat4___redArg(v___x_14_, v_x_13_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_externalLinearLayer(lean_object* v_state_16_){
_start:
{
lean_object* v___x_17_; lean_object* v___x_18_; 
v___x_17_ = lp_swirl_x2dfv_Fundamentals_BabyBear_fbbFieldOps;
v___x_18_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalLinearLayer___redArg(v___x_17_, v_state_16_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_divByTwoPow(lean_object* v_k_19_, lean_object* v_x_20_){
_start:
{
lean_object* v___x_21_; lean_object* v___x_22_; 
v___x_21_ = lp_swirl_x2dfv_Fundamentals_BabyBear_fbbFieldOps;
v___x_22_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_divByTwoPow___redArg(v___x_21_, v_k_19_, v_x_20_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_internalLinearLayer(lean_object* v_state_23_){
_start:
{
lean_object* v___x_24_; lean_object* v___x_25_; 
v___x_24_ = lp_swirl_x2dfv_Fundamentals_BabyBear_fbbFieldOps;
v___x_25_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalLinearLayer___redArg(v___x_24_, v_state_23_);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_externalRound(lean_object* v_rc_26_, lean_object* v_state_27_){
_start:
{
lean_object* v___x_28_; lean_object* v___x_29_; 
v___x_28_ = lp_swirl_x2dfv_Fundamentals_BabyBear_fbbFieldOps;
v___x_29_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalRound___redArg(v___x_28_, v_rc_26_, v_state_27_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_internalRound(lean_object* v_rc_30_, lean_object* v_state_31_){
_start:
{
lean_object* v___x_32_; lean_object* v___x_33_; 
v___x_32_ = lp_swirl_x2dfv_Fundamentals_BabyBear_fbbFieldOps;
v___x_33_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRound___redArg(v___x_32_, v_rc_30_, v_state_31_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_Poseidon2_babyBearRc16ExternalInitial_spec__0(lean_object* v_a_34_, lean_object* v_a_35_){
_start:
{
if (lean_obj_tag(v_a_34_) == 0)
{
lean_object* v___x_36_; 
v___x_36_ = l_List_reverse___redArg(v_a_35_);
return v___x_36_;
}
else
{
lean_object* v_head_37_; lean_object* v_tail_38_; lean_object* v___x_40_; uint8_t v_isShared_41_; uint8_t v_isSharedCheck_48_; 
v_head_37_ = lean_ctor_get(v_a_34_, 0);
v_tail_38_ = lean_ctor_get(v_a_34_, 1);
v_isSharedCheck_48_ = !lean_is_exclusive(v_a_34_);
if (v_isSharedCheck_48_ == 0)
{
v___x_40_ = v_a_34_;
v_isShared_41_ = v_isSharedCheck_48_;
goto v_resetjp_39_;
}
else
{
lean_inc(v_tail_38_);
lean_inc(v_head_37_);
lean_dec(v_a_34_);
v___x_40_ = lean_box(0);
v_isShared_41_ = v_isSharedCheck_48_;
goto v_resetjp_39_;
}
v_resetjp_39_:
{
lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___x_45_; 
v___x_42_ = lp_swirl_x2dfv_Fundamentals_BabyBear_fbbFieldOps;
v___x_43_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateOfNats___redArg(v___x_42_, v_head_37_);
if (v_isShared_41_ == 0)
{
lean_ctor_set(v___x_40_, 1, v_a_35_);
lean_ctor_set(v___x_40_, 0, v___x_43_);
v___x_45_ = v___x_40_;
goto v_reusejp_44_;
}
else
{
lean_object* v_reuseFailAlloc_47_; 
v_reuseFailAlloc_47_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_47_, 0, v___x_43_);
lean_ctor_set(v_reuseFailAlloc_47_, 1, v_a_35_);
v___x_45_ = v_reuseFailAlloc_47_;
goto v_reusejp_44_;
}
v_reusejp_44_:
{
v_a_34_ = v_tail_38_;
v_a_35_ = v___x_45_;
goto _start;
}
}
}
}
}
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_Poseidon2_babyBearRc16ExternalInitial___closed__0(void){
_start:
{
lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; 
v___x_49_ = lean_box(0);
v___x_50_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat;
v___x_51_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_Poseidon2_babyBearRc16ExternalInitial_spec__0(v___x_50_, v___x_49_);
return v___x_51_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_Poseidon2_babyBearRc16ExternalInitial(void){
_start:
{
lean_object* v___x_52_; 
v___x_52_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_Poseidon2_babyBearRc16ExternalInitial___closed__0, &lp_swirl_x2dfv_Fundamentals_Poseidon2_babyBearRc16ExternalInitial___closed__0_once, _init_lp_swirl_x2dfv_Fundamentals_Poseidon2_babyBearRc16ExternalInitial___closed__0);
return v___x_52_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_Poseidon2_babyBearRc16ExternalFinal___closed__0(void){
_start:
{
lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; 
v___x_53_ = lean_box(0);
v___x_54_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat;
v___x_55_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_Poseidon2_babyBearRc16ExternalInitial_spec__0(v___x_54_, v___x_53_);
return v___x_55_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_Poseidon2_babyBearRc16ExternalFinal(void){
_start:
{
lean_object* v___x_56_; 
v___x_56_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_Poseidon2_babyBearRc16ExternalFinal___closed__0, &lp_swirl_x2dfv_Fundamentals_Poseidon2_babyBearRc16ExternalFinal___closed__0_once, _init_lp_swirl_x2dfv_Fundamentals_Poseidon2_babyBearRc16ExternalFinal___closed__0);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_Poseidon2_babyBearRc16Internal_spec__0(lean_object* v_a_57_, lean_object* v_a_58_){
_start:
{
lean_object* v___x_59_; 
v___x_59_ = lp_swirl_x2dfv_Fundamentals_BabyBear_fbbFieldOps;
if (lean_obj_tag(v_a_57_) == 0)
{
lean_object* v___x_60_; 
v___x_60_ = l_List_reverse___redArg(v_a_58_);
return v___x_60_;
}
else
{
lean_object* v_toRingOps_61_; lean_object* v_toSemiringOps_62_; lean_object* v_natCast_63_; lean_object* v_head_64_; lean_object* v_tail_65_; lean_object* v___x_67_; uint8_t v_isShared_68_; uint8_t v_isSharedCheck_74_; 
v_toRingOps_61_ = lean_ctor_get(v___x_59_, 0);
v_toSemiringOps_62_ = lean_ctor_get(v_toRingOps_61_, 0);
v_natCast_63_ = lean_ctor_get(v_toSemiringOps_62_, 2);
v_head_64_ = lean_ctor_get(v_a_57_, 0);
v_tail_65_ = lean_ctor_get(v_a_57_, 1);
v_isSharedCheck_74_ = !lean_is_exclusive(v_a_57_);
if (v_isSharedCheck_74_ == 0)
{
v___x_67_ = v_a_57_;
v_isShared_68_ = v_isSharedCheck_74_;
goto v_resetjp_66_;
}
else
{
lean_inc(v_tail_65_);
lean_inc(v_head_64_);
lean_dec(v_a_57_);
v___x_67_ = lean_box(0);
v_isShared_68_ = v_isSharedCheck_74_;
goto v_resetjp_66_;
}
v_resetjp_66_:
{
lean_object* v___x_69_; lean_object* v___x_71_; 
lean_inc(v_natCast_63_);
v___x_69_ = lean_apply_1(v_natCast_63_, v_head_64_);
if (v_isShared_68_ == 0)
{
lean_ctor_set(v___x_67_, 1, v_a_58_);
lean_ctor_set(v___x_67_, 0, v___x_69_);
v___x_71_ = v___x_67_;
goto v_reusejp_70_;
}
else
{
lean_object* v_reuseFailAlloc_73_; 
v_reuseFailAlloc_73_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_73_, 0, v___x_69_);
lean_ctor_set(v_reuseFailAlloc_73_, 1, v_a_58_);
v___x_71_ = v_reuseFailAlloc_73_;
goto v_reusejp_70_;
}
v_reusejp_70_:
{
v_a_57_ = v_tail_65_;
v_a_58_ = v___x_71_;
goto _start;
}
}
}
}
}
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_Poseidon2_babyBearRc16Internal___closed__0(void){
_start:
{
lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; 
v___x_75_ = lean_box(0);
v___x_76_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat;
v___x_77_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_Poseidon2_babyBearRc16Internal_spec__0(v___x_76_, v___x_75_);
return v___x_77_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_Poseidon2_babyBearRc16Internal(void){
_start:
{
lean_object* v___x_78_; 
v___x_78_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_Poseidon2_babyBearRc16Internal___closed__0, &lp_swirl_x2dfv_Fundamentals_Poseidon2_babyBearRc16Internal___closed__0_once, _init_lp_swirl_x2dfv_Fundamentals_Poseidon2_babyBearRc16Internal___closed__0);
return v___x_78_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_initialExternalRounds(lean_object* v_state_79_){
_start:
{
lean_object* v___x_80_; lean_object* v___x_81_; 
v___x_80_ = lp_swirl_x2dfv_Fundamentals_BabyBear_fbbFieldOps;
v___x_81_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_initialExternalRounds___redArg(v___x_80_, v_state_79_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_internalRounds(lean_object* v_state_82_){
_start:
{
lean_object* v___x_83_; lean_object* v___x_84_; 
v___x_83_ = lp_swirl_x2dfv_Fundamentals_BabyBear_fbbFieldOps;
v___x_84_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRounds___redArg(v___x_83_, v_state_82_);
return v___x_84_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_finalExternalRounds(lean_object* v_state_85_){
_start:
{
lean_object* v___x_86_; lean_object* v___x_87_; 
v___x_86_ = lp_swirl_x2dfv_Fundamentals_BabyBear_fbbFieldOps;
v___x_87_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_finalExternalRounds___redArg(v___x_86_, v_state_85_);
return v___x_87_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_permute(lean_object* v_state_88_){
_start:
{
lean_object* v___x_89_; lean_object* v___x_90_; 
v___x_89_ = lp_swirl_x2dfv_Fundamentals_BabyBear_fbbFieldOps;
v___x_90_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_permute___redArg(v___x_89_, v_state_88_);
return v___x_90_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_compressWithCapacity(lean_object* v_left_91_, lean_object* v_right_92_){
_start:
{
lean_object* v___x_93_; lean_object* v___x_94_; 
v___x_93_ = lp_swirl_x2dfv_Fundamentals_BabyBear_fbbFieldOps;
v___x_94_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_compressWithCapacity___redArg(v___x_93_, v_left_91_, v_right_92_);
return v___x_94_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_swirl_x2dfv_Fundamentals_Spec_BabyBear(uint8_t builtin);
lean_object* initialize_swirl_x2dfv_Fundamentals_Spec_LawfulFieldOps(uint8_t builtin);
lean_object* initialize_swirl_x2dfv_Fundamentals_Spec_Poseidon2_Generic(uint8_t builtin);
lean_object* initialize_swirl_x2dfv_Fundamentals_Spec_Poseidon2_Sponge(uint8_t builtin);
void lean_initialize();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_swirl_x2dfv_Fundamentals_Spec_Poseidon2(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
lean_initialize();
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2dfv_Fundamentals_Spec_BabyBear(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2dfv_Fundamentals_Spec_LawfulFieldOps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2dfv_Fundamentals_Spec_Poseidon2_Generic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2dfv_Fundamentals_Spec_Poseidon2_Sponge(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_swirl_x2dfv_Fundamentals_Poseidon2_WIDTH = _init_lp_swirl_x2dfv_Fundamentals_Poseidon2_WIDTH();
lean_mark_persistent(lp_swirl_x2dfv_Fundamentals_Poseidon2_WIDTH);
lp_swirl_x2dfv_Fundamentals_Poseidon2_RATE = _init_lp_swirl_x2dfv_Fundamentals_Poseidon2_RATE();
lean_mark_persistent(lp_swirl_x2dfv_Fundamentals_Poseidon2_RATE);
lp_swirl_x2dfv_Fundamentals_Poseidon2_babyBearRc16ExternalInitial = _init_lp_swirl_x2dfv_Fundamentals_Poseidon2_babyBearRc16ExternalInitial();
lean_mark_persistent(lp_swirl_x2dfv_Fundamentals_Poseidon2_babyBearRc16ExternalInitial);
lp_swirl_x2dfv_Fundamentals_Poseidon2_babyBearRc16ExternalFinal = _init_lp_swirl_x2dfv_Fundamentals_Poseidon2_babyBearRc16ExternalFinal();
lean_mark_persistent(lp_swirl_x2dfv_Fundamentals_Poseidon2_babyBearRc16ExternalFinal);
lp_swirl_x2dfv_Fundamentals_Poseidon2_babyBearRc16Internal = _init_lp_swirl_x2dfv_Fundamentals_Poseidon2_babyBearRc16Internal();
lean_mark_persistent(lp_swirl_x2dfv_Fundamentals_Poseidon2_babyBearRc16Internal);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
