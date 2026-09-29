// Lean compiler output
// Module: Mathlib.Algebra.Order.Nonneg.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Order.GroupWithZero.Basic public import Mathlib.Algebra.Order.Monoid.Unbundled.Pow public import Mathlib.Algebra.Order.ZeroLEOne public import Mathlib.Algebra.Ring.Defs public import Mathlib.Algebra.Ring.InjSurj public import Mathlib.Data.Nat.Cast.Order.Basic
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
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_NSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_instMulZeroClassOfSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toMonoidWithZero___redArg(lean_object*);
lean_object* lp_mathlib_NPow_ofPow___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CommSemiring_toCommMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_inhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_inhabited___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_inhabited(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_inhabited___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_zero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_zero___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_zero(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_zero___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_add___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_add___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_add(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_add___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_nsmul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_nsmul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_nsmul(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_nsmul___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_one___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_one___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_one(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_one___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_mul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_mul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_mul(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_mul___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_coeAddMonoidHom___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_coeAddMonoidHom___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Nonneg_coeAddMonoidHom___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Nonneg_coeAddMonoidHom___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Nonneg_coeAddMonoidHom___closed__0 = (const lean_object*)&lp_mathlib_Nonneg_coeAddMonoidHom___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_coeAddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_coeAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addCommMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addCancelCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addCancelCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addCancelCommMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_natCast___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_natCast___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_natCast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_natCast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addMonoidWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addMonoidWithOne(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addMonoidWithOne___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_pow___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_pow___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_pow(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_pow___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_semiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_semiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_semiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_monoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_monoidWithZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_monoidWithZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_coeRingHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_coeRingHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_commSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_commSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_commSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_commMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_commMonoidWithZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_commMonoidWithZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_toNonneg___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_toNonneg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_sub___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_sub___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_sub(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_inhabited___redArg(lean_object* v_a_1_){
_start:
{
lean_inc(v_a_1_);
return v_a_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_inhabited___redArg___boxed(lean_object* v_a_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_Nonneg_inhabited___redArg(v_a_2_);
lean_dec(v_a_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_inhabited(lean_object* v_00_u03b1_4_, lean_object* v_inst_5_, lean_object* v_a_6_){
_start:
{
lean_inc(v_a_6_);
return v_a_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_inhabited___boxed(lean_object* v_00_u03b1_7_, lean_object* v_inst_8_, lean_object* v_a_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_mathlib_Nonneg_inhabited(v_00_u03b1_7_, v_inst_8_, v_a_9_);
lean_dec(v_a_9_);
lean_dec_ref(v_inst_8_);
return v_res_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_zero___redArg(lean_object* v_inst_11_){
_start:
{
lean_inc(v_inst_11_);
return v_inst_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_zero___redArg___boxed(lean_object* v_inst_12_){
_start:
{
lean_object* v_res_13_; 
v_res_13_ = lp_mathlib_Nonneg_zero___redArg(v_inst_12_);
lean_dec(v_inst_12_);
return v_res_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_zero(lean_object* v_00_u03b1_14_, lean_object* v_inst_15_, lean_object* v_inst_16_){
_start:
{
lean_inc(v_inst_15_);
return v_inst_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_zero___boxed(lean_object* v_00_u03b1_17_, lean_object* v_inst_18_, lean_object* v_inst_19_){
_start:
{
lean_object* v_res_20_; 
v_res_20_ = lp_mathlib_Nonneg_zero(v_00_u03b1_17_, v_inst_18_, v_inst_19_);
lean_dec_ref(v_inst_19_);
lean_dec(v_inst_18_);
return v_res_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_add___redArg___lam__0(lean_object* v_toAdd_21_, lean_object* v_x_22_, lean_object* v_y_23_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = lean_apply_2(v_toAdd_21_, v_x_22_, v_y_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_add___redArg(lean_object* v_inst_25_){
_start:
{
lean_object* v___x_26_; lean_object* v_toAdd_27_; lean_object* v___f_28_; 
v___x_26_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_25_);
v_toAdd_27_ = lean_ctor_get(v___x_26_, 1);
lean_inc(v_toAdd_27_);
lean_dec_ref(v___x_26_);
v___f_28_ = lean_alloc_closure((void*)(lp_mathlib_Nonneg_add___redArg___lam__0), 3, 1);
lean_closure_set(v___f_28_, 0, v_toAdd_27_);
return v___f_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_add(lean_object* v_00_u03b1_29_, lean_object* v_inst_30_, lean_object* v_inst_31_, lean_object* v_inst_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = lp_mathlib_Nonneg_add___redArg(v_inst_30_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_add___boxed(lean_object* v_00_u03b1_34_, lean_object* v_inst_35_, lean_object* v_inst_36_, lean_object* v_inst_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_mathlib_Nonneg_add(v_00_u03b1_34_, v_inst_35_, v_inst_36_, v_inst_37_);
lean_dec_ref(v_inst_36_);
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_nsmul___redArg___lam__0(lean_object* v_toNSMul_39_, lean_object* v_n_40_, lean_object* v_x_41_){
_start:
{
lean_object* v___x_42_; 
v___x_42_ = lean_apply_2(v_toNSMul_39_, v_n_40_, v_x_41_);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_nsmul___redArg(lean_object* v_inst_43_){
_start:
{
lean_object* v_toNSMul_44_; lean_object* v___f_45_; 
v_toNSMul_44_ = lean_ctor_get(v_inst_43_, 2);
lean_inc(v_toNSMul_44_);
lean_dec_ref(v_inst_43_);
v___f_45_ = lean_alloc_closure((void*)(lp_mathlib_Nonneg_nsmul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_45_, 0, v_toNSMul_44_);
return v___f_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_nsmul(lean_object* v_00_u03b1_46_, lean_object* v_inst_47_, lean_object* v_inst_48_, lean_object* v_inst_49_){
_start:
{
lean_object* v___x_50_; 
v___x_50_ = lp_mathlib_Nonneg_nsmul___redArg(v_inst_47_);
return v___x_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_nsmul___boxed(lean_object* v_00_u03b1_51_, lean_object* v_inst_52_, lean_object* v_inst_53_, lean_object* v_inst_54_){
_start:
{
lean_object* v_res_55_; 
v_res_55_ = lp_mathlib_Nonneg_nsmul(v_00_u03b1_51_, v_inst_52_, v_inst_53_, v_inst_54_);
lean_dec_ref(v_inst_53_);
return v_res_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_one___redArg(lean_object* v_inst_56_){
_start:
{
lean_inc(v_inst_56_);
return v_inst_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_one___redArg___boxed(lean_object* v_inst_57_){
_start:
{
lean_object* v_res_58_; 
v_res_58_ = lp_mathlib_Nonneg_one___redArg(v_inst_57_);
lean_dec(v_inst_57_);
return v_res_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_one(lean_object* v_00_u03b1_59_, lean_object* v_inst_60_, lean_object* v_inst_61_, lean_object* v_inst_62_, lean_object* v_inst_63_){
_start:
{
lean_inc(v_inst_61_);
return v_inst_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_one___boxed(lean_object* v_00_u03b1_64_, lean_object* v_inst_65_, lean_object* v_inst_66_, lean_object* v_inst_67_, lean_object* v_inst_68_){
_start:
{
lean_object* v_res_69_; 
v_res_69_ = lp_mathlib_Nonneg_one(v_00_u03b1_64_, v_inst_65_, v_inst_66_, v_inst_67_, v_inst_68_);
lean_dec(v_inst_66_);
lean_dec(v_inst_65_);
return v_res_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_mul___redArg___lam__0(lean_object* v_toMul_70_, lean_object* v_x_71_, lean_object* v_y_72_){
_start:
{
lean_object* v___x_73_; 
v___x_73_ = lean_apply_2(v_toMul_70_, v_x_71_, v_y_72_);
return v___x_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_mul___redArg(lean_object* v_inst_74_){
_start:
{
lean_object* v_toMul_75_; lean_object* v___f_76_; 
v_toMul_75_ = lean_ctor_get(v_inst_74_, 0);
lean_inc(v_toMul_75_);
lean_dec_ref(v_inst_74_);
v___f_76_ = lean_alloc_closure((void*)(lp_mathlib_Nonneg_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_76_, 0, v_toMul_75_);
return v___f_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_mul(lean_object* v_00_u03b1_77_, lean_object* v_inst_78_, lean_object* v_inst_79_, lean_object* v_inst_80_){
_start:
{
lean_object* v___x_81_; 
v___x_81_ = lp_mathlib_Nonneg_mul___redArg(v_inst_78_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_mul___boxed(lean_object* v_00_u03b1_82_, lean_object* v_inst_83_, lean_object* v_inst_84_, lean_object* v_inst_85_){
_start:
{
lean_object* v_res_86_; 
v_res_86_ = lp_mathlib_Nonneg_mul(v_00_u03b1_82_, v_inst_83_, v_inst_84_, v_inst_85_);
lean_dec_ref(v_inst_84_);
return v_res_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addMonoid___redArg(lean_object* v_inst_87_){
_start:
{
lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v_toZero_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___f_93_; lean_object* v___x_94_; 
v___x_88_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_87_);
lean_inc_ref(v___x_88_);
v___x_89_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_88_);
v_toZero_90_ = lean_ctor_get(v___x_89_, 0);
lean_inc(v_toZero_90_);
lean_dec_ref(v___x_89_);
v___x_91_ = lp_mathlib_Nonneg_add___redArg(v___x_88_);
v___x_92_ = lp_mathlib_Nonneg_nsmul___redArg(v_inst_87_);
v___f_93_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_93_, 0, v___x_92_);
v___x_94_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_94_, 0, v_toZero_90_);
lean_ctor_set(v___x_94_, 1, v___x_91_);
lean_ctor_set(v___x_94_, 2, v___f_93_);
return v___x_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addMonoid(lean_object* v_00_u03b1_95_, lean_object* v_inst_96_, lean_object* v_inst_97_, lean_object* v_inst_98_){
_start:
{
lean_object* v___x_99_; 
v___x_99_ = lp_mathlib_Nonneg_addMonoid___redArg(v_inst_96_);
return v___x_99_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addMonoid___boxed(lean_object* v_00_u03b1_100_, lean_object* v_inst_101_, lean_object* v_inst_102_, lean_object* v_inst_103_){
_start:
{
lean_object* v_res_104_; 
v_res_104_ = lp_mathlib_Nonneg_addMonoid(v_00_u03b1_100_, v_inst_101_, v_inst_102_, v_inst_103_);
lean_dec_ref(v_inst_102_);
return v_res_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_coeAddMonoidHom___lam__0(lean_object* v_self_105_){
_start:
{
lean_inc(v_self_105_);
return v_self_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_coeAddMonoidHom___lam__0___boxed(lean_object* v_self_106_){
_start:
{
lean_object* v_res_107_; 
v_res_107_ = lp_mathlib_Nonneg_coeAddMonoidHom___lam__0(v_self_106_);
lean_dec(v_self_106_);
return v_res_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_coeAddMonoidHom(lean_object* v_00_u03b1_109_, lean_object* v_inst_110_, lean_object* v_inst_111_, lean_object* v_inst_112_){
_start:
{
lean_object* v___f_113_; 
v___f_113_ = ((lean_object*)(lp_mathlib_Nonneg_coeAddMonoidHom___closed__0));
return v___f_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_coeAddMonoidHom___boxed(lean_object* v_00_u03b1_114_, lean_object* v_inst_115_, lean_object* v_inst_116_, lean_object* v_inst_117_){
_start:
{
lean_object* v_res_118_; 
v_res_118_ = lp_mathlib_Nonneg_coeAddMonoidHom(v_00_u03b1_114_, v_inst_115_, v_inst_116_, v_inst_117_);
lean_dec_ref(v_inst_116_);
lean_dec_ref(v_inst_115_);
return v_res_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addCommMonoid___redArg(lean_object* v_inst_119_){
_start:
{
lean_object* v___x_120_; 
v___x_120_ = lp_mathlib_Nonneg_addMonoid___redArg(v_inst_119_);
return v___x_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addCommMonoid(lean_object* v_00_u03b1_121_, lean_object* v_inst_122_, lean_object* v_inst_123_, lean_object* v_inst_124_){
_start:
{
lean_object* v___x_125_; 
v___x_125_ = lp_mathlib_Nonneg_addMonoid___redArg(v_inst_122_);
return v___x_125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addCommMonoid___boxed(lean_object* v_00_u03b1_126_, lean_object* v_inst_127_, lean_object* v_inst_128_, lean_object* v_inst_129_){
_start:
{
lean_object* v_res_130_; 
v_res_130_ = lp_mathlib_Nonneg_addCommMonoid(v_00_u03b1_126_, v_inst_127_, v_inst_128_, v_inst_129_);
lean_dec_ref(v_inst_128_);
return v_res_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addCancelCommMonoid___redArg(lean_object* v_inst_131_){
_start:
{
lean_object* v___x_132_; 
v___x_132_ = lp_mathlib_Nonneg_addMonoid___redArg(v_inst_131_);
return v___x_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addCancelCommMonoid(lean_object* v_00_u03b1_133_, lean_object* v_inst_134_, lean_object* v_inst_135_, lean_object* v_inst_136_){
_start:
{
lean_object* v___x_137_; 
v___x_137_ = lp_mathlib_Nonneg_addMonoid___redArg(v_inst_134_);
return v___x_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addCancelCommMonoid___boxed(lean_object* v_00_u03b1_138_, lean_object* v_inst_139_, lean_object* v_inst_140_, lean_object* v_inst_141_){
_start:
{
lean_object* v_res_142_; 
v_res_142_ = lp_mathlib_Nonneg_addCancelCommMonoid(v_00_u03b1_138_, v_inst_139_, v_inst_140_, v_inst_141_);
lean_dec_ref(v_inst_140_);
return v_res_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_natCast___redArg___lam__0(lean_object* v_toNatCast_143_, lean_object* v_n_144_){
_start:
{
lean_object* v___x_145_; 
v___x_145_ = lean_apply_1(v_toNatCast_143_, v_n_144_);
return v___x_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_natCast___redArg(lean_object* v_inst_146_){
_start:
{
lean_object* v_toNatCast_147_; lean_object* v___f_148_; 
v_toNatCast_147_ = lean_ctor_get(v_inst_146_, 0);
lean_inc(v_toNatCast_147_);
lean_dec_ref(v_inst_146_);
v___f_148_ = lean_alloc_closure((void*)(lp_mathlib_Nonneg_natCast___redArg___lam__0), 2, 1);
lean_closure_set(v___f_148_, 0, v_toNatCast_147_);
return v___f_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_natCast(lean_object* v_00_u03b1_149_, lean_object* v_inst_150_, lean_object* v_inst_151_, lean_object* v_inst_152_, lean_object* v_inst_153_){
_start:
{
lean_object* v___x_154_; 
v___x_154_ = lp_mathlib_Nonneg_natCast___redArg(v_inst_150_);
return v___x_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_natCast___boxed(lean_object* v_00_u03b1_155_, lean_object* v_inst_156_, lean_object* v_inst_157_, lean_object* v_inst_158_, lean_object* v_inst_159_){
_start:
{
lean_object* v_res_160_; 
v_res_160_ = lp_mathlib_Nonneg_natCast(v_00_u03b1_155_, v_inst_156_, v_inst_157_, v_inst_158_, v_inst_159_);
lean_dec_ref(v_inst_157_);
return v_res_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addMonoidWithOne___redArg(lean_object* v_inst_161_){
_start:
{
lean_object* v_toAddMonoid_162_; lean_object* v_toOne_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; 
v_toAddMonoid_162_ = lean_ctor_get(v_inst_161_, 1);
lean_inc_ref(v_toAddMonoid_162_);
v_toOne_163_ = lean_ctor_get(v_inst_161_, 2);
lean_inc(v_toOne_163_);
v___x_164_ = lp_mathlib_Nonneg_natCast___redArg(v_inst_161_);
v___x_165_ = lp_mathlib_Nonneg_addMonoid___redArg(v_toAddMonoid_162_);
v___x_166_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_166_, 0, v___x_164_);
lean_ctor_set(v___x_166_, 1, v___x_165_);
lean_ctor_set(v___x_166_, 2, v_toOne_163_);
return v___x_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addMonoidWithOne(lean_object* v_00_u03b1_167_, lean_object* v_inst_168_, lean_object* v_inst_169_, lean_object* v_inst_170_, lean_object* v_inst_171_){
_start:
{
lean_object* v___x_172_; 
v___x_172_ = lp_mathlib_Nonneg_addMonoidWithOne___redArg(v_inst_168_);
return v___x_172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_addMonoidWithOne___boxed(lean_object* v_00_u03b1_173_, lean_object* v_inst_174_, lean_object* v_inst_175_, lean_object* v_inst_176_, lean_object* v_inst_177_){
_start:
{
lean_object* v_res_178_; 
v_res_178_ = lp_mathlib_Nonneg_addMonoidWithOne(v_00_u03b1_173_, v_inst_174_, v_inst_175_, v_inst_176_, v_inst_177_);
lean_dec_ref(v_inst_175_);
return v_res_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_pow___redArg___lam__0(lean_object* v_toNPow_179_, lean_object* v_x_180_, lean_object* v_n_181_){
_start:
{
lean_object* v___x_182_; 
v___x_182_ = lean_apply_2(v_toNPow_179_, v_n_181_, v_x_180_);
return v___x_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_pow___redArg(lean_object* v_inst_183_){
_start:
{
lean_object* v_toMonoid_184_; lean_object* v_toNPow_185_; lean_object* v___f_186_; 
v_toMonoid_184_ = lean_ctor_get(v_inst_183_, 0);
lean_inc_ref(v_toMonoid_184_);
lean_dec_ref(v_inst_183_);
v_toNPow_185_ = lean_ctor_get(v_toMonoid_184_, 2);
lean_inc(v_toNPow_185_);
lean_dec_ref(v_toMonoid_184_);
v___f_186_ = lean_alloc_closure((void*)(lp_mathlib_Nonneg_pow___redArg___lam__0), 3, 1);
lean_closure_set(v___f_186_, 0, v_toNPow_185_);
return v___f_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_pow(lean_object* v_00_u03b1_187_, lean_object* v_inst_188_, lean_object* v_inst_189_, lean_object* v_inst_190_, lean_object* v_inst_191_){
_start:
{
lean_object* v___x_192_; 
v___x_192_ = lp_mathlib_Nonneg_pow___redArg(v_inst_188_);
return v___x_192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_pow___boxed(lean_object* v_00_u03b1_193_, lean_object* v_inst_194_, lean_object* v_inst_195_, lean_object* v_inst_196_, lean_object* v_inst_197_){
_start:
{
lean_object* v_res_198_; 
v_res_198_ = lp_mathlib_Nonneg_pow(v_00_u03b1_193_, v_inst_194_, v_inst_195_, v_inst_196_, v_inst_197_);
lean_dec_ref(v_inst_195_);
return v_res_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_semiring___redArg(lean_object* v_inst_199_){
_start:
{
lean_object* v_toAddCommMonoid_200_; lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v_toOne_205_; lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_209_; uint8_t v_isShared_210_; uint8_t v_isSharedCheck_218_; 
v_toAddCommMonoid_200_ = lean_ctor_get(v_inst_199_, 0);
lean_inc_ref(v_toAddCommMonoid_200_);
v___x_201_ = lp_mathlib_Nonneg_addMonoid___redArg(v_toAddCommMonoid_200_);
lean_inc_ref_n(v_inst_199_, 2);
v___x_202_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_inst_199_);
v___x_203_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_199_);
v___x_204_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v___x_203_);
v_toOne_205_ = lean_ctor_get(v___x_204_, 2);
lean_inc(v_toOne_205_);
v___x_206_ = lp_mathlib_Nonneg_mul___redArg(v___x_202_);
v___x_207_ = lp_mathlib_Semiring_toMonoidWithZero___redArg(v_inst_199_);
v_isSharedCheck_218_ = !lean_is_exclusive(v_inst_199_);
if (v_isSharedCheck_218_ == 0)
{
lean_object* v_unused_219_; lean_object* v_unused_220_; lean_object* v_unused_221_; 
v_unused_219_ = lean_ctor_get(v_inst_199_, 2);
lean_dec(v_unused_219_);
v_unused_220_ = lean_ctor_get(v_inst_199_, 1);
lean_dec(v_unused_220_);
v_unused_221_ = lean_ctor_get(v_inst_199_, 0);
lean_dec(v_unused_221_);
v___x_209_ = v_inst_199_;
v_isShared_210_ = v_isSharedCheck_218_;
goto v_resetjp_208_;
}
else
{
lean_dec(v_inst_199_);
v___x_209_ = lean_box(0);
v_isShared_210_ = v_isSharedCheck_218_;
goto v_resetjp_208_;
}
v_resetjp_208_:
{
lean_object* v___x_211_; lean_object* v___f_212_; lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_216_; 
v___x_211_ = lp_mathlib_Nonneg_pow___redArg(v___x_207_);
v___f_212_ = lean_alloc_closure((void*)(lp_mathlib_NPow_ofPow___redArg___lam__0), 3, 1);
lean_closure_set(v___f_212_, 0, v___x_211_);
v___x_213_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_213_, 0, v_toOne_205_);
lean_ctor_set(v___x_213_, 1, v___x_206_);
lean_ctor_set(v___x_213_, 2, v___f_212_);
v___x_214_ = lp_mathlib_Nonneg_natCast___redArg(v___x_204_);
if (v_isShared_210_ == 0)
{
lean_ctor_set(v___x_209_, 2, v___x_214_);
lean_ctor_set(v___x_209_, 1, v___x_213_);
lean_ctor_set(v___x_209_, 0, v___x_201_);
v___x_216_ = v___x_209_;
goto v_reusejp_215_;
}
else
{
lean_object* v_reuseFailAlloc_217_; 
v_reuseFailAlloc_217_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_217_, 0, v___x_201_);
lean_ctor_set(v_reuseFailAlloc_217_, 1, v___x_213_);
lean_ctor_set(v_reuseFailAlloc_217_, 2, v___x_214_);
v___x_216_ = v_reuseFailAlloc_217_;
goto v_reusejp_215_;
}
v_reusejp_215_:
{
return v___x_216_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_semiring(lean_object* v_00_u03b1_222_, lean_object* v_inst_223_, lean_object* v_inst_224_, lean_object* v_inst_225_, lean_object* v_inst_226_, lean_object* v_inst_227_){
_start:
{
lean_object* v___x_228_; 
v___x_228_ = lp_mathlib_Nonneg_semiring___redArg(v_inst_223_);
return v___x_228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_semiring___boxed(lean_object* v_00_u03b1_229_, lean_object* v_inst_230_, lean_object* v_inst_231_, lean_object* v_inst_232_, lean_object* v_inst_233_, lean_object* v_inst_234_){
_start:
{
lean_object* v_res_235_; 
v_res_235_ = lp_mathlib_Nonneg_semiring(v_00_u03b1_229_, v_inst_230_, v_inst_231_, v_inst_232_, v_inst_233_, v_inst_234_);
lean_dec_ref(v_inst_231_);
return v_res_235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_monoidWithZero___redArg(lean_object* v_inst_236_){
_start:
{
lean_object* v___x_237_; lean_object* v___x_238_; 
v___x_237_ = lp_mathlib_Nonneg_semiring___redArg(v_inst_236_);
v___x_238_ = lp_mathlib_Semiring_toMonoidWithZero___redArg(v___x_237_);
lean_dec_ref(v___x_237_);
return v___x_238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_monoidWithZero(lean_object* v_00_u03b1_239_, lean_object* v_inst_240_, lean_object* v_inst_241_, lean_object* v_inst_242_, lean_object* v_inst_243_, lean_object* v_inst_244_){
_start:
{
lean_object* v___x_245_; 
v___x_245_ = lp_mathlib_Nonneg_monoidWithZero___redArg(v_inst_240_);
return v___x_245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_monoidWithZero___boxed(lean_object* v_00_u03b1_246_, lean_object* v_inst_247_, lean_object* v_inst_248_, lean_object* v_inst_249_, lean_object* v_inst_250_, lean_object* v_inst_251_){
_start:
{
lean_object* v_res_252_; 
v_res_252_ = lp_mathlib_Nonneg_monoidWithZero(v_00_u03b1_246_, v_inst_247_, v_inst_248_, v_inst_249_, v_inst_250_, v_inst_251_);
lean_dec_ref(v_inst_248_);
return v_res_252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_coeRingHom(lean_object* v_00_u03b1_253_, lean_object* v_inst_254_, lean_object* v_inst_255_, lean_object* v_inst_256_, lean_object* v_inst_257_, lean_object* v_inst_258_){
_start:
{
lean_object* v___f_259_; 
v___f_259_ = ((lean_object*)(lp_mathlib_Nonneg_coeAddMonoidHom___closed__0));
return v___f_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_coeRingHom___boxed(lean_object* v_00_u03b1_260_, lean_object* v_inst_261_, lean_object* v_inst_262_, lean_object* v_inst_263_, lean_object* v_inst_264_, lean_object* v_inst_265_){
_start:
{
lean_object* v_res_266_; 
v_res_266_ = lp_mathlib_Nonneg_coeRingHom(v_00_u03b1_260_, v_inst_261_, v_inst_262_, v_inst_263_, v_inst_264_, v_inst_265_);
lean_dec_ref(v_inst_262_);
lean_dec_ref(v_inst_261_);
return v_res_266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_commSemiring___redArg(lean_object* v_inst_267_){
_start:
{
lean_object* v___x_268_; 
v___x_268_ = lp_mathlib_Nonneg_semiring___redArg(v_inst_267_);
return v___x_268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_commSemiring(lean_object* v_00_u03b1_269_, lean_object* v_inst_270_, lean_object* v_inst_271_, lean_object* v_inst_272_, lean_object* v_inst_273_, lean_object* v_inst_274_){
_start:
{
lean_object* v___x_275_; 
v___x_275_ = lp_mathlib_Nonneg_semiring___redArg(v_inst_270_);
return v___x_275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_commSemiring___boxed(lean_object* v_00_u03b1_276_, lean_object* v_inst_277_, lean_object* v_inst_278_, lean_object* v_inst_279_, lean_object* v_inst_280_, lean_object* v_inst_281_){
_start:
{
lean_object* v_res_282_; 
v_res_282_ = lp_mathlib_Nonneg_commSemiring(v_00_u03b1_276_, v_inst_277_, v_inst_278_, v_inst_279_, v_inst_280_, v_inst_281_);
lean_dec_ref(v_inst_278_);
return v_res_282_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_commMonoidWithZero___redArg(lean_object* v_inst_283_){
_start:
{
lean_object* v___x_284_; lean_object* v___x_285_; 
v___x_284_ = lp_mathlib_Nonneg_semiring___redArg(v_inst_283_);
v___x_285_ = lp_mathlib_CommSemiring_toCommMonoidWithZero___redArg(v___x_284_);
lean_dec_ref(v___x_284_);
return v___x_285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_commMonoidWithZero(lean_object* v_00_u03b1_286_, lean_object* v_inst_287_, lean_object* v_inst_288_, lean_object* v_inst_289_, lean_object* v_inst_290_, lean_object* v_inst_291_){
_start:
{
lean_object* v___x_292_; 
v___x_292_ = lp_mathlib_Nonneg_commMonoidWithZero___redArg(v_inst_287_);
return v___x_292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_commMonoidWithZero___boxed(lean_object* v_00_u03b1_293_, lean_object* v_inst_294_, lean_object* v_inst_295_, lean_object* v_inst_296_, lean_object* v_inst_297_, lean_object* v_inst_298_){
_start:
{
lean_object* v_res_299_; 
v_res_299_ = lp_mathlib_Nonneg_commMonoidWithZero(v_00_u03b1_293_, v_inst_294_, v_inst_295_, v_inst_296_, v_inst_297_, v_inst_298_);
lean_dec_ref(v_inst_295_);
return v_res_299_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_toNonneg___redArg(lean_object* v_inst_300_, lean_object* v_inst_301_, lean_object* v_a_302_){
_start:
{
lean_object* v_sup_303_; lean_object* v___x_304_; 
v_sup_303_ = lean_ctor_get(v_inst_301_, 1);
lean_inc(v_sup_303_);
lean_dec_ref(v_inst_301_);
v___x_304_ = lean_apply_2(v_sup_303_, v_a_302_, v_inst_300_);
return v___x_304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_toNonneg(lean_object* v_00_u03b1_305_, lean_object* v_inst_306_, lean_object* v_inst_307_, lean_object* v_a_308_){
_start:
{
lean_object* v___x_309_; 
v___x_309_ = lp_mathlib_Nonneg_toNonneg___redArg(v_inst_306_, v_inst_307_, v_a_308_);
return v___x_309_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_sub___redArg___lam__0(lean_object* v_inst_310_, lean_object* v_inst_311_, lean_object* v_inst_312_, lean_object* v_x_313_, lean_object* v_y_314_){
_start:
{
lean_object* v___x_315_; lean_object* v___x_316_; 
v___x_315_ = lean_apply_2(v_inst_310_, v_x_313_, v_y_314_);
v___x_316_ = lp_mathlib_Nonneg_toNonneg___redArg(v_inst_311_, v_inst_312_, v___x_315_);
return v___x_316_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_sub___redArg(lean_object* v_inst_317_, lean_object* v_inst_318_, lean_object* v_inst_319_){
_start:
{
lean_object* v___f_320_; 
v___f_320_ = lean_alloc_closure((void*)(lp_mathlib_Nonneg_sub___redArg___lam__0), 5, 3);
lean_closure_set(v___f_320_, 0, v_inst_319_);
lean_closure_set(v___f_320_, 1, v_inst_317_);
lean_closure_set(v___f_320_, 2, v_inst_318_);
return v___f_320_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_sub(lean_object* v_00_u03b1_321_, lean_object* v_inst_322_, lean_object* v_inst_323_, lean_object* v_inst_324_){
_start:
{
lean_object* v___f_325_; 
v___f_325_ = lean_alloc_closure((void*)(lp_mathlib_Nonneg_sub___redArg___lam__0), 5, 3);
lean_closure_set(v___f_325_, 0, v_inst_324_);
lean_closure_set(v___f_325_, 1, v_inst_322_);
lean_closure_set(v___f_325_, 2, v_inst_323_);
return v___f_325_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_Pow(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_ZeroLEOne(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_InjSurj(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_Pow(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_ZeroLEOne(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_Pow(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_ZeroLEOne(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_InjSurj(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_Pow(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_ZeroLEOne(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
