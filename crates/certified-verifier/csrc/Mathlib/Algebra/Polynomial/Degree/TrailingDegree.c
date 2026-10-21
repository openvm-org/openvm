// Lean compiler output
// Module: Mathlib.Algebra.Polynomial.Degree.TrailingDegree
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Polynomial.Degree.Support public import Mathlib.Data.ENat.Monoid
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
extern lean_object* lp_mathlib_Nat_instLinearOrder;
lean_object* lp_mathlib_Finset_min___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_ENat_toNat(lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lp_mathlib_instMulZeroClassOfSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_trailingDegree___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_trailingDegree(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_trailingDegree___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_natTrailingDegree___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_natTrailingDegree(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_natTrailingDegree___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_trailingCoeff___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_trailingCoeff(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_trailingCoeff___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Polynomial_TrailingMonic_decidable___aux__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_TrailingMonic_decidable___aux__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Polynomial_TrailingMonic_decidable___aux__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_TrailingMonic_decidable___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Polynomial_TrailingMonic_decidable___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_TrailingMonic_decidable___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Polynomial_TrailingMonic_decidable(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_TrailingMonic_decidable___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_nextCoeffUp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_nextCoeffUp(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_trailingDegree___redArg(lean_object* v_p_1_){
_start:
{
lean_object* v___x_2_; lean_object* v_support_3_; lean_object* v___x_4_; 
v___x_2_ = lp_mathlib_Nat_instLinearOrder;
v_support_3_ = lean_ctor_get(v_p_1_, 0);
lean_inc(v_support_3_);
lean_dec_ref(v_p_1_);
v___x_4_ = lp_mathlib_Finset_min___redArg(v___x_2_, v_support_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_trailingDegree(lean_object* v_R_5_, lean_object* v_inst_6_, lean_object* v_p_7_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lp_mathlib_Polynomial_trailingDegree___redArg(v_p_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_trailingDegree___boxed(lean_object* v_R_9_, lean_object* v_inst_10_, lean_object* v_p_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_mathlib_Polynomial_trailingDegree(v_R_9_, v_inst_10_, v_p_11_);
lean_dec_ref(v_inst_10_);
return v_res_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_natTrailingDegree___redArg(lean_object* v_p_13_){
_start:
{
lean_object* v___x_14_; lean_object* v___x_15_; 
v___x_14_ = lp_mathlib_Polynomial_trailingDegree___redArg(v_p_13_);
v___x_15_ = lp_mathlib_ENat_toNat(v___x_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_natTrailingDegree(lean_object* v_R_16_, lean_object* v_inst_17_, lean_object* v_p_18_){
_start:
{
lean_object* v___x_19_; 
v___x_19_ = lp_mathlib_Polynomial_natTrailingDegree___redArg(v_p_18_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_natTrailingDegree___boxed(lean_object* v_R_20_, lean_object* v_inst_21_, lean_object* v_p_22_){
_start:
{
lean_object* v_res_23_; 
v_res_23_ = lp_mathlib_Polynomial_natTrailingDegree(v_R_20_, v_inst_21_, v_p_22_);
lean_dec_ref(v_inst_21_);
return v_res_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_trailingCoeff___redArg(lean_object* v_p_24_){
_start:
{
lean_object* v_toFun_25_; lean_object* v___x_26_; lean_object* v___x_27_; 
v_toFun_25_ = lean_ctor_get(v_p_24_, 1);
lean_inc(v_toFun_25_);
v___x_26_ = lp_mathlib_Polynomial_natTrailingDegree___redArg(v_p_24_);
v___x_27_ = lean_apply_1(v_toFun_25_, v___x_26_);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_trailingCoeff(lean_object* v_R_28_, lean_object* v_inst_29_, lean_object* v_p_30_){
_start:
{
lean_object* v___x_31_; 
v___x_31_ = lp_mathlib_Polynomial_trailingCoeff___redArg(v_p_30_);
return v___x_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_trailingCoeff___boxed(lean_object* v_R_32_, lean_object* v_inst_33_, lean_object* v_p_34_){
_start:
{
lean_object* v_res_35_; 
v_res_35_ = lp_mathlib_Polynomial_trailingCoeff(v_R_32_, v_inst_33_, v_p_34_);
lean_dec_ref(v_inst_33_);
return v_res_35_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Polynomial_TrailingMonic_decidable___aux__1___redArg(lean_object* v_inst_36_, lean_object* v_p_37_, lean_object* v_inst_38_){
_start:
{
lean_object* v___x_39_; lean_object* v___x_40_; lean_object* v_toOne_41_; lean_object* v___x_42_; lean_object* v___x_43_; uint8_t v___x_44_; 
v___x_39_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_36_);
v___x_40_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v___x_39_);
v_toOne_41_ = lean_ctor_get(v___x_40_, 2);
lean_inc(v_toOne_41_);
lean_dec_ref(v___x_40_);
v___x_42_ = lp_mathlib_Polynomial_trailingCoeff___redArg(v_p_37_);
v___x_43_ = lean_apply_2(v_inst_38_, v___x_42_, v_toOne_41_);
v___x_44_ = lean_unbox(v___x_43_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_TrailingMonic_decidable___aux__1___redArg___boxed(lean_object* v_inst_45_, lean_object* v_p_46_, lean_object* v_inst_47_){
_start:
{
uint8_t v_res_48_; lean_object* v_r_49_; 
v_res_48_ = lp_mathlib_Polynomial_TrailingMonic_decidable___aux__1___redArg(v_inst_45_, v_p_46_, v_inst_47_);
v_r_49_ = lean_box(v_res_48_);
return v_r_49_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Polynomial_TrailingMonic_decidable___aux__1(lean_object* v_R_50_, lean_object* v_inst_51_, lean_object* v_p_52_, lean_object* v_inst_53_){
_start:
{
uint8_t v___x_54_; 
v___x_54_ = lp_mathlib_Polynomial_TrailingMonic_decidable___aux__1___redArg(v_inst_51_, v_p_52_, v_inst_53_);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_TrailingMonic_decidable___aux__1___boxed(lean_object* v_R_55_, lean_object* v_inst_56_, lean_object* v_p_57_, lean_object* v_inst_58_){
_start:
{
uint8_t v_res_59_; lean_object* v_r_60_; 
v_res_59_ = lp_mathlib_Polynomial_TrailingMonic_decidable___aux__1(v_R_55_, v_inst_56_, v_p_57_, v_inst_58_);
v_r_60_ = lean_box(v_res_59_);
return v_r_60_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Polynomial_TrailingMonic_decidable___redArg(lean_object* v_inst_61_, lean_object* v_p_62_, lean_object* v_inst_63_){
_start:
{
uint8_t v___x_64_; 
v___x_64_ = lp_mathlib_Polynomial_TrailingMonic_decidable___aux__1___redArg(v_inst_61_, v_p_62_, v_inst_63_);
return v___x_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_TrailingMonic_decidable___redArg___boxed(lean_object* v_inst_65_, lean_object* v_p_66_, lean_object* v_inst_67_){
_start:
{
uint8_t v_res_68_; lean_object* v_r_69_; 
v_res_68_ = lp_mathlib_Polynomial_TrailingMonic_decidable___redArg(v_inst_65_, v_p_66_, v_inst_67_);
v_r_69_ = lean_box(v_res_68_);
return v_r_69_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Polynomial_TrailingMonic_decidable(lean_object* v_R_70_, lean_object* v_inst_71_, lean_object* v_p_72_, lean_object* v_inst_73_){
_start:
{
uint8_t v___x_74_; 
v___x_74_ = lp_mathlib_Polynomial_TrailingMonic_decidable___aux__1___redArg(v_inst_71_, v_p_72_, v_inst_73_);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_TrailingMonic_decidable___boxed(lean_object* v_R_75_, lean_object* v_inst_76_, lean_object* v_p_77_, lean_object* v_inst_78_){
_start:
{
uint8_t v_res_79_; lean_object* v_r_80_; 
v_res_79_ = lp_mathlib_Polynomial_TrailingMonic_decidable(v_R_75_, v_inst_76_, v_p_77_, v_inst_78_);
v_r_80_ = lean_box(v_res_79_);
return v_r_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_nextCoeffUp___redArg(lean_object* v_inst_81_, lean_object* v_p_82_){
_start:
{
lean_object* v___x_83_; lean_object* v___x_84_; uint8_t v___x_85_; 
lean_inc_ref(v_p_82_);
v___x_83_ = lp_mathlib_Polynomial_natTrailingDegree___redArg(v_p_82_);
v___x_84_ = lean_unsigned_to_nat(0u);
v___x_85_ = lean_nat_dec_eq(v___x_83_, v___x_84_);
if (v___x_85_ == 0)
{
lean_object* v_toFun_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; 
lean_dec_ref(v_inst_81_);
v_toFun_86_ = lean_ctor_get(v_p_82_, 1);
lean_inc(v_toFun_86_);
lean_dec_ref(v_p_82_);
v___x_87_ = lean_unsigned_to_nat(1u);
v___x_88_ = lean_nat_add(v___x_83_, v___x_87_);
lean_dec(v___x_83_);
v___x_89_ = lean_apply_1(v_toFun_86_, v___x_88_);
return v___x_89_;
}
else
{
lean_object* v___x_90_; lean_object* v_toZero_91_; 
lean_dec(v___x_83_);
lean_dec_ref(v_p_82_);
v___x_90_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_inst_81_);
v_toZero_91_ = lean_ctor_get(v___x_90_, 1);
lean_inc(v_toZero_91_);
lean_dec_ref(v___x_90_);
return v_toZero_91_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_nextCoeffUp(lean_object* v_R_92_, lean_object* v_inst_93_, lean_object* v_p_94_){
_start:
{
lean_object* v___x_95_; 
v___x_95_ = lp_mathlib_Polynomial_nextCoeffUp___redArg(v_inst_93_, v_p_94_);
return v___x_95_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_Degree_Support(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_ENat_Monoid(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_Degree_TrailingDegree(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_Degree_Support(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_ENat_Monoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Polynomial_Degree_TrailingDegree(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_Degree_Support(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_ENat_Monoid(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_Degree_TrailingDegree(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_Degree_Support(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_ENat_Monoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_Degree_TrailingDegree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Polynomial_Degree_TrailingDegree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Polynomial_Degree_TrailingDegree(builtin);
}
#ifdef __cplusplus
}
#endif
