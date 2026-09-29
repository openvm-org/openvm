// Lean compiler output
// Module: Mathlib.Algebra.Polynomial.Degree.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.MonoidAlgebra.Degree public import Mathlib.Algebra.Order.Ring.WithTop public import Mathlib.Algebra.Polynomial.Basic public import Mathlib.Data.Nat.Cast.WithTop public import Mathlib.Data.Nat.SuccPred public import Mathlib.Order.SuccPred.WithBot
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
lean_object* lp_mathlib_Finset_max___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_WithBot_unbotD___redArg(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lp_mathlib_instMulZeroClassOfSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_degree___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_degree(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_degree___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_natDegree___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_natDegree(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_natDegree___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_leadingCoeff___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_leadingCoeff(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_leadingCoeff___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Polynomial_Monic_decidable___aux__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_Monic_decidable___aux__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Polynomial_Monic_decidable___aux__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_Monic_decidable___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Polynomial_Monic_decidable___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_Monic_decidable___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Polynomial_Monic_decidable(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_Monic_decidable___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_nextCoeff___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_nextCoeff(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_degree___redArg(lean_object* v_p_1_){
_start:
{
lean_object* v___x_2_; lean_object* v_support_3_; lean_object* v___x_4_; 
v___x_2_ = lp_mathlib_Nat_instLinearOrder;
v_support_3_ = lean_ctor_get(v_p_1_, 0);
lean_inc(v_support_3_);
lean_dec_ref(v_p_1_);
v___x_4_ = lp_mathlib_Finset_max___redArg(v___x_2_, v_support_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_degree(lean_object* v_R_5_, lean_object* v_inst_6_, lean_object* v_p_7_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lp_mathlib_Polynomial_degree___redArg(v_p_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_degree___boxed(lean_object* v_R_9_, lean_object* v_inst_10_, lean_object* v_p_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_mathlib_Polynomial_degree(v_R_9_, v_inst_10_, v_p_11_);
lean_dec_ref(v_inst_10_);
return v_res_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_natDegree___redArg(lean_object* v_p_13_){
_start:
{
lean_object* v___x_14_; lean_object* v___x_15_; lean_object* v___x_16_; 
v___x_14_ = lean_unsigned_to_nat(0u);
v___x_15_ = lp_mathlib_Polynomial_degree___redArg(v_p_13_);
v___x_16_ = lp_mathlib_WithBot_unbotD___redArg(v___x_14_, v___x_15_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_natDegree(lean_object* v_R_17_, lean_object* v_inst_18_, lean_object* v_p_19_){
_start:
{
lean_object* v___x_20_; 
v___x_20_ = lp_mathlib_Polynomial_natDegree___redArg(v_p_19_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_natDegree___boxed(lean_object* v_R_21_, lean_object* v_inst_22_, lean_object* v_p_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_mathlib_Polynomial_natDegree(v_R_21_, v_inst_22_, v_p_23_);
lean_dec_ref(v_inst_22_);
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_leadingCoeff___redArg(lean_object* v_p_25_){
_start:
{
lean_object* v_toFun_26_; lean_object* v___x_27_; lean_object* v___x_28_; 
v_toFun_26_ = lean_ctor_get(v_p_25_, 1);
lean_inc(v_toFun_26_);
v___x_27_ = lp_mathlib_Polynomial_natDegree___redArg(v_p_25_);
v___x_28_ = lean_apply_1(v_toFun_26_, v___x_27_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_leadingCoeff(lean_object* v_R_29_, lean_object* v_inst_30_, lean_object* v_p_31_){
_start:
{
lean_object* v___x_32_; 
v___x_32_ = lp_mathlib_Polynomial_leadingCoeff___redArg(v_p_31_);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_leadingCoeff___boxed(lean_object* v_R_33_, lean_object* v_inst_34_, lean_object* v_p_35_){
_start:
{
lean_object* v_res_36_; 
v_res_36_ = lp_mathlib_Polynomial_leadingCoeff(v_R_33_, v_inst_34_, v_p_35_);
lean_dec_ref(v_inst_34_);
return v_res_36_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Polynomial_Monic_decidable___aux__1___redArg(lean_object* v_inst_37_, lean_object* v_p_38_, lean_object* v_inst_39_){
_start:
{
lean_object* v___x_40_; lean_object* v___x_41_; lean_object* v_toOne_42_; lean_object* v___x_43_; lean_object* v___x_44_; uint8_t v___x_45_; 
v___x_40_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_37_);
v___x_41_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v___x_40_);
v_toOne_42_ = lean_ctor_get(v___x_41_, 2);
lean_inc(v_toOne_42_);
lean_dec_ref(v___x_41_);
v___x_43_ = lp_mathlib_Polynomial_leadingCoeff___redArg(v_p_38_);
v___x_44_ = lean_apply_2(v_inst_39_, v___x_43_, v_toOne_42_);
v___x_45_ = lean_unbox(v___x_44_);
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_Monic_decidable___aux__1___redArg___boxed(lean_object* v_inst_46_, lean_object* v_p_47_, lean_object* v_inst_48_){
_start:
{
uint8_t v_res_49_; lean_object* v_r_50_; 
v_res_49_ = lp_mathlib_Polynomial_Monic_decidable___aux__1___redArg(v_inst_46_, v_p_47_, v_inst_48_);
v_r_50_ = lean_box(v_res_49_);
return v_r_50_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Polynomial_Monic_decidable___aux__1(lean_object* v_R_51_, lean_object* v_inst_52_, lean_object* v_p_53_, lean_object* v_inst_54_){
_start:
{
uint8_t v___x_55_; 
v___x_55_ = lp_mathlib_Polynomial_Monic_decidable___aux__1___redArg(v_inst_52_, v_p_53_, v_inst_54_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_Monic_decidable___aux__1___boxed(lean_object* v_R_56_, lean_object* v_inst_57_, lean_object* v_p_58_, lean_object* v_inst_59_){
_start:
{
uint8_t v_res_60_; lean_object* v_r_61_; 
v_res_60_ = lp_mathlib_Polynomial_Monic_decidable___aux__1(v_R_56_, v_inst_57_, v_p_58_, v_inst_59_);
v_r_61_ = lean_box(v_res_60_);
return v_r_61_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Polynomial_Monic_decidable___redArg(lean_object* v_inst_62_, lean_object* v_p_63_, lean_object* v_inst_64_){
_start:
{
uint8_t v___x_65_; 
v___x_65_ = lp_mathlib_Polynomial_Monic_decidable___aux__1___redArg(v_inst_62_, v_p_63_, v_inst_64_);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_Monic_decidable___redArg___boxed(lean_object* v_inst_66_, lean_object* v_p_67_, lean_object* v_inst_68_){
_start:
{
uint8_t v_res_69_; lean_object* v_r_70_; 
v_res_69_ = lp_mathlib_Polynomial_Monic_decidable___redArg(v_inst_66_, v_p_67_, v_inst_68_);
v_r_70_ = lean_box(v_res_69_);
return v_r_70_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Polynomial_Monic_decidable(lean_object* v_R_71_, lean_object* v_inst_72_, lean_object* v_p_73_, lean_object* v_inst_74_){
_start:
{
uint8_t v___x_75_; 
v___x_75_ = lp_mathlib_Polynomial_Monic_decidable___aux__1___redArg(v_inst_72_, v_p_73_, v_inst_74_);
return v___x_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_Monic_decidable___boxed(lean_object* v_R_76_, lean_object* v_inst_77_, lean_object* v_p_78_, lean_object* v_inst_79_){
_start:
{
uint8_t v_res_80_; lean_object* v_r_81_; 
v_res_80_ = lp_mathlib_Polynomial_Monic_decidable(v_R_76_, v_inst_77_, v_p_78_, v_inst_79_);
v_r_81_ = lean_box(v_res_80_);
return v_r_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_nextCoeff___redArg(lean_object* v_inst_82_, lean_object* v_p_83_){
_start:
{
lean_object* v___x_84_; lean_object* v___x_85_; uint8_t v___x_86_; 
lean_inc_ref(v_p_83_);
v___x_84_ = lp_mathlib_Polynomial_natDegree___redArg(v_p_83_);
v___x_85_ = lean_unsigned_to_nat(0u);
v___x_86_ = lean_nat_dec_eq(v___x_84_, v___x_85_);
if (v___x_86_ == 0)
{
lean_object* v_toFun_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; 
lean_dec_ref(v_inst_82_);
v_toFun_87_ = lean_ctor_get(v_p_83_, 1);
lean_inc(v_toFun_87_);
lean_dec_ref(v_p_83_);
v___x_88_ = lean_unsigned_to_nat(1u);
v___x_89_ = lean_nat_sub(v___x_84_, v___x_88_);
lean_dec(v___x_84_);
v___x_90_ = lean_apply_1(v_toFun_87_, v___x_89_);
return v___x_90_;
}
else
{
lean_object* v___x_91_; lean_object* v_toZero_92_; 
lean_dec(v___x_84_);
lean_dec_ref(v_p_83_);
v___x_91_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_inst_82_);
v_toZero_92_ = lean_ctor_get(v___x_91_, 1);
lean_inc(v_toZero_92_);
lean_dec_ref(v___x_91_);
return v_toZero_92_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_nextCoeff(lean_object* v_R_93_, lean_object* v_inst_94_, lean_object* v_p_95_){
_start:
{
lean_object* v___x_96_; 
v___x_96_ = lp_mathlib_Polynomial_nextCoeff___redArg(v_inst_94_, v_p_95_);
return v___x_96_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Degree(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_WithTop(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_WithTop(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_SuccPred(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_SuccPred_WithBot(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_Degree_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Degree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_WithTop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_WithTop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_SuccPred(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_SuccPred_WithBot(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Polynomial_Degree_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Degree(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_WithTop(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Cast_WithTop(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_SuccPred(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_SuccPred_WithBot(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_Degree_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Degree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Ring_WithTop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Cast_WithTop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_SuccPred(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_SuccPred_WithBot(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_Degree_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Polynomial_Degree_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Polynomial_Degree_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
