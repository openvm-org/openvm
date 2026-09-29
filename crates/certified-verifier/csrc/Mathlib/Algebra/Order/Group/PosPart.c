// Lean compiler output
// Module: Mathlib.Algebra.Order.Group.PosPart
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Order.Group.Unbundled.Abs public import Mathlib.Algebra.Notation
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
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOneLePart___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOneLePart___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOneLePart___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOneLePart(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOneLePart___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instPosPart___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instPosPart___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instPosPart___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instPosPart(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instPosPart___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLeOnePart___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLeOnePart___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLeOnePart(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instNegPart___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instNegPart___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instNegPart(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOneLePart___redArg___lam__0(lean_object* v_toSemilatticeSup_1_, lean_object* v_toOne_2_, lean_object* v_a_3_){
_start:
{
lean_object* v_sup_4_; lean_object* v___x_5_; 
v_sup_4_ = lean_ctor_get(v_toSemilatticeSup_1_, 1);
lean_inc(v_sup_4_);
lean_dec_ref(v_toSemilatticeSup_1_);
v___x_5_ = lean_apply_2(v_sup_4_, v_a_3_, v_toOne_2_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOneLePart___redArg(lean_object* v_inst_6_, lean_object* v_inst_7_){
_start:
{
lean_object* v_toSemilatticeSup_8_; lean_object* v_toMonoid_9_; lean_object* v___x_10_; lean_object* v___x_11_; lean_object* v_toOne_12_; lean_object* v___f_13_; 
v_toSemilatticeSup_8_ = lean_ctor_get(v_inst_6_, 0);
lean_inc_ref(v_toSemilatticeSup_8_);
lean_dec_ref(v_inst_6_);
v_toMonoid_9_ = lean_ctor_get(v_inst_7_, 0);
v___x_10_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_9_);
v___x_11_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_10_);
v_toOne_12_ = lean_ctor_get(v___x_11_, 0);
lean_inc(v_toOne_12_);
lean_dec_ref(v___x_11_);
v___f_13_ = lean_alloc_closure((void*)(lp_mathlib_instOneLePart___redArg___lam__0), 3, 2);
lean_closure_set(v___f_13_, 0, v_toSemilatticeSup_8_);
lean_closure_set(v___f_13_, 1, v_toOne_12_);
return v___f_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOneLePart___redArg___boxed(lean_object* v_inst_14_, lean_object* v_inst_15_){
_start:
{
lean_object* v_res_16_; 
v_res_16_ = lp_mathlib_instOneLePart___redArg(v_inst_14_, v_inst_15_);
lean_dec_ref(v_inst_15_);
return v_res_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOneLePart(lean_object* v_00_u03b1_17_, lean_object* v_inst_18_, lean_object* v_inst_19_){
_start:
{
lean_object* v___x_20_; 
v___x_20_ = lp_mathlib_instOneLePart___redArg(v_inst_18_, v_inst_19_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOneLePart___boxed(lean_object* v_00_u03b1_21_, lean_object* v_inst_22_, lean_object* v_inst_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_mathlib_instOneLePart(v_00_u03b1_21_, v_inst_22_, v_inst_23_);
lean_dec_ref(v_inst_23_);
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instPosPart___redArg___lam__0(lean_object* v_toSemilatticeSup_25_, lean_object* v_toZero_26_, lean_object* v_a_27_){
_start:
{
lean_object* v_sup_28_; lean_object* v___x_29_; 
v_sup_28_ = lean_ctor_get(v_toSemilatticeSup_25_, 1);
lean_inc(v_sup_28_);
lean_dec_ref(v_toSemilatticeSup_25_);
v___x_29_ = lean_apply_2(v_sup_28_, v_a_27_, v_toZero_26_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instPosPart___redArg(lean_object* v_inst_30_, lean_object* v_inst_31_){
_start:
{
lean_object* v_toSemilatticeSup_32_; lean_object* v_toAddMonoid_33_; lean_object* v___x_34_; lean_object* v___x_35_; lean_object* v_toZero_36_; lean_object* v___f_37_; 
v_toSemilatticeSup_32_ = lean_ctor_get(v_inst_30_, 0);
lean_inc_ref(v_toSemilatticeSup_32_);
lean_dec_ref(v_inst_30_);
v_toAddMonoid_33_ = lean_ctor_get(v_inst_31_, 0);
v___x_34_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_33_);
v___x_35_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_34_);
v_toZero_36_ = lean_ctor_get(v___x_35_, 0);
lean_inc(v_toZero_36_);
lean_dec_ref(v___x_35_);
v___f_37_ = lean_alloc_closure((void*)(lp_mathlib_instPosPart___redArg___lam__0), 3, 2);
lean_closure_set(v___f_37_, 0, v_toSemilatticeSup_32_);
lean_closure_set(v___f_37_, 1, v_toZero_36_);
return v___f_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instPosPart___redArg___boxed(lean_object* v_inst_38_, lean_object* v_inst_39_){
_start:
{
lean_object* v_res_40_; 
v_res_40_ = lp_mathlib_instPosPart___redArg(v_inst_38_, v_inst_39_);
lean_dec_ref(v_inst_39_);
return v_res_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instPosPart(lean_object* v_00_u03b1_41_, lean_object* v_inst_42_, lean_object* v_inst_43_){
_start:
{
lean_object* v___x_44_; 
v___x_44_ = lp_mathlib_instPosPart___redArg(v_inst_42_, v_inst_43_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instPosPart___boxed(lean_object* v_00_u03b1_45_, lean_object* v_inst_46_, lean_object* v_inst_47_){
_start:
{
lean_object* v_res_48_; 
v_res_48_ = lp_mathlib_instPosPart(v_00_u03b1_45_, v_inst_46_, v_inst_47_);
lean_dec_ref(v_inst_47_);
return v_res_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLeOnePart___redArg___lam__0(lean_object* v_toSemilatticeSup_49_, lean_object* v_toInv_50_, lean_object* v_toOne_51_, lean_object* v_a_52_){
_start:
{
lean_object* v_sup_53_; lean_object* v___x_54_; lean_object* v___x_55_; 
v_sup_53_ = lean_ctor_get(v_toSemilatticeSup_49_, 1);
lean_inc(v_sup_53_);
lean_dec_ref(v_toSemilatticeSup_49_);
v___x_54_ = lean_apply_1(v_toInv_50_, v_a_52_);
v___x_55_ = lean_apply_2(v_sup_53_, v___x_54_, v_toOne_51_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLeOnePart___redArg(lean_object* v_inst_56_, lean_object* v_inst_57_){
_start:
{
lean_object* v_toSemilatticeSup_58_; lean_object* v_toMonoid_59_; lean_object* v_toInv_60_; lean_object* v___x_61_; lean_object* v___x_62_; lean_object* v_toOne_63_; lean_object* v___f_64_; 
v_toSemilatticeSup_58_ = lean_ctor_get(v_inst_56_, 0);
lean_inc_ref(v_toSemilatticeSup_58_);
lean_dec_ref(v_inst_56_);
v_toMonoid_59_ = lean_ctor_get(v_inst_57_, 0);
lean_inc_ref(v_toMonoid_59_);
v_toInv_60_ = lean_ctor_get(v_inst_57_, 1);
lean_inc(v_toInv_60_);
lean_dec_ref(v_inst_57_);
v___x_61_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_59_);
lean_dec_ref(v_toMonoid_59_);
v___x_62_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_61_);
v_toOne_63_ = lean_ctor_get(v___x_62_, 0);
lean_inc(v_toOne_63_);
lean_dec_ref(v___x_62_);
v___f_64_ = lean_alloc_closure((void*)(lp_mathlib_instLeOnePart___redArg___lam__0), 4, 3);
lean_closure_set(v___f_64_, 0, v_toSemilatticeSup_58_);
lean_closure_set(v___f_64_, 1, v_toInv_60_);
lean_closure_set(v___f_64_, 2, v_toOne_63_);
return v___f_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLeOnePart(lean_object* v_00_u03b1_65_, lean_object* v_inst_66_, lean_object* v_inst_67_){
_start:
{
lean_object* v___x_68_; 
v___x_68_ = lp_mathlib_instLeOnePart___redArg(v_inst_66_, v_inst_67_);
return v___x_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instNegPart___redArg___lam__0(lean_object* v_toSemilatticeSup_69_, lean_object* v_toNeg_70_, lean_object* v_toZero_71_, lean_object* v_a_72_){
_start:
{
lean_object* v_sup_73_; lean_object* v___x_74_; lean_object* v___x_75_; 
v_sup_73_ = lean_ctor_get(v_toSemilatticeSup_69_, 1);
lean_inc(v_sup_73_);
lean_dec_ref(v_toSemilatticeSup_69_);
v___x_74_ = lean_apply_1(v_toNeg_70_, v_a_72_);
v___x_75_ = lean_apply_2(v_sup_73_, v___x_74_, v_toZero_71_);
return v___x_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instNegPart___redArg(lean_object* v_inst_76_, lean_object* v_inst_77_){
_start:
{
lean_object* v_toSemilatticeSup_78_; lean_object* v_toAddMonoid_79_; lean_object* v_toNeg_80_; lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v_toZero_83_; lean_object* v___f_84_; 
v_toSemilatticeSup_78_ = lean_ctor_get(v_inst_76_, 0);
lean_inc_ref(v_toSemilatticeSup_78_);
lean_dec_ref(v_inst_76_);
v_toAddMonoid_79_ = lean_ctor_get(v_inst_77_, 0);
lean_inc_ref(v_toAddMonoid_79_);
v_toNeg_80_ = lean_ctor_get(v_inst_77_, 1);
lean_inc(v_toNeg_80_);
lean_dec_ref(v_inst_77_);
v___x_81_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_79_);
lean_dec_ref(v_toAddMonoid_79_);
v___x_82_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_81_);
v_toZero_83_ = lean_ctor_get(v___x_82_, 0);
lean_inc(v_toZero_83_);
lean_dec_ref(v___x_82_);
v___f_84_ = lean_alloc_closure((void*)(lp_mathlib_instNegPart___redArg___lam__0), 4, 3);
lean_closure_set(v___f_84_, 0, v_toSemilatticeSup_78_);
lean_closure_set(v___f_84_, 1, v_toNeg_80_);
lean_closure_set(v___f_84_, 2, v_toZero_83_);
return v___f_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instNegPart(lean_object* v_00_u03b1_85_, lean_object* v_inst_86_, lean_object* v_inst_87_){
_start:
{
lean_object* v___x_88_; 
v___x_88_ = lp_mathlib_instNegPart___redArg(v_inst_86_, v_inst_87_);
return v___x_88_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Abs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Notation(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_PosPart(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Abs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Order_Group_PosPart(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Abs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Notation(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_PosPart(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Abs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_PosPart(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Order_Group_PosPart(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Order_Group_PosPart(builtin);
}
#ifdef __cplusplus
}
#endif
