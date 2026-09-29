// Lean compiler output
// Module: Mathlib.Algebra.Module.NatInt
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Module.Defs public import Mathlib.Data.Int.Cast.Lemmas
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
lean_object* lp_mathlib_NSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_ZSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddCommGroup___redArg(lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_SubNegMonoid_sub_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMulActionNatOfAddMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMulActionNatOfAddMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSMulWithZeroNat___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSMulWithZeroNat(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMulActionIntOfSubtractionMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMulActionIntOfSubtractionMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSMulWithZeroInt___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSMulWithZeroInt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_toNatModule___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_toNatModule(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_toIntModule___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_toIntModule(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_addCommMonoidToAddCommGroup___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_addCommMonoidToAddCommGroup___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_addCommMonoidToAddCommGroup___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_addCommMonoidToAddCommGroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_uniqueNatModule___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_uniqueNatModule(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_uniqueIntModule___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_uniqueIntModule(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMulActionNatOfAddMonoid___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v_toNSMul_2_; lean_object* v___f_3_; 
v_toNSMul_2_ = lean_ctor_get(v_inst_1_, 2);
lean_inc(v_toNSMul_2_);
lean_dec_ref(v_inst_1_);
v___f_3_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_3_, 0, v_toNSMul_2_);
return v___f_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMulActionNatOfAddMonoid(lean_object* v_M_4_, lean_object* v_inst_5_){
_start:
{
lean_object* v___x_6_; 
v___x_6_ = lp_mathlib_instMulActionNatOfAddMonoid___redArg(v_inst_5_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSMulWithZeroNat___redArg(lean_object* v_inst_7_){
_start:
{
lean_object* v_toNSMul_8_; lean_object* v___f_9_; 
v_toNSMul_8_ = lean_ctor_get(v_inst_7_, 2);
lean_inc(v_toNSMul_8_);
lean_dec_ref(v_inst_7_);
v___f_9_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_9_, 0, v_toNSMul_8_);
return v___f_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSMulWithZeroNat(lean_object* v_M_10_, lean_object* v_inst_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lp_mathlib_instSMulWithZeroNat___redArg(v_inst_11_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMulActionIntOfSubtractionMonoid___redArg(lean_object* v_inst_13_){
_start:
{
lean_object* v_toZSMul_14_; lean_object* v___f_15_; 
v_toZSMul_14_ = lean_ctor_get(v_inst_13_, 3);
lean_inc(v_toZSMul_14_);
lean_dec_ref(v_inst_13_);
v___f_15_ = lean_alloc_closure((void*)(lp_mathlib_ZSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_15_, 0, v_toZSMul_14_);
return v___f_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMulActionIntOfSubtractionMonoid(lean_object* v_M_16_, lean_object* v_inst_17_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = lp_mathlib_instMulActionIntOfSubtractionMonoid___redArg(v_inst_17_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSMulWithZeroInt___redArg(lean_object* v_inst_19_){
_start:
{
lean_object* v_toZSMul_20_; lean_object* v___f_21_; 
v_toZSMul_20_ = lean_ctor_get(v_inst_19_, 3);
lean_inc(v_toZSMul_20_);
lean_dec_ref(v_inst_19_);
v___f_21_ = lean_alloc_closure((void*)(lp_mathlib_ZSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_21_, 0, v_toZSMul_20_);
return v___f_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSMulWithZeroInt(lean_object* v_M_22_, lean_object* v_inst_23_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = lp_mathlib_instSMulWithZeroInt___redArg(v_inst_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_toNatModule___redArg(lean_object* v_inst_25_){
_start:
{
lean_object* v___x_26_; 
v___x_26_ = lp_mathlib_instMulActionNatOfAddMonoid___redArg(v_inst_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_toNatModule(lean_object* v_M_27_, lean_object* v_inst_28_){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lp_mathlib_instMulActionNatOfAddMonoid___redArg(v_inst_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_toIntModule___redArg(lean_object* v_inst_30_){
_start:
{
lean_object* v_toZSMul_31_; lean_object* v___f_32_; 
v_toZSMul_31_ = lean_ctor_get(v_inst_30_, 3);
lean_inc(v_toZSMul_31_);
lean_dec_ref(v_inst_30_);
v___f_32_ = lean_alloc_closure((void*)(lp_mathlib_ZSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_32_, 0, v_toZSMul_31_);
return v___f_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_toIntModule(lean_object* v_M_33_, lean_object* v_inst_34_){
_start:
{
lean_object* v___x_35_; 
v___x_35_ = lp_mathlib_AddCommGroup_toIntModule___redArg(v_inst_34_);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_addCommMonoidToAddCommGroup___redArg___lam__0(lean_object* v_toIntCast_36_, lean_object* v_inst_37_, lean_object* v_z_38_, lean_object* v_a_39_){
_start:
{
lean_object* v___x_40_; lean_object* v___x_41_; 
v___x_40_ = lean_apply_1(v_toIntCast_36_, v_z_38_);
v___x_41_ = lean_apply_2(v_inst_37_, v___x_40_, v_a_39_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_addCommMonoidToAddCommGroup___redArg___lam__1(lean_object* v_toNeg_42_, lean_object* v_toOne_43_, lean_object* v_inst_44_, lean_object* v_a_45_){
_start:
{
lean_object* v___x_46_; lean_object* v___x_47_; 
v___x_46_ = lean_apply_1(v_toNeg_42_, v_toOne_43_);
v___x_47_ = lean_apply_2(v_inst_44_, v___x_46_, v_a_45_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_addCommMonoidToAddCommGroup___redArg(lean_object* v_inst_48_, lean_object* v_inst_49_, lean_object* v_inst_50_){
_start:
{
lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v_toNeg_53_; lean_object* v___x_54_; lean_object* v_toAddMonoidWithOne_55_; lean_object* v_toIntCast_56_; lean_object* v_toOne_57_; lean_object* v___f_58_; lean_object* v___f_59_; lean_object* v___x_60_; lean_object* v___x_61_; 
v___x_51_ = lp_mathlib_Ring_toAddCommGroup___redArg(v_inst_48_);
v___x_52_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v___x_51_);
lean_dec_ref(v___x_51_);
v_toNeg_53_ = lean_ctor_get(v___x_52_, 1);
lean_inc(v_toNeg_53_);
lean_dec_ref(v___x_52_);
v___x_54_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_inst_48_);
v_toAddMonoidWithOne_55_ = lean_ctor_get(v___x_54_, 1);
lean_inc_ref(v_toAddMonoidWithOne_55_);
v_toIntCast_56_ = lean_ctor_get(v___x_54_, 0);
lean_inc(v_toIntCast_56_);
lean_dec_ref(v___x_54_);
v_toOne_57_ = lean_ctor_get(v_toAddMonoidWithOne_55_, 2);
lean_inc(v_toOne_57_);
lean_dec_ref(v_toAddMonoidWithOne_55_);
lean_inc(v_inst_50_);
v___f_58_ = lean_alloc_closure((void*)(lp_mathlib_Module_addCommMonoidToAddCommGroup___redArg___lam__0), 4, 2);
lean_closure_set(v___f_58_, 0, v_toIntCast_56_);
lean_closure_set(v___f_58_, 1, v_inst_50_);
v___f_59_ = lean_alloc_closure((void*)(lp_mathlib_Module_addCommMonoidToAddCommGroup___redArg___lam__1), 4, 3);
lean_closure_set(v___f_59_, 0, v_toNeg_53_);
lean_closure_set(v___f_59_, 1, v_toOne_57_);
lean_closure_set(v___f_59_, 2, v_inst_50_);
lean_inc_ref(v___f_59_);
lean_inc_ref(v_inst_49_);
v___x_60_ = lean_alloc_closure((void*)(lp_mathlib_SubNegMonoid_sub_x27), 5, 3);
lean_closure_set(v___x_60_, 0, lean_box(0));
lean_closure_set(v___x_60_, 1, v_inst_49_);
lean_closure_set(v___x_60_, 2, v___f_59_);
v___x_61_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_61_, 0, v_inst_49_);
lean_ctor_set(v___x_61_, 1, v___f_59_);
lean_ctor_set(v___x_61_, 2, v___x_60_);
lean_ctor_set(v___x_61_, 3, v___f_58_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_addCommMonoidToAddCommGroup(lean_object* v_R_62_, lean_object* v_M_63_, lean_object* v_inst_64_, lean_object* v_inst_65_, lean_object* v_inst_66_){
_start:
{
lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v_toNeg_69_; lean_object* v___x_70_; lean_object* v_toAddMonoidWithOne_71_; lean_object* v_toIntCast_72_; lean_object* v_toOne_73_; lean_object* v___f_74_; lean_object* v___f_75_; lean_object* v___x_76_; lean_object* v___x_77_; 
v___x_67_ = lp_mathlib_Ring_toAddCommGroup___redArg(v_inst_64_);
v___x_68_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v___x_67_);
lean_dec_ref(v___x_67_);
v_toNeg_69_ = lean_ctor_get(v___x_68_, 1);
lean_inc(v_toNeg_69_);
lean_dec_ref(v___x_68_);
v___x_70_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_inst_64_);
v_toAddMonoidWithOne_71_ = lean_ctor_get(v___x_70_, 1);
lean_inc_ref(v_toAddMonoidWithOne_71_);
v_toIntCast_72_ = lean_ctor_get(v___x_70_, 0);
lean_inc(v_toIntCast_72_);
lean_dec_ref(v___x_70_);
v_toOne_73_ = lean_ctor_get(v_toAddMonoidWithOne_71_, 2);
lean_inc(v_toOne_73_);
lean_dec_ref(v_toAddMonoidWithOne_71_);
lean_inc(v_inst_66_);
v___f_74_ = lean_alloc_closure((void*)(lp_mathlib_Module_addCommMonoidToAddCommGroup___redArg___lam__0), 4, 2);
lean_closure_set(v___f_74_, 0, v_toIntCast_72_);
lean_closure_set(v___f_74_, 1, v_inst_66_);
v___f_75_ = lean_alloc_closure((void*)(lp_mathlib_Module_addCommMonoidToAddCommGroup___redArg___lam__1), 4, 3);
lean_closure_set(v___f_75_, 0, v_toNeg_69_);
lean_closure_set(v___f_75_, 1, v_toOne_73_);
lean_closure_set(v___f_75_, 2, v_inst_66_);
lean_inc_ref(v___f_75_);
lean_inc_ref(v_inst_65_);
v___x_76_ = lean_alloc_closure((void*)(lp_mathlib_SubNegMonoid_sub_x27), 5, 3);
lean_closure_set(v___x_76_, 0, lean_box(0));
lean_closure_set(v___x_76_, 1, v_inst_65_);
lean_closure_set(v___x_76_, 2, v___f_75_);
v___x_77_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_77_, 0, v_inst_65_);
lean_ctor_set(v___x_77_, 1, v___f_75_);
lean_ctor_set(v___x_77_, 2, v___x_76_);
lean_ctor_set(v___x_77_, 3, v___f_74_);
return v___x_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_uniqueNatModule___redArg(lean_object* v_inst_78_){
_start:
{
lean_object* v___x_79_; 
v___x_79_ = lp_mathlib_instMulActionNatOfAddMonoid___redArg(v_inst_78_);
return v___x_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_uniqueNatModule(lean_object* v_M_80_, lean_object* v_inst_81_){
_start:
{
lean_object* v___x_82_; 
v___x_82_ = lp_mathlib_instMulActionNatOfAddMonoid___redArg(v_inst_81_);
return v___x_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_uniqueIntModule___redArg(lean_object* v_inst_83_){
_start:
{
lean_object* v___x_84_; 
v___x_84_ = lp_mathlib_AddCommGroup_toIntModule___redArg(v_inst_83_);
return v___x_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_uniqueIntModule(lean_object* v_M_85_, lean_object* v_inst_86_){
_start:
{
lean_object* v___x_87_; 
v___x_87_ = lp_mathlib_AddCommGroup_toIntModule___redArg(v_inst_86_);
return v___x_87_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Lemmas(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_NatInt(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Module_NatInt(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Cast_Lemmas(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Module_NatInt(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Cast_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_NatInt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Module_NatInt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Module_NatInt(builtin);
}
#ifdef __cplusplus
}
#endif
