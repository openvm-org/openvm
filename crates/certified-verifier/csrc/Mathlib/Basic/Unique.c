// Lean compiler output
// Module: Mathlib.Basic.Unique
// Imports: public import Init public meta import Init public import Mathlib.Basic.IsEmpty.Defs public import Mathlib.Logic.Function.Basic public import Mathlib.Tactic.Inhabit
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
lean_object* l_Fin_instInhabited___redArg(lean_object*);
lean_object* l_Pi_instInhabited___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfSubsingleton___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfSubsingleton___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfSubsingleton(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfSubsingleton___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instUnique;
LEAN_EXPORT lean_object* lp_mathlib_uniqueProp(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueTrue;
LEAN_EXPORT lean_object* lp_mathlib_Unique_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_instInhabited___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_mk_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_mk_x27___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_mk_x27(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_mk_x27___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_unique___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_unique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_unique(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_uniqueOfIsEmpty___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_uniqueOfIsEmpty___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Pi_uniqueOfIsEmpty___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Pi_uniqueOfIsEmpty___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Pi_uniqueOfIsEmpty___closed__0 = (const lean_object*)&lp_mathlib_Pi_uniqueOfIsEmpty___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Pi_uniqueOfIsEmpty(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_unique___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_unique(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_unique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_unique___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_unique(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_unique___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_uniqueOfSurjectiveConst___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_uniqueOfSurjectiveConst___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_uniqueOfSurjectiveConst(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_uniqueOfSurjectiveConst___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueElim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option_instUniqueOfIsEmpty(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_subtypeEq___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_subtypeEq___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_subtypeEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_subtypeEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_subtypeEq_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_subtypeEq_x27___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_subtypeEq_x27(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_subtypeEq_x27___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Fin_instUnique___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Fin_instUnique___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Fin_instUnique;
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfSubsingleton___redArg(lean_object* v_a_1_){
_start:
{
lean_inc(v_a_1_);
return v_a_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfSubsingleton___redArg___boxed(lean_object* v_a_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_uniqueOfSubsingleton___redArg(v_a_2_);
lean_dec(v_a_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfSubsingleton(lean_object* v_00_u03b1_4_, lean_object* v_inst_5_, lean_object* v_a_6_){
_start:
{
lean_inc(v_a_6_);
return v_a_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfSubsingleton___boxed(lean_object* v_00_u03b1_7_, lean_object* v_inst_8_, lean_object* v_a_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_mathlib_uniqueOfSubsingleton(v_00_u03b1_7_, v_inst_8_, v_a_9_);
lean_dec(v_a_9_);
return v_res_10_;
}
}
static lean_object* _init_lp_mathlib_PUnit_instUnique(void){
_start:
{
lean_object* v___x_11_; 
v___x_11_ = lean_box(0);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueProp(lean_object* v_p_12_, lean_object* v_h_13_){
_start:
{
lean_object* v___x_14_; 
v___x_14_ = lean_box(0);
return v___x_14_;
}
}
static lean_object* _init_lp_mathlib_instUniqueTrue(void){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lean_box(0);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_instInhabited___redArg(lean_object* v_inst_16_){
_start:
{
lean_inc(v_inst_16_);
return v_inst_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_instInhabited___redArg___boxed(lean_object* v_inst_17_){
_start:
{
lean_object* v_res_18_; 
v_res_18_ = lp_mathlib_Unique_instInhabited___redArg(v_inst_17_);
lean_dec(v_inst_17_);
return v_res_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_instInhabited(lean_object* v_00_u03b1_19_, lean_object* v_inst_20_){
_start:
{
lean_inc(v_inst_20_);
return v_inst_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_instInhabited___boxed(lean_object* v_00_u03b1_21_, lean_object* v_inst_22_){
_start:
{
lean_object* v_res_23_; 
v_res_23_ = lp_mathlib_Unique_instInhabited(v_00_u03b1_21_, v_inst_22_);
lean_dec(v_inst_22_);
return v_res_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_mk_x27___redArg(lean_object* v_h_u2081_24_){
_start:
{
lean_inc(v_h_u2081_24_);
return v_h_u2081_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_mk_x27___redArg___boxed(lean_object* v_h_u2081_25_){
_start:
{
lean_object* v_res_26_; 
v_res_26_ = lp_mathlib_Unique_mk_x27___redArg(v_h_u2081_25_);
lean_dec(v_h_u2081_25_);
return v_res_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_mk_x27(lean_object* v_00_u03b1_27_, lean_object* v_h_u2081_28_, lean_object* v_inst_29_){
_start:
{
lean_inc(v_h_u2081_28_);
return v_h_u2081_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_mk_x27___boxed(lean_object* v_00_u03b1_30_, lean_object* v_h_u2081_31_, lean_object* v_inst_32_){
_start:
{
lean_object* v_res_33_; 
v_res_33_ = lp_mathlib_Unique_mk_x27(v_00_u03b1_30_, v_h_u2081_31_, v_inst_32_);
lean_dec(v_h_u2081_31_);
return v_res_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_unique___redArg___lam__0(lean_object* v_inst_34_, lean_object* v_a_35_){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lean_apply_1(v_inst_34_, v_a_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_unique___redArg(lean_object* v_inst_37_){
_start:
{
lean_object* v___f_38_; lean_object* v___f_39_; 
v___f_38_ = lean_alloc_closure((void*)(lp_mathlib_Pi_unique___redArg___lam__0), 2, 1);
lean_closure_set(v___f_38_, 0, v_inst_37_);
v___f_39_ = lean_alloc_closure((void*)(l_Pi_instInhabited___redArg___lam__0), 2, 1);
lean_closure_set(v___f_39_, 0, v___f_38_);
return v___f_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_unique(lean_object* v_00_u03b1_40_, lean_object* v_00_u03b2_41_, lean_object* v_inst_42_){
_start:
{
lean_object* v___x_43_; 
v___x_43_ = lp_mathlib_Pi_unique___redArg(v_inst_42_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_uniqueOfIsEmpty___lam__0(lean_object* v_a_44_){
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_uniqueOfIsEmpty___lam__0___boxed(lean_object* v_a_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_mathlib_Pi_uniqueOfIsEmpty___lam__0(v_a_45_);
lean_dec(v_a_45_);
return v_res_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_uniqueOfIsEmpty(lean_object* v_00_u03b1_48_, lean_object* v_inst_49_, lean_object* v_00_u03b2_50_){
_start:
{
lean_object* v___f_51_; 
v___f_51_ = ((lean_object*)(lp_mathlib_Pi_uniqueOfIsEmpty___closed__0));
return v___f_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_unique___redArg(lean_object* v_f_52_, lean_object* v_inst_53_){
_start:
{
lean_object* v___x_54_; 
v___x_54_ = lean_apply_1(v_f_52_, v_inst_53_);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_unique(lean_object* v_00_u03b2_55_, lean_object* v_00_u03b1_56_, lean_object* v_f_57_, lean_object* v_hf_58_, lean_object* v_inst_59_){
_start:
{
lean_object* v___x_60_; 
v___x_60_ = lean_apply_1(v_f_57_, v_inst_59_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_unique___redArg(lean_object* v_inst_61_){
_start:
{
lean_inc(v_inst_61_);
return v_inst_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_unique___redArg___boxed(lean_object* v_inst_62_){
_start:
{
lean_object* v_res_63_; 
v_res_63_ = lp_mathlib_Function_Injective_unique___redArg(v_inst_62_);
lean_dec(v_inst_62_);
return v_res_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_unique(lean_object* v_00_u03b1_64_, lean_object* v_00_u03b2_65_, lean_object* v_f_66_, lean_object* v_inst_67_, lean_object* v_inst_68_, lean_object* v_hf_69_){
_start:
{
lean_inc(v_inst_67_);
return v_inst_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_unique___boxed(lean_object* v_00_u03b1_70_, lean_object* v_00_u03b2_71_, lean_object* v_f_72_, lean_object* v_inst_73_, lean_object* v_inst_74_, lean_object* v_hf_75_){
_start:
{
lean_object* v_res_76_; 
v_res_76_ = lp_mathlib_Function_Injective_unique(v_00_u03b1_70_, v_00_u03b2_71_, v_f_72_, v_inst_73_, v_inst_74_, v_hf_75_);
lean_dec(v_inst_73_);
lean_dec(v_f_72_);
return v_res_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_uniqueOfSurjectiveConst___redArg(lean_object* v_b_77_){
_start:
{
lean_inc(v_b_77_);
return v_b_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_uniqueOfSurjectiveConst___redArg___boxed(lean_object* v_b_78_){
_start:
{
lean_object* v_res_79_; 
v_res_79_ = lp_mathlib_Function_Surjective_uniqueOfSurjectiveConst___redArg(v_b_78_);
lean_dec(v_b_78_);
return v_res_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_uniqueOfSurjectiveConst(lean_object* v_00_u03b1_80_, lean_object* v_00_u03b2_81_, lean_object* v_b_82_, lean_object* v_h_83_){
_start:
{
lean_inc(v_b_82_);
return v_b_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_uniqueOfSurjectiveConst___boxed(lean_object* v_00_u03b1_84_, lean_object* v_00_u03b2_85_, lean_object* v_b_86_, lean_object* v_h_87_){
_start:
{
lean_object* v_res_88_; 
v_res_88_ = lp_mathlib_Function_Surjective_uniqueOfSurjectiveConst(v_00_u03b1_84_, v_00_u03b2_85_, v_b_86_, v_h_87_);
lean_dec(v_b_86_);
return v_res_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueElim___redArg(lean_object* v_x_89_){
_start:
{
lean_inc(v_x_89_);
return v_x_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueElim___redArg___boxed(lean_object* v_x_90_){
_start:
{
lean_object* v_res_91_; 
v_res_91_ = lp_mathlib_uniqueElim___redArg(v_x_90_);
lean_dec(v_x_90_);
return v_res_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueElim(lean_object* v_00_u03b9_92_, lean_object* v_00_u03b1_93_, lean_object* v_inst_94_, lean_object* v_x_95_, lean_object* v_i_96_){
_start:
{
lean_inc(v_x_95_);
return v_x_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueElim___boxed(lean_object* v_00_u03b9_97_, lean_object* v_00_u03b1_98_, lean_object* v_inst_99_, lean_object* v_x_100_, lean_object* v_i_101_){
_start:
{
lean_object* v_res_102_; 
v_res_102_ = lp_mathlib_uniqueElim(v_00_u03b9_97_, v_00_u03b1_98_, v_inst_99_, v_x_100_, v_i_101_);
lean_dec(v_i_101_);
lean_dec(v_x_100_);
lean_dec(v_inst_99_);
return v_res_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option_instUniqueOfIsEmpty(lean_object* v_00_u03b1_103_, lean_object* v_inst_104_){
_start:
{
lean_object* v___x_105_; 
v___x_105_ = lean_box(0);
return v___x_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_subtypeEq___redArg(lean_object* v_y_106_){
_start:
{
lean_inc(v_y_106_);
return v_y_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_subtypeEq___redArg___boxed(lean_object* v_y_107_){
_start:
{
lean_object* v_res_108_; 
v_res_108_ = lp_mathlib_Unique_subtypeEq___redArg(v_y_107_);
lean_dec(v_y_107_);
return v_res_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_subtypeEq(lean_object* v_00_u03b1_109_, lean_object* v_y_110_){
_start:
{
lean_inc(v_y_110_);
return v_y_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_subtypeEq___boxed(lean_object* v_00_u03b1_111_, lean_object* v_y_112_){
_start:
{
lean_object* v_res_113_; 
v_res_113_ = lp_mathlib_Unique_subtypeEq(v_00_u03b1_111_, v_y_112_);
lean_dec(v_y_112_);
return v_res_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_subtypeEq_x27___redArg(lean_object* v_y_114_){
_start:
{
lean_inc(v_y_114_);
return v_y_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_subtypeEq_x27___redArg___boxed(lean_object* v_y_115_){
_start:
{
lean_object* v_res_116_; 
v_res_116_ = lp_mathlib_Unique_subtypeEq_x27___redArg(v_y_115_);
lean_dec(v_y_115_);
return v_res_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_subtypeEq_x27(lean_object* v_00_u03b1_117_, lean_object* v_y_118_){
_start:
{
lean_inc(v_y_118_);
return v_y_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_subtypeEq_x27___boxed(lean_object* v_00_u03b1_119_, lean_object* v_y_120_){
_start:
{
lean_object* v_res_121_; 
v_res_121_ = lp_mathlib_Unique_subtypeEq_x27(v_00_u03b1_119_, v_y_120_);
lean_dec(v_y_120_);
return v_res_121_;
}
}
static lean_object* _init_lp_mathlib_Fin_instUnique___closed__0(void){
_start:
{
lean_object* v___x_122_; lean_object* v___x_123_; 
v___x_122_ = lean_unsigned_to_nat(1u);
v___x_123_ = l_Fin_instInhabited___redArg(v___x_122_);
return v___x_123_;
}
}
static lean_object* _init_lp_mathlib_Fin_instUnique(void){
_start:
{
lean_object* v___x_124_; 
v___x_124_ = lean_obj_once(&lp_mathlib_Fin_instUnique___closed__0, &lp_mathlib_Fin_instUnique___closed__0_once, _init_lp_mathlib_Fin_instUnique___closed__0);
return v___x_124_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_IsEmpty_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Function_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Inhabit(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Basic_Unique(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_IsEmpty_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Function_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Inhabit(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_PUnit_instUnique = _init_lp_mathlib_PUnit_instUnique();
lean_mark_persistent(lp_mathlib_PUnit_instUnique);
lp_mathlib_instUniqueTrue = _init_lp_mathlib_instUniqueTrue();
lp_mathlib_Fin_instUnique = _init_lp_mathlib_Fin_instUnique();
lean_mark_persistent(lp_mathlib_Fin_instUnique);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Basic_Unique(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Basic_IsEmpty_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Function_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Inhabit(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Basic_Unique(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_IsEmpty_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Function_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Inhabit(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Unique(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Basic_Unique(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Basic_Unique(builtin);
}
#ifdef __cplusplus
}
#endif
