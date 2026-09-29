// Lean compiler output
// Module: Mathlib.Algebra.GroupWithZero.Action.TransferInstance
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Action.TransferInstance public import Mathlib.Algebra.GroupWithZero.Action.Defs
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
lean_object* lp_mathlib_Equiv_smul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_smulZeroClass___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_smulZeroClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_smulZeroClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_smulWithZero___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_smulWithZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_smulWithZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulActionWithZero___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulActionWithZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulActionWithZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_distribSMul___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_distribSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_distribSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_distribMulAction___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_distribMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_distribMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_distribSMul___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_distribSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_distribSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_distribMulAction___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_distribMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_distribMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_smulZeroClass___redArg(lean_object* v_e_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v___f_3_; 
v___f_3_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_smul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_3_, 0, v_e_1_);
lean_closure_set(v___f_3_, 1, v_inst_2_);
return v___f_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_smulZeroClass(lean_object* v_M_4_, lean_object* v_A_5_, lean_object* v_B_6_, lean_object* v_e_7_, lean_object* v_inst_8_, lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_map__zero_11_){
_start:
{
lean_object* v___f_12_; 
v___f_12_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_smul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_12_, 0, v_e_7_);
lean_closure_set(v___f_12_, 1, v_inst_10_);
return v___f_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_smulZeroClass___boxed(lean_object* v_M_13_, lean_object* v_A_14_, lean_object* v_B_15_, lean_object* v_e_16_, lean_object* v_inst_17_, lean_object* v_inst_18_, lean_object* v_inst_19_, lean_object* v_map__zero_20_){
_start:
{
lean_object* v_res_21_; 
v_res_21_ = lp_mathlib_Equiv_smulZeroClass(v_M_13_, v_A_14_, v_B_15_, v_e_16_, v_inst_17_, v_inst_18_, v_inst_19_, v_map__zero_20_);
lean_dec(v_inst_18_);
lean_dec(v_inst_17_);
return v_res_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_smulWithZero___redArg(lean_object* v_e_22_, lean_object* v_inst_23_){
_start:
{
lean_object* v___f_24_; 
v___f_24_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_smul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_24_, 0, v_e_22_);
lean_closure_set(v___f_24_, 1, v_inst_23_);
return v___f_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_smulWithZero(lean_object* v_M_u2080_25_, lean_object* v_A_26_, lean_object* v_B_27_, lean_object* v_e_28_, lean_object* v_inst_29_, lean_object* v_inst_30_, lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_map__zero_33_){
_start:
{
lean_object* v___f_34_; 
v___f_34_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_smul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_34_, 0, v_e_28_);
lean_closure_set(v___f_34_, 1, v_inst_32_);
return v___f_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_smulWithZero___boxed(lean_object* v_M_u2080_35_, lean_object* v_A_36_, lean_object* v_B_37_, lean_object* v_e_38_, lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_inst_41_, lean_object* v_inst_42_, lean_object* v_map__zero_43_){
_start:
{
lean_object* v_res_44_; 
v_res_44_ = lp_mathlib_Equiv_smulWithZero(v_M_u2080_35_, v_A_36_, v_B_37_, v_e_38_, v_inst_39_, v_inst_40_, v_inst_41_, v_inst_42_, v_map__zero_43_);
lean_dec(v_inst_41_);
lean_dec(v_inst_40_);
lean_dec(v_inst_39_);
return v_res_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulActionWithZero___redArg(lean_object* v_e_45_, lean_object* v_inst_46_){
_start:
{
lean_object* v___f_47_; 
v___f_47_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_smul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_47_, 0, v_e_45_);
lean_closure_set(v___f_47_, 1, v_inst_46_);
return v___f_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulActionWithZero(lean_object* v_M_u2080_48_, lean_object* v_A_49_, lean_object* v_B_50_, lean_object* v_e_51_, lean_object* v_inst_52_, lean_object* v_inst_53_, lean_object* v_inst_54_, lean_object* v_inst_55_, lean_object* v_map__zero_56_){
_start:
{
lean_object* v___f_57_; 
v___f_57_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_smul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_57_, 0, v_e_51_);
lean_closure_set(v___f_57_, 1, v_inst_55_);
return v___f_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulActionWithZero___boxed(lean_object* v_M_u2080_58_, lean_object* v_A_59_, lean_object* v_B_60_, lean_object* v_e_61_, lean_object* v_inst_62_, lean_object* v_inst_63_, lean_object* v_inst_64_, lean_object* v_inst_65_, lean_object* v_map__zero_66_){
_start:
{
lean_object* v_res_67_; 
v_res_67_ = lp_mathlib_Equiv_mulActionWithZero(v_M_u2080_58_, v_A_59_, v_B_60_, v_e_61_, v_inst_62_, v_inst_63_, v_inst_64_, v_inst_65_, v_map__zero_66_);
lean_dec(v_inst_64_);
lean_dec(v_inst_63_);
lean_dec_ref(v_inst_62_);
return v_res_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_distribSMul___redArg(lean_object* v_inst_68_, lean_object* v_e_69_){
_start:
{
lean_object* v___f_70_; 
v___f_70_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_smul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_70_, 0, v_e_69_);
lean_closure_set(v___f_70_, 1, v_inst_68_);
return v___f_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_distribSMul(lean_object* v_M_71_, lean_object* v_A_72_, lean_object* v_B_73_, lean_object* v_inst_74_, lean_object* v_inst_75_, lean_object* v_inst_76_, lean_object* v_e_77_){
_start:
{
lean_object* v___f_78_; 
v___f_78_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_smul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_78_, 0, v_e_77_);
lean_closure_set(v___f_78_, 1, v_inst_76_);
return v___f_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_distribSMul___boxed(lean_object* v_M_79_, lean_object* v_A_80_, lean_object* v_B_81_, lean_object* v_inst_82_, lean_object* v_inst_83_, lean_object* v_inst_84_, lean_object* v_e_85_){
_start:
{
lean_object* v_res_86_; 
v_res_86_ = lp_mathlib_AddEquiv_distribSMul(v_M_79_, v_A_80_, v_B_81_, v_inst_82_, v_inst_83_, v_inst_84_, v_e_85_);
lean_dec_ref(v_inst_83_);
lean_dec_ref(v_inst_82_);
return v_res_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_distribMulAction___redArg(lean_object* v_inst_87_, lean_object* v_e_88_){
_start:
{
lean_object* v___f_89_; 
v___f_89_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_smul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_89_, 0, v_e_88_);
lean_closure_set(v___f_89_, 1, v_inst_87_);
return v___f_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_distribMulAction(lean_object* v_M_90_, lean_object* v_A_91_, lean_object* v_B_92_, lean_object* v_inst_93_, lean_object* v_inst_94_, lean_object* v_inst_95_, lean_object* v_inst_96_, lean_object* v_e_97_){
_start:
{
lean_object* v___f_98_; 
v___f_98_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_smul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_98_, 0, v_e_97_);
lean_closure_set(v___f_98_, 1, v_inst_96_);
return v___f_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_distribMulAction___boxed(lean_object* v_M_99_, lean_object* v_A_100_, lean_object* v_B_101_, lean_object* v_inst_102_, lean_object* v_inst_103_, lean_object* v_inst_104_, lean_object* v_inst_105_, lean_object* v_e_106_){
_start:
{
lean_object* v_res_107_; 
v_res_107_ = lp_mathlib_AddEquiv_distribMulAction(v_M_99_, v_A_100_, v_B_101_, v_inst_102_, v_inst_103_, v_inst_104_, v_inst_105_, v_e_106_);
lean_dec_ref(v_inst_104_);
lean_dec_ref(v_inst_103_);
lean_dec_ref(v_inst_102_);
return v_res_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_distribSMul___redArg(lean_object* v_inst_108_, lean_object* v_e_109_){
_start:
{
lean_object* v___f_110_; 
v___f_110_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_smul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_110_, 0, v_e_109_);
lean_closure_set(v___f_110_, 1, v_inst_108_);
return v___f_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_distribSMul(lean_object* v_M_111_, lean_object* v_A_112_, lean_object* v_B_113_, lean_object* v_inst_114_, lean_object* v_inst_115_, lean_object* v_inst_116_, lean_object* v_e_117_){
_start:
{
lean_object* v___f_118_; 
v___f_118_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_smul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_118_, 0, v_e_117_);
lean_closure_set(v___f_118_, 1, v_inst_116_);
return v___f_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_distribSMul___boxed(lean_object* v_M_119_, lean_object* v_A_120_, lean_object* v_B_121_, lean_object* v_inst_122_, lean_object* v_inst_123_, lean_object* v_inst_124_, lean_object* v_e_125_){
_start:
{
lean_object* v_res_126_; 
v_res_126_ = lp_mathlib_Equiv_distribSMul(v_M_119_, v_A_120_, v_B_121_, v_inst_122_, v_inst_123_, v_inst_124_, v_e_125_);
lean_dec_ref(v_inst_123_);
lean_dec_ref(v_inst_122_);
return v_res_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_distribMulAction___redArg(lean_object* v_inst_127_, lean_object* v_e_128_){
_start:
{
lean_object* v___f_129_; 
v___f_129_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_smul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_129_, 0, v_e_128_);
lean_closure_set(v___f_129_, 1, v_inst_127_);
return v___f_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_distribMulAction(lean_object* v_M_130_, lean_object* v_A_131_, lean_object* v_B_132_, lean_object* v_inst_133_, lean_object* v_inst_134_, lean_object* v_inst_135_, lean_object* v_inst_136_, lean_object* v_e_137_){
_start:
{
lean_object* v___f_138_; 
v___f_138_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_smul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_138_, 0, v_e_137_);
lean_closure_set(v___f_138_, 1, v_inst_136_);
return v___f_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_distribMulAction___boxed(lean_object* v_M_139_, lean_object* v_A_140_, lean_object* v_B_141_, lean_object* v_inst_142_, lean_object* v_inst_143_, lean_object* v_inst_144_, lean_object* v_inst_145_, lean_object* v_e_146_){
_start:
{
lean_object* v_res_147_; 
v_res_147_ = lp_mathlib_Equiv_distribMulAction(v_M_139_, v_A_140_, v_B_141_, v_inst_142_, v_inst_143_, v_inst_144_, v_inst_145_, v_e_146_);
lean_dec_ref(v_inst_144_);
lean_dec_ref(v_inst_143_);
lean_dec_ref(v_inst_142_);
return v_res_147_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_TransferInstance(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_TransferInstance(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_TransferInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_TransferInstance(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_TransferInstance(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_TransferInstance(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Action_TransferInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_TransferInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_TransferInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_TransferInstance(builtin);
}
#ifdef __cplusplus
}
#endif
