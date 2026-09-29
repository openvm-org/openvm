// Lean compiler output
// Module: Mathlib.Algebra.Order.Group.OrderIso
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Units.Equiv public import Mathlib.Algebra.Order.Group.Unbundled.Basic public import Mathlib.Order.Hom.Basic
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
lean_object* lp_mathlib_Equiv_subRight___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_mulLeft___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_mulRight___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_addLeft___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_divLeft___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_addRight___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_subLeft___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_divRight___redArg(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_OrderIso_inv___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OrderIso_inv___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_inv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_inv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_inv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_inv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_neg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_neg___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_neg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_neg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_divLeft___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_divLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_subLeft___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_subLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_mulRight___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_mulRight___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_mulRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_mulRight___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_addRight___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_addRight___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_addRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_addRight___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_divRight___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_divRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_subRight___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_subRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_mulLeft___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_mulLeft___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_mulLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_mulLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_addLeft___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_addLeft___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_addLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_addLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_OrderIso_inv___redArg___closed__0(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_inv___redArg(lean_object* v_inst_2_){
_start:
{
lean_object* v_toInv_3_; lean_object* v___x_4_; lean_object* v___x_5_; lean_object* v___x_6_; 
v_toInv_3_ = lean_ctor_get(v_inst_2_, 1);
lean_inc_n(v_toInv_3_, 2);
v___x_4_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4_, 0, v_toInv_3_);
lean_ctor_set(v___x_4_, 1, v_toInv_3_);
v___x_5_ = lean_obj_once(&lp_mathlib_OrderIso_inv___redArg___closed__0, &lp_mathlib_OrderIso_inv___redArg___closed__0_once, _init_lp_mathlib_OrderIso_inv___redArg___closed__0);
v___x_6_ = lp_mathlib_Equiv_trans___redArg(v___x_4_, v___x_5_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_inv___redArg___boxed(lean_object* v_inst_7_){
_start:
{
lean_object* v_res_8_; 
v_res_8_ = lp_mathlib_OrderIso_inv___redArg(v_inst_7_);
lean_dec_ref(v_inst_7_);
return v_res_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_inv(lean_object* v_00_u03b1_9_, lean_object* v_inst_10_, lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_inst_13_){
_start:
{
lean_object* v___x_14_; 
v___x_14_ = lp_mathlib_OrderIso_inv___redArg(v_inst_10_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_inv___boxed(lean_object* v_00_u03b1_15_, lean_object* v_inst_16_, lean_object* v_inst_17_, lean_object* v_inst_18_, lean_object* v_inst_19_){
_start:
{
lean_object* v_res_20_; 
v_res_20_ = lp_mathlib_OrderIso_inv(v_00_u03b1_15_, v_inst_16_, v_inst_17_, v_inst_18_, v_inst_19_);
lean_dec_ref(v_inst_16_);
return v_res_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_neg___redArg(lean_object* v_inst_21_){
_start:
{
lean_object* v_toNeg_22_; lean_object* v___x_23_; lean_object* v___x_24_; lean_object* v___x_25_; 
v_toNeg_22_ = lean_ctor_get(v_inst_21_, 1);
lean_inc_n(v_toNeg_22_, 2);
v___x_23_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_23_, 0, v_toNeg_22_);
lean_ctor_set(v___x_23_, 1, v_toNeg_22_);
v___x_24_ = lean_obj_once(&lp_mathlib_OrderIso_inv___redArg___closed__0, &lp_mathlib_OrderIso_inv___redArg___closed__0_once, _init_lp_mathlib_OrderIso_inv___redArg___closed__0);
v___x_25_ = lp_mathlib_Equiv_trans___redArg(v___x_23_, v___x_24_);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_neg___redArg___boxed(lean_object* v_inst_26_){
_start:
{
lean_object* v_res_27_; 
v_res_27_ = lp_mathlib_OrderIso_neg___redArg(v_inst_26_);
lean_dec_ref(v_inst_26_);
return v_res_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_neg(lean_object* v_00_u03b1_28_, lean_object* v_inst_29_, lean_object* v_inst_30_, lean_object* v_inst_31_, lean_object* v_inst_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = lp_mathlib_OrderIso_neg___redArg(v_inst_29_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_neg___boxed(lean_object* v_00_u03b1_34_, lean_object* v_inst_35_, lean_object* v_inst_36_, lean_object* v_inst_37_, lean_object* v_inst_38_){
_start:
{
lean_object* v_res_39_; 
v_res_39_ = lp_mathlib_OrderIso_neg(v_00_u03b1_34_, v_inst_35_, v_inst_36_, v_inst_37_, v_inst_38_);
lean_dec_ref(v_inst_35_);
return v_res_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_divLeft___redArg(lean_object* v_inst_40_, lean_object* v_a_41_){
_start:
{
lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___x_44_; 
v___x_42_ = lp_mathlib_Equiv_divLeft___redArg(v_inst_40_, v_a_41_);
v___x_43_ = lean_obj_once(&lp_mathlib_OrderIso_inv___redArg___closed__0, &lp_mathlib_OrderIso_inv___redArg___closed__0_once, _init_lp_mathlib_OrderIso_inv___redArg___closed__0);
v___x_44_ = lp_mathlib_Equiv_trans___redArg(v___x_42_, v___x_43_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_divLeft(lean_object* v_00_u03b1_45_, lean_object* v_inst_46_, lean_object* v_inst_47_, lean_object* v_inst_48_, lean_object* v_inst_49_, lean_object* v_a_50_){
_start:
{
lean_object* v___x_51_; 
v___x_51_ = lp_mathlib_OrderIso_divLeft___redArg(v_inst_46_, v_a_50_);
return v___x_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_subLeft___redArg(lean_object* v_inst_52_, lean_object* v_a_53_){
_start:
{
lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; 
v___x_54_ = lp_mathlib_Equiv_subLeft___redArg(v_inst_52_, v_a_53_);
v___x_55_ = lean_obj_once(&lp_mathlib_OrderIso_inv___redArg___closed__0, &lp_mathlib_OrderIso_inv___redArg___closed__0_once, _init_lp_mathlib_OrderIso_inv___redArg___closed__0);
v___x_56_ = lp_mathlib_Equiv_trans___redArg(v___x_54_, v___x_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_subLeft(lean_object* v_00_u03b1_57_, lean_object* v_inst_58_, lean_object* v_inst_59_, lean_object* v_inst_60_, lean_object* v_inst_61_, lean_object* v_a_62_){
_start:
{
lean_object* v___x_63_; 
v___x_63_ = lp_mathlib_OrderIso_subLeft___redArg(v_inst_58_, v_a_62_);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_mulRight___redArg(lean_object* v_inst_64_, lean_object* v_a_65_){
_start:
{
lean_object* v___x_66_; 
v___x_66_ = lp_mathlib_Equiv_mulRight___redArg(v_inst_64_, v_a_65_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_mulRight___redArg___boxed(lean_object* v_inst_67_, lean_object* v_a_68_){
_start:
{
lean_object* v_res_69_; 
v_res_69_ = lp_mathlib_OrderIso_mulRight___redArg(v_inst_67_, v_a_68_);
lean_dec_ref(v_inst_67_);
return v_res_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_mulRight(lean_object* v_00_u03b1_70_, lean_object* v_inst_71_, lean_object* v_inst_72_, lean_object* v_inst_73_, lean_object* v_a_74_){
_start:
{
lean_object* v___x_75_; 
v___x_75_ = lp_mathlib_Equiv_mulRight___redArg(v_inst_71_, v_a_74_);
return v___x_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_mulRight___boxed(lean_object* v_00_u03b1_76_, lean_object* v_inst_77_, lean_object* v_inst_78_, lean_object* v_inst_79_, lean_object* v_a_80_){
_start:
{
lean_object* v_res_81_; 
v_res_81_ = lp_mathlib_OrderIso_mulRight(v_00_u03b1_76_, v_inst_77_, v_inst_78_, v_inst_79_, v_a_80_);
lean_dec_ref(v_inst_77_);
return v_res_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_addRight___redArg(lean_object* v_inst_82_, lean_object* v_a_83_){
_start:
{
lean_object* v___x_84_; 
v___x_84_ = lp_mathlib_Equiv_addRight___redArg(v_inst_82_, v_a_83_);
return v___x_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_addRight___redArg___boxed(lean_object* v_inst_85_, lean_object* v_a_86_){
_start:
{
lean_object* v_res_87_; 
v_res_87_ = lp_mathlib_OrderIso_addRight___redArg(v_inst_85_, v_a_86_);
lean_dec_ref(v_inst_85_);
return v_res_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_addRight(lean_object* v_00_u03b1_88_, lean_object* v_inst_89_, lean_object* v_inst_90_, lean_object* v_inst_91_, lean_object* v_a_92_){
_start:
{
lean_object* v___x_93_; 
v___x_93_ = lp_mathlib_Equiv_addRight___redArg(v_inst_89_, v_a_92_);
return v___x_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_addRight___boxed(lean_object* v_00_u03b1_94_, lean_object* v_inst_95_, lean_object* v_inst_96_, lean_object* v_inst_97_, lean_object* v_a_98_){
_start:
{
lean_object* v_res_99_; 
v_res_99_ = lp_mathlib_OrderIso_addRight(v_00_u03b1_94_, v_inst_95_, v_inst_96_, v_inst_97_, v_a_98_);
lean_dec_ref(v_inst_95_);
return v_res_99_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_divRight___redArg(lean_object* v_inst_100_, lean_object* v_a_101_){
_start:
{
lean_object* v___x_102_; 
v___x_102_ = lp_mathlib_Equiv_divRight___redArg(v_inst_100_, v_a_101_);
return v___x_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_divRight(lean_object* v_00_u03b1_103_, lean_object* v_inst_104_, lean_object* v_inst_105_, lean_object* v_inst_106_, lean_object* v_a_107_){
_start:
{
lean_object* v___x_108_; 
v___x_108_ = lp_mathlib_Equiv_divRight___redArg(v_inst_104_, v_a_107_);
return v___x_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_subRight___redArg(lean_object* v_inst_109_, lean_object* v_a_110_){
_start:
{
lean_object* v___x_111_; 
v___x_111_ = lp_mathlib_Equiv_subRight___redArg(v_inst_109_, v_a_110_);
return v___x_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_subRight(lean_object* v_00_u03b1_112_, lean_object* v_inst_113_, lean_object* v_inst_114_, lean_object* v_inst_115_, lean_object* v_a_116_){
_start:
{
lean_object* v___x_117_; 
v___x_117_ = lp_mathlib_Equiv_subRight___redArg(v_inst_113_, v_a_116_);
return v___x_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_mulLeft___redArg(lean_object* v_inst_118_, lean_object* v_a_119_){
_start:
{
lean_object* v___x_120_; 
v___x_120_ = lp_mathlib_Equiv_mulLeft___redArg(v_inst_118_, v_a_119_);
return v___x_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_mulLeft___redArg___boxed(lean_object* v_inst_121_, lean_object* v_a_122_){
_start:
{
lean_object* v_res_123_; 
v_res_123_ = lp_mathlib_OrderIso_mulLeft___redArg(v_inst_121_, v_a_122_);
lean_dec_ref(v_inst_121_);
return v_res_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_mulLeft(lean_object* v_00_u03b1_124_, lean_object* v_inst_125_, lean_object* v_inst_126_, lean_object* v_inst_127_, lean_object* v_a_128_){
_start:
{
lean_object* v___x_129_; 
v___x_129_ = lp_mathlib_Equiv_mulLeft___redArg(v_inst_125_, v_a_128_);
return v___x_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_mulLeft___boxed(lean_object* v_00_u03b1_130_, lean_object* v_inst_131_, lean_object* v_inst_132_, lean_object* v_inst_133_, lean_object* v_a_134_){
_start:
{
lean_object* v_res_135_; 
v_res_135_ = lp_mathlib_OrderIso_mulLeft(v_00_u03b1_130_, v_inst_131_, v_inst_132_, v_inst_133_, v_a_134_);
lean_dec_ref(v_inst_131_);
return v_res_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_addLeft___redArg(lean_object* v_inst_136_, lean_object* v_a_137_){
_start:
{
lean_object* v___x_138_; 
v___x_138_ = lp_mathlib_Equiv_addLeft___redArg(v_inst_136_, v_a_137_);
return v___x_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_addLeft___redArg___boxed(lean_object* v_inst_139_, lean_object* v_a_140_){
_start:
{
lean_object* v_res_141_; 
v_res_141_ = lp_mathlib_OrderIso_addLeft___redArg(v_inst_139_, v_a_140_);
lean_dec_ref(v_inst_139_);
return v_res_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_addLeft(lean_object* v_00_u03b1_142_, lean_object* v_inst_143_, lean_object* v_inst_144_, lean_object* v_inst_145_, lean_object* v_a_146_){
_start:
{
lean_object* v___x_147_; 
v___x_147_ = lp_mathlib_Equiv_addLeft___redArg(v_inst_143_, v_a_146_);
return v___x_147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_addLeft___boxed(lean_object* v_00_u03b1_148_, lean_object* v_inst_149_, lean_object* v_inst_150_, lean_object* v_inst_151_, lean_object* v_a_152_){
_start:
{
lean_object* v_res_153_; 
v_res_153_ = lp_mathlib_OrderIso_addLeft(v_00_u03b1_148_, v_inst_149_, v_inst_150_, v_inst_151_, v_a_152_);
lean_dec_ref(v_inst_149_);
return v_res_153_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Units_Equiv(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Hom_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_OrderIso(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Units_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Order_Group_OrderIso(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Units_Equiv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Hom_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_OrderIso(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Units_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_OrderIso(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Order_Group_OrderIso(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Order_Group_OrderIso(builtin);
}
#ifdef __cplusplus
}
#endif
