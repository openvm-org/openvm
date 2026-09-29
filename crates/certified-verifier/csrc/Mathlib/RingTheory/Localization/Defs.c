// Lean compiler output
// Module: Mathlib.RingTheory.Localization.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.BigOperators.Group.Finset.Defs public import Mathlib.Algebra.Regular.Basic public import Mathlib.Algebra.Ring.NonZeroDivisors public import Mathlib.Data.Fintype.Prod public import Mathlib.GroupTheory.MonoidLocalization.Divisibility public import Mathlib.GroupTheory.MonoidLocalization.MonoidWithZero public import Mathlib.RingTheory.OreLocalization.Ring public import Mathlib.Tactic.Ring
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
lean_object* lp_mathlib_OreLocalization_instInhabited___redArg(lean_object*);
lean_object* lp_mathlib_instMulZeroOneClassOfSemiring___redArg(lean_object*);
lean_object* lp_mathlib_uniqueOfZeroEqOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsLocalization_toLocalizationMap___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsLocalization_toLocalizationMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsLocalization_toLocalizationMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsLocalization_toLocalizationMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsLocalization_uniqueOfZeroMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsLocalization_uniqueOfZeroMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsLocalization_uniqueOfZeroMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_instUniqueLocalization___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_instUniqueLocalization___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_instUniqueLocalization(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_instUniqueLocalization___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_mkAddMonoidHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_mkAddMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_mkAddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_mkAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsLocalization_toLocalizationMap___redArg___lam__0(lean_object* v_algebraMap_1_, lean_object* v___y_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_apply_1(v_algebraMap_1_, v___y_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsLocalization_toLocalizationMap___redArg(lean_object* v_inst_4_){
_start:
{
lean_object* v_algebraMap_5_; lean_object* v___f_6_; 
v_algebraMap_5_ = lean_ctor_get(v_inst_4_, 1);
lean_inc(v_algebraMap_5_);
lean_dec_ref(v_inst_4_);
v___f_6_ = lean_alloc_closure((void*)(lp_mathlib_IsLocalization_toLocalizationMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_6_, 0, v_algebraMap_5_);
return v___f_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsLocalization_toLocalizationMap(lean_object* v_R_7_, lean_object* v_inst_8_, lean_object* v_M_9_, lean_object* v_S_10_, lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_inst_13_){
_start:
{
lean_object* v_algebraMap_14_; lean_object* v___f_15_; 
v_algebraMap_14_ = lean_ctor_get(v_inst_12_, 1);
lean_inc(v_algebraMap_14_);
lean_dec_ref(v_inst_12_);
v___f_15_ = lean_alloc_closure((void*)(lp_mathlib_IsLocalization_toLocalizationMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_15_, 0, v_algebraMap_14_);
return v___f_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsLocalization_toLocalizationMap___boxed(lean_object* v_R_16_, lean_object* v_inst_17_, lean_object* v_M_18_, lean_object* v_S_19_, lean_object* v_inst_20_, lean_object* v_inst_21_, lean_object* v_inst_22_){
_start:
{
lean_object* v_res_23_; 
v_res_23_ = lp_mathlib_IsLocalization_toLocalizationMap(v_R_16_, v_inst_17_, v_M_18_, v_S_19_, v_inst_20_, v_inst_21_, v_inst_22_);
lean_dec_ref(v_inst_20_);
lean_dec_ref(v_inst_17_);
return v_res_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsLocalization_uniqueOfZeroMem___redArg(lean_object* v_inst_24_){
_start:
{
lean_object* v___x_25_; lean_object* v___x_26_; 
v___x_25_ = lp_mathlib_instMulZeroOneClassOfSemiring___redArg(v_inst_24_);
v___x_26_ = lp_mathlib_uniqueOfZeroEqOne___redArg(v___x_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsLocalization_uniqueOfZeroMem(lean_object* v_R_27_, lean_object* v_inst_28_, lean_object* v_M_29_, lean_object* v_S_30_, lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_h_34_){
_start:
{
lean_object* v___x_35_; 
v___x_35_ = lp_mathlib_IsLocalization_uniqueOfZeroMem___redArg(v_inst_31_);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsLocalization_uniqueOfZeroMem___boxed(lean_object* v_R_36_, lean_object* v_inst_37_, lean_object* v_M_38_, lean_object* v_S_39_, lean_object* v_inst_40_, lean_object* v_inst_41_, lean_object* v_inst_42_, lean_object* v_h_43_){
_start:
{
lean_object* v_res_44_; 
v_res_44_ = lp_mathlib_IsLocalization_uniqueOfZeroMem(v_R_36_, v_inst_37_, v_M_38_, v_S_39_, v_inst_40_, v_inst_41_, v_inst_42_, v_h_43_);
lean_dec_ref(v_inst_41_);
lean_dec_ref(v_inst_37_);
return v_res_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_instUniqueLocalization___redArg(lean_object* v_inst_45_){
_start:
{
lean_object* v_toMonoid_46_; lean_object* v___x_47_; 
v_toMonoid_46_ = lean_ctor_get(v_inst_45_, 1);
v___x_47_ = lp_mathlib_OreLocalization_instInhabited___redArg(v_toMonoid_46_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_instUniqueLocalization___redArg___boxed(lean_object* v_inst_48_){
_start:
{
lean_object* v_res_49_; 
v_res_49_ = lp_mathlib_Localization_instUniqueLocalization___redArg(v_inst_48_);
lean_dec_ref(v_inst_48_);
return v_res_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_instUniqueLocalization(lean_object* v_R_50_, lean_object* v_inst_51_, lean_object* v_M_52_, lean_object* v_inst_53_){
_start:
{
lean_object* v___x_54_; 
v___x_54_ = lp_mathlib_Localization_instUniqueLocalization___redArg(v_inst_51_);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_instUniqueLocalization___boxed(lean_object* v_R_55_, lean_object* v_inst_56_, lean_object* v_M_57_, lean_object* v_inst_58_){
_start:
{
lean_object* v_res_59_; 
v_res_59_ = lp_mathlib_Localization_instUniqueLocalization(v_R_55_, v_inst_56_, v_M_57_, v_inst_58_);
lean_dec_ref(v_inst_56_);
return v_res_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_mkAddMonoidHom___redArg___lam__0(lean_object* v_b_60_, lean_object* v_a_61_){
_start:
{
lean_object* v___x_62_; 
v___x_62_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_62_, 0, v_a_61_);
lean_ctor_set(v___x_62_, 1, v_b_60_);
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_mkAddMonoidHom___redArg(lean_object* v_b_63_){
_start:
{
lean_object* v___f_64_; 
v___f_64_ = lean_alloc_closure((void*)(lp_mathlib_Localization_mkAddMonoidHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_64_, 0, v_b_63_);
return v___f_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_mkAddMonoidHom(lean_object* v_R_65_, lean_object* v_inst_66_, lean_object* v_M_67_, lean_object* v_b_68_){
_start:
{
lean_object* v___f_69_; 
v___f_69_ = lean_alloc_closure((void*)(lp_mathlib_Localization_mkAddMonoidHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_69_, 0, v_b_68_);
return v___f_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_mkAddMonoidHom___boxed(lean_object* v_R_70_, lean_object* v_inst_71_, lean_object* v_M_72_, lean_object* v_b_73_){
_start:
{
lean_object* v_res_74_; 
v_res_74_ = lp_mathlib_Localization_mkAddMonoidHom(v_R_70_, v_inst_71_, v_M_72_, v_b_73_);
lean_dec_ref(v_inst_71_);
return v_res_74_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Regular_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_NonZeroDivisors(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_Divisibility(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_MonoidWithZero(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_OreLocalization_Ring(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Ring(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_Localization_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Regular_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_NonZeroDivisors(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_Divisibility(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_MonoidWithZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_OreLocalization_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_RingTheory_Localization_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Regular_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_NonZeroDivisors(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_Divisibility(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_MonoidWithZero(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_OreLocalization_Ring(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Ring(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_Localization_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Regular_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_NonZeroDivisors(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_Divisibility(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_MonoidWithZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_OreLocalization_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_Localization_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_RingTheory_Localization_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_RingTheory_Localization_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
