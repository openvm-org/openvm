// Lean compiler output
// Module: VM.Spec.Machine.Field
// Imports: public import Init public meta import Init
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
lean_object* lean_nat_log2(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
uint32_t lean_uint32_of_nat(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_asNat___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_asNat(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_field___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_field(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_fieldChar___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_fieldChar___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_fieldChar(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_fieldChar___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_fieldBits___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_fieldBits___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_fieldBits(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_fieldBits___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint32_t lp_openvm_x2dfv_VM_Spec_Machine_asU32___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_asU32___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint32_t lp_openvm_x2dfv_VM_Spec_Machine_asU32(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_asU32___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_asNat___redArg(lean_object* v_inst_1_, lean_object* v_x_2_){
_start:
{
lean_object* v_asNat_3_; lean_object* v___x_4_; 
v_asNat_3_ = lean_ctor_get(v_inst_1_, 1);
lean_inc_ref(v_asNat_3_);
lean_dec_ref(v_inst_1_);
v___x_4_ = lean_apply_1(v_asNat_3_, v_x_2_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_asNat(lean_object* v_F_5_, lean_object* v_inst_6_, lean_object* v_x_7_){
_start:
{
lean_object* v_asNat_8_; lean_object* v___x_9_; 
v_asNat_8_ = lean_ctor_get(v_inst_6_, 1);
lean_inc_ref(v_asNat_8_);
lean_dec_ref(v_inst_6_);
v___x_9_ = lean_apply_1(v_asNat_8_, v_x_7_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_field___redArg(lean_object* v_inst_10_, lean_object* v_n_11_){
_start:
{
lean_object* v_field_12_; lean_object* v___x_13_; 
v_field_12_ = lean_ctor_get(v_inst_10_, 2);
lean_inc(v_field_12_);
lean_dec_ref(v_inst_10_);
v___x_13_ = lean_apply_1(v_field_12_, v_n_11_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_field(lean_object* v_F_14_, lean_object* v_inst_15_, lean_object* v_n_16_){
_start:
{
lean_object* v_field_17_; lean_object* v___x_18_; 
v_field_17_ = lean_ctor_get(v_inst_15_, 2);
lean_inc(v_field_17_);
lean_dec_ref(v_inst_15_);
v___x_18_ = lean_apply_1(v_field_17_, v_n_16_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_fieldChar___redArg(lean_object* v_inst_19_){
_start:
{
lean_object* v_charP_20_; 
v_charP_20_ = lean_ctor_get(v_inst_19_, 0);
lean_inc(v_charP_20_);
return v_charP_20_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_fieldChar___redArg___boxed(lean_object* v_inst_21_){
_start:
{
lean_object* v_res_22_; 
v_res_22_ = lp_openvm_x2dfv_VM_Spec_Machine_fieldChar___redArg(v_inst_21_);
lean_dec_ref(v_inst_21_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_fieldChar(lean_object* v_F_23_, lean_object* v_inst_24_){
_start:
{
lean_object* v_charP_25_; 
v_charP_25_ = lean_ctor_get(v_inst_24_, 0);
lean_inc(v_charP_25_);
return v_charP_25_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_fieldChar___boxed(lean_object* v_F_26_, lean_object* v_inst_27_){
_start:
{
lean_object* v_res_28_; 
v_res_28_ = lp_openvm_x2dfv_VM_Spec_Machine_fieldChar(v_F_26_, v_inst_27_);
lean_dec_ref(v_inst_27_);
return v_res_28_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_fieldBits___redArg(lean_object* v_inst_29_){
_start:
{
lean_object* v_charP_30_; lean_object* v___x_31_; lean_object* v___x_32_; lean_object* v___x_33_; 
v_charP_30_ = lean_ctor_get(v_inst_29_, 0);
v___x_31_ = lean_nat_log2(v_charP_30_);
v___x_32_ = lean_unsigned_to_nat(1u);
v___x_33_ = lean_nat_add(v___x_31_, v___x_32_);
lean_dec(v___x_31_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_fieldBits___redArg___boxed(lean_object* v_inst_34_){
_start:
{
lean_object* v_res_35_; 
v_res_35_ = lp_openvm_x2dfv_VM_Spec_Machine_fieldBits___redArg(v_inst_34_);
lean_dec_ref(v_inst_34_);
return v_res_35_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_fieldBits(lean_object* v_F_36_, lean_object* v_inst_37_){
_start:
{
lean_object* v_charP_38_; lean_object* v___x_39_; lean_object* v___x_40_; lean_object* v___x_41_; 
v_charP_38_ = lean_ctor_get(v_inst_37_, 0);
v___x_39_ = lean_nat_log2(v_charP_38_);
v___x_40_ = lean_unsigned_to_nat(1u);
v___x_41_ = lean_nat_add(v___x_39_, v___x_40_);
lean_dec(v___x_39_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_fieldBits___boxed(lean_object* v_F_42_, lean_object* v_inst_43_){
_start:
{
lean_object* v_res_44_; 
v_res_44_ = lp_openvm_x2dfv_VM_Spec_Machine_fieldBits(v_F_42_, v_inst_43_);
lean_dec_ref(v_inst_43_);
return v_res_44_;
}
}
LEAN_EXPORT uint32_t lp_openvm_x2dfv_VM_Spec_Machine_asU32___redArg(lean_object* v_inst_45_, lean_object* v_x_46_){
_start:
{
lean_object* v_asNat_47_; lean_object* v___x_48_; uint32_t v___x_49_; 
v_asNat_47_ = lean_ctor_get(v_inst_45_, 1);
lean_inc_ref(v_asNat_47_);
lean_dec_ref(v_inst_45_);
v___x_48_ = lean_apply_1(v_asNat_47_, v_x_46_);
v___x_49_ = lean_uint32_of_nat(v___x_48_);
lean_dec(v___x_48_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_asU32___redArg___boxed(lean_object* v_inst_50_, lean_object* v_x_51_){
_start:
{
uint32_t v_res_52_; lean_object* v_r_53_; 
v_res_52_ = lp_openvm_x2dfv_VM_Spec_Machine_asU32___redArg(v_inst_50_, v_x_51_);
v_r_53_ = lean_box_uint32(v_res_52_);
return v_r_53_;
}
}
LEAN_EXPORT uint32_t lp_openvm_x2dfv_VM_Spec_Machine_asU32(lean_object* v_F_54_, lean_object* v_inst_55_, lean_object* v_x_56_){
_start:
{
uint32_t v___x_57_; 
v___x_57_ = lp_openvm_x2dfv_VM_Spec_Machine_asU32___redArg(v_inst_55_, v_x_56_);
return v___x_57_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_asU32___boxed(lean_object* v_F_58_, lean_object* v_inst_59_, lean_object* v_x_60_){
_start:
{
uint32_t v_res_61_; lean_object* v_r_62_; 
v_res_61_ = lp_openvm_x2dfv_VM_Spec_Machine_asU32(v_F_58_, v_inst_59_, v_x_60_);
v_r_62_ = lean_box_uint32(v_res_61_);
return v_r_62_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_openvm_x2dfv_VM_Spec_Machine_Field(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
lean_initialize_runtime_module();
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
