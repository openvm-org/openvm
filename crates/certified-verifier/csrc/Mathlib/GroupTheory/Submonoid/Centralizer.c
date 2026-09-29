// Lean compiler output
// Module: Mathlib.GroupTheory.Submonoid.Centralizer
// Imports: public import Init public meta import Init public import Mathlib.GroupTheory.Subsemigroup.Centralizer public import Mathlib.GroupTheory.Submonoid.Center
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
lean_object* lp_mathlib_SubmonoidClass_toMonoid___redArg(lean_object*);
lean_object* lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_centralizer(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_centralizer___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_centralizer(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_centralizer___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Submonoid_decidableMemCentralizer___redArg(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_decidableMemCentralizer___redArg___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Submonoid_decidableMemCentralizer(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_decidableMemCentralizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_AddSubmonoid_decidableMemCentralizer___redArg(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_decidableMemCentralizer___redArg___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_AddSubmonoid_decidableMemCentralizer(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_decidableMemCentralizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_closureCommMonoidOfComm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_closureCommMonoidOfComm(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_closureAddCommMonoidOfComm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_closureAddCommMonoidOfComm(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_centralizer(lean_object* v_M_1_, lean_object* v_S_2_, lean_object* v_inst_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_box(0);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_centralizer___boxed(lean_object* v_M_5_, lean_object* v_S_6_, lean_object* v_inst_7_){
_start:
{
lean_object* v_res_8_; 
v_res_8_ = lp_mathlib_Submonoid_centralizer(v_M_5_, v_S_6_, v_inst_7_);
lean_dec_ref(v_inst_7_);
return v_res_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_centralizer(lean_object* v_M_9_, lean_object* v_S_10_, lean_object* v_inst_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lean_box(0);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_centralizer___boxed(lean_object* v_M_13_, lean_object* v_S_14_, lean_object* v_inst_15_){
_start:
{
lean_object* v_res_16_; 
v_res_16_ = lp_mathlib_AddSubmonoid_centralizer(v_M_13_, v_S_14_, v_inst_15_);
lean_dec_ref(v_inst_15_);
return v_res_16_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Submonoid_decidableMemCentralizer___redArg(uint8_t v_inst_17_){
_start:
{
return v_inst_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_decidableMemCentralizer___redArg___boxed(lean_object* v_inst_18_){
_start:
{
uint8_t v_inst_10__boxed_19_; uint8_t v_res_20_; lean_object* v_r_21_; 
v_inst_10__boxed_19_ = lean_unbox(v_inst_18_);
v_res_20_ = lp_mathlib_Submonoid_decidableMemCentralizer___redArg(v_inst_10__boxed_19_);
v_r_21_ = lean_box(v_res_20_);
return v_r_21_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Submonoid_decidableMemCentralizer(lean_object* v_M_22_, lean_object* v_S_23_, lean_object* v_inst_24_, lean_object* v_a_25_, uint8_t v_inst_26_){
_start:
{
return v_inst_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_decidableMemCentralizer___boxed(lean_object* v_M_27_, lean_object* v_S_28_, lean_object* v_inst_29_, lean_object* v_a_30_, lean_object* v_inst_31_){
_start:
{
uint8_t v_inst_14__boxed_32_; uint8_t v_res_33_; lean_object* v_r_34_; 
v_inst_14__boxed_32_ = lean_unbox(v_inst_31_);
v_res_33_ = lp_mathlib_Submonoid_decidableMemCentralizer(v_M_27_, v_S_28_, v_inst_29_, v_a_30_, v_inst_14__boxed_32_);
lean_dec(v_a_30_);
lean_dec_ref(v_inst_29_);
v_r_34_ = lean_box(v_res_33_);
return v_r_34_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_AddSubmonoid_decidableMemCentralizer___redArg(uint8_t v_inst_35_){
_start:
{
return v_inst_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_decidableMemCentralizer___redArg___boxed(lean_object* v_inst_36_){
_start:
{
uint8_t v_inst_10__boxed_37_; uint8_t v_res_38_; lean_object* v_r_39_; 
v_inst_10__boxed_37_ = lean_unbox(v_inst_36_);
v_res_38_ = lp_mathlib_AddSubmonoid_decidableMemCentralizer___redArg(v_inst_10__boxed_37_);
v_r_39_ = lean_box(v_res_38_);
return v_r_39_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_AddSubmonoid_decidableMemCentralizer(lean_object* v_M_40_, lean_object* v_S_41_, lean_object* v_inst_42_, lean_object* v_a_43_, uint8_t v_inst_44_){
_start:
{
return v_inst_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_decidableMemCentralizer___boxed(lean_object* v_M_45_, lean_object* v_S_46_, lean_object* v_inst_47_, lean_object* v_a_48_, lean_object* v_inst_49_){
_start:
{
uint8_t v_inst_14__boxed_50_; uint8_t v_res_51_; lean_object* v_r_52_; 
v_inst_14__boxed_50_ = lean_unbox(v_inst_49_);
v_res_51_ = lp_mathlib_AddSubmonoid_decidableMemCentralizer(v_M_45_, v_S_46_, v_inst_47_, v_a_48_, v_inst_14__boxed_50_);
lean_dec(v_a_48_);
lean_dec_ref(v_inst_47_);
v_r_52_ = lean_box(v_res_51_);
return v_r_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_closureCommMonoidOfComm___redArg(lean_object* v_inst_53_){
_start:
{
lean_object* v___x_54_; 
v___x_54_ = lp_mathlib_SubmonoidClass_toMonoid___redArg(v_inst_53_);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_closureCommMonoidOfComm(lean_object* v_M_55_, lean_object* v_inst_56_, lean_object* v_s_57_, lean_object* v_hcomm_58_){
_start:
{
lean_object* v___x_59_; 
v___x_59_ = lp_mathlib_SubmonoidClass_toMonoid___redArg(v_inst_56_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_closureAddCommMonoidOfComm___redArg(lean_object* v_inst_60_){
_start:
{
lean_object* v___x_61_; 
v___x_61_ = lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(v_inst_60_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_closureAddCommMonoidOfComm(lean_object* v_M_62_, lean_object* v_inst_63_, lean_object* v_s_64_, lean_object* v_hcomm_65_){
_start:
{
lean_object* v___x_66_; 
v___x_66_ = lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(v_inst_63_);
return v___x_66_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Subsemigroup_Centralizer(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Submonoid_Center(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Submonoid_Centralizer(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Subsemigroup_Centralizer(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Submonoid_Center(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_GroupTheory_Submonoid_Centralizer(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_GroupTheory_Subsemigroup_Centralizer(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Submonoid_Center(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_Submonoid_Centralizer(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Subsemigroup_Centralizer(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Submonoid_Center(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Submonoid_Centralizer(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_GroupTheory_Submonoid_Centralizer(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_GroupTheory_Submonoid_Centralizer(builtin);
}
#ifdef __cplusplus
}
#endif
