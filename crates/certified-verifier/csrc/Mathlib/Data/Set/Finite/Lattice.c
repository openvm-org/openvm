// Lean compiler output
// Module: Mathlib.Data.Set.Finite.Lattice
// Imports: public import Init public meta import Init public import Mathlib.Data.Set.Finite.Powerset public import Mathlib.Data.Set.Finite.Range public import Mathlib.Data.Set.Lattice.Image import Mathlib.Data.Fintype.Option
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
lean_object* lp_mathlib_Set_toFinset___redArg(lean_object*);
lean_object* lp_mathlib_Finset_biUnion___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Fintype_subtype___redArg(lean_object*);
lean_object* lp_mathlib_PLift_fintype___redArg(lean_object*);
lean_object* lp_mathlib_Multiset_attach___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeiUnion___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeiUnion___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeiUnion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypesUnion___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypesUnion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeBiUnion___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeBiUnion___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeBiUnion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeBiUnion_x27___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeBiUnion_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeBiUnion_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeiUnion___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_i_2_){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; 
v___x_3_ = lean_apply_1(v_inst_1_, v_i_2_);
v___x_4_ = lp_mathlib_Set_toFinset___redArg(v___x_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeiUnion___redArg(lean_object* v_inst_5_, lean_object* v_inst_6_, lean_object* v_inst_7_){
_start:
{
lean_object* v___f_8_; lean_object* v___x_9_; lean_object* v___x_10_; 
v___f_8_ = lean_alloc_closure((void*)(lp_mathlib_Set_fintypeiUnion___redArg___lam__0), 2, 1);
lean_closure_set(v___f_8_, 0, v_inst_7_);
v___x_9_ = lp_mathlib_Finset_biUnion___redArg(v_inst_5_, v_inst_6_, v___f_8_);
v___x_10_ = lp_mathlib_Fintype_subtype___redArg(v___x_9_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeiUnion(lean_object* v_00_u03b1_11_, lean_object* v_00_u03b9_12_, lean_object* v_inst_13_, lean_object* v_inst_14_, lean_object* v_f_15_, lean_object* v_inst_16_){
_start:
{
lean_object* v___x_17_; 
v___x_17_ = lp_mathlib_Set_fintypeiUnion___redArg(v_inst_13_, v_inst_14_, v_inst_16_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypesUnion___redArg(lean_object* v_inst_18_, lean_object* v_inst_19_, lean_object* v_H_20_){
_start:
{
lean_object* v___x_21_; lean_object* v___x_22_; 
v___x_21_ = lp_mathlib_PLift_fintype___redArg(v_inst_19_);
v___x_22_ = lp_mathlib_Set_fintypeiUnion___redArg(v_inst_18_, v___x_21_, v_H_20_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypesUnion(lean_object* v_00_u03b1_23_, lean_object* v_inst_24_, lean_object* v_s_25_, lean_object* v_inst_26_, lean_object* v_H_27_){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = lp_mathlib_Set_fintypesUnion___redArg(v_inst_24_, v_inst_26_, v_H_27_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeBiUnion___redArg___lam__0(lean_object* v_H_29_, lean_object* v_x_30_){
_start:
{
lean_object* v___x_31_; lean_object* v___x_32_; 
v___x_31_ = lean_apply_2(v_H_29_, v_x_30_, lean_box(0));
v___x_32_ = lp_mathlib_Set_toFinset___redArg(v___x_31_);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeBiUnion___redArg(lean_object* v_inst_33_, lean_object* v_inst_34_, lean_object* v_H_35_){
_start:
{
lean_object* v___f_36_; lean_object* v___x_37_; lean_object* v___x_38_; lean_object* v___x_39_; lean_object* v___x_40_; 
v___f_36_ = lean_alloc_closure((void*)(lp_mathlib_Set_fintypeBiUnion___redArg___lam__0), 2, 1);
lean_closure_set(v___f_36_, 0, v_H_35_);
v___x_37_ = lp_mathlib_Set_toFinset___redArg(v_inst_34_);
v___x_38_ = lp_mathlib_Multiset_attach___redArg(v___x_37_);
v___x_39_ = lp_mathlib_Finset_biUnion___redArg(v_inst_33_, v___x_38_, v___f_36_);
v___x_40_ = lp_mathlib_Fintype_subtype___redArg(v___x_39_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeBiUnion(lean_object* v_00_u03b1_41_, lean_object* v_inst_42_, lean_object* v_00_u03b9_43_, lean_object* v_s_44_, lean_object* v_inst_45_, lean_object* v_t_46_, lean_object* v_H_47_){
_start:
{
lean_object* v___x_48_; 
v___x_48_ = lp_mathlib_Set_fintypeBiUnion___redArg(v_inst_42_, v_inst_45_, v_H_47_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeBiUnion_x27___redArg___lam__0(lean_object* v_inst_49_, lean_object* v_x_50_){
_start:
{
lean_object* v___x_51_; lean_object* v___x_52_; 
v___x_51_ = lean_apply_1(v_inst_49_, v_x_50_);
v___x_52_ = lp_mathlib_Set_toFinset___redArg(v___x_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeBiUnion_x27___redArg(lean_object* v_inst_53_, lean_object* v_inst_54_, lean_object* v_inst_55_){
_start:
{
lean_object* v___f_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; 
v___f_56_ = lean_alloc_closure((void*)(lp_mathlib_Set_fintypeBiUnion_x27___redArg___lam__0), 2, 1);
lean_closure_set(v___f_56_, 0, v_inst_55_);
v___x_57_ = lp_mathlib_Set_toFinset___redArg(v_inst_54_);
v___x_58_ = lp_mathlib_Finset_biUnion___redArg(v_inst_53_, v___x_57_, v___f_56_);
v___x_59_ = lp_mathlib_Fintype_subtype___redArg(v___x_58_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeBiUnion_x27(lean_object* v_00_u03b1_60_, lean_object* v_inst_61_, lean_object* v_00_u03b9_62_, lean_object* v_s_63_, lean_object* v_inst_64_, lean_object* v_t_65_, lean_object* v_inst_66_){
_start:
{
lean_object* v___x_67_; 
v___x_67_ = lp_mathlib_Set_fintypeBiUnion_x27___redArg(v_inst_61_, v_inst_64_, v_inst_66_);
return v___x_67_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Powerset(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Range(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Lattice_Image(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Option(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Lattice(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Powerset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Range(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Lattice_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Option(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Set_Finite_Lattice(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Set_Finite_Powerset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Finite_Range(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Lattice_Image(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Option(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Set_Finite_Lattice(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Finite_Powerset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Finite_Range(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Lattice_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Option(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Set_Finite_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Set_Finite_Lattice(builtin);
}
#ifdef __cplusplus
}
#endif
