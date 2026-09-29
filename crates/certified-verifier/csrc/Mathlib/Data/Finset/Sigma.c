// Lean compiler output
// Module: Mathlib.Data.Finset.Sigma
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.Lattice.Fold public import Mathlib.Data.Set.Sigma public import Mathlib.Order.CompleteLattice.Finset
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
lean_object* lp_mathlib_Function_Embedding_sigmaMk___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_map___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_sigma___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_sigma___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_sigma___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_sigma(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_sigmaLift___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_sigmaLift(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_sigma___redArg___lam__0(lean_object* v_t_1_, lean_object* v_i_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_apply_1(v_t_1_, v_i_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_sigma___redArg(lean_object* v_s_4_, lean_object* v_t_5_){
_start:
{
lean_object* v___f_6_; lean_object* v___x_7_; 
v___f_6_ = lean_alloc_closure((void*)(lp_mathlib_Finset_sigma___redArg___lam__0), 2, 1);
lean_closure_set(v___f_6_, 0, v_t_5_);
v___x_7_ = lp_mathlib_Multiset_sigma___redArg(v_s_4_, v___f_6_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_sigma(lean_object* v_00_u03b9_8_, lean_object* v_00_u03b1_9_, lean_object* v_s_10_, lean_object* v_t_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lp_mathlib_Finset_sigma___redArg(v_s_10_, v_t_11_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_sigmaLift___redArg(lean_object* v_inst_13_, lean_object* v_f_14_, lean_object* v_a_15_, lean_object* v_b_16_){
_start:
{
lean_object* v_fst_17_; lean_object* v_snd_18_; lean_object* v_fst_19_; lean_object* v_snd_20_; lean_object* v___x_21_; uint8_t v___x_22_; 
v_fst_17_ = lean_ctor_get(v_a_15_, 0);
lean_inc(v_fst_17_);
v_snd_18_ = lean_ctor_get(v_a_15_, 1);
lean_inc(v_snd_18_);
lean_dec_ref(v_a_15_);
v_fst_19_ = lean_ctor_get(v_b_16_, 0);
lean_inc_n(v_fst_19_, 2);
v_snd_20_ = lean_ctor_get(v_b_16_, 1);
lean_inc(v_snd_20_);
lean_dec_ref(v_b_16_);
v___x_21_ = lean_apply_2(v_inst_13_, v_fst_17_, v_fst_19_);
v___x_22_ = lean_unbox(v___x_21_);
if (v___x_22_ == 0)
{
lean_object* v___x_23_; 
lean_dec(v_snd_20_);
lean_dec(v_fst_19_);
lean_dec(v_snd_18_);
lean_dec(v_f_14_);
v___x_23_ = lean_box(0);
return v___x_23_;
}
else
{
lean_object* v___f_24_; lean_object* v___x_25_; lean_object* v___x_26_; 
lean_inc(v_fst_19_);
v___f_24_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_sigmaMk___redArg___lam__0), 2, 1);
lean_closure_set(v___f_24_, 0, v_fst_19_);
v___x_25_ = lean_apply_3(v_f_14_, v_fst_19_, v_snd_18_, v_snd_20_);
v___x_26_ = lp_mathlib_Finset_map___redArg(v___f_24_, v___x_25_);
return v___x_26_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_sigmaLift(lean_object* v_00_u03b9_27_, lean_object* v_00_u03b1_28_, lean_object* v_00_u03b2_29_, lean_object* v_00_u03b3_30_, lean_object* v_inst_31_, lean_object* v_f_32_, lean_object* v_a_33_, lean_object* v_b_34_){
_start:
{
lean_object* v___x_35_; 
v___x_35_ = lp_mathlib_Finset_sigmaLift___redArg(v_inst_31_, v_f_32_, v_a_33_, v_b_34_);
return v___x_35_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Lattice_Fold(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Sigma(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_CompleteLattice_Finset(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Sigma(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Lattice_Fold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Sigma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_CompleteLattice_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Finset_Sigma(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Finset_Lattice_Fold(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Sigma(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_CompleteLattice_Finset(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_Sigma(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Lattice_Fold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Sigma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_CompleteLattice_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Sigma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Finset_Sigma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Finset_Sigma(builtin);
}
#ifdef __cplusplus
}
#endif
