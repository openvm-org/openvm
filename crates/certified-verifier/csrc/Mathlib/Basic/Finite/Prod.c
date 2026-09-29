// Lean compiler output
// Module: Mathlib.Basic.Finite.Prod
// Imports: public import Init public meta import Init public import Mathlib.Data.Set.Finite.Basic public import Mathlib.Data.Fintype.Prod public import Mathlib.Data.Fintype.Pi public import Mathlib.Algebra.Order.Group.Multiset public import Mathlib.Data.ULift public import Mathlib.Data.Set.NAry
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
lean_object* lp_mathlib_Multiset_product___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Fintype_subtype___redArg(lean_object*);
lean_object* lp_mathlib_Set_fintypeImage___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_List_offDiag___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeProd___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeProd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeOffDiag___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeOffDiag(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeImage2___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeImage2___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeImage2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeProd___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_3_ = lp_mathlib_Set_toFinset___redArg(v_inst_1_);
v___x_4_ = lp_mathlib_Set_toFinset___redArg(v_inst_2_);
v___x_5_ = lp_mathlib_Multiset_product___redArg(v___x_3_, v___x_4_);
v___x_6_ = lp_mathlib_Fintype_subtype___redArg(v___x_5_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeProd(lean_object* v_00_u03b1_7_, lean_object* v_00_u03b2_8_, lean_object* v_s_9_, lean_object* v_t_10_, lean_object* v_inst_11_, lean_object* v_inst_12_){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = lp_mathlib_Set_fintypeProd___redArg(v_inst_11_, v_inst_12_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeOffDiag___redArg(lean_object* v_inst_14_){
_start:
{
lean_object* v___x_15_; lean_object* v___x_16_; lean_object* v___x_17_; 
v___x_15_ = lp_mathlib_Set_toFinset___redArg(v_inst_14_);
v___x_16_ = lp_mathlib_List_offDiag___redArg(v___x_15_);
v___x_17_ = lp_mathlib_Fintype_subtype___redArg(v___x_16_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeOffDiag(lean_object* v_00_u03b1_18_, lean_object* v_s_19_, lean_object* v_inst_20_){
_start:
{
lean_object* v___x_21_; 
v___x_21_ = lp_mathlib_Set_fintypeOffDiag___redArg(v_inst_20_);
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeImage2___redArg___lam__0(lean_object* v_f_22_, lean_object* v_x_23_){
_start:
{
lean_object* v_fst_24_; lean_object* v_snd_25_; lean_object* v___x_26_; 
v_fst_24_ = lean_ctor_get(v_x_23_, 0);
lean_inc(v_fst_24_);
v_snd_25_ = lean_ctor_get(v_x_23_, 1);
lean_inc(v_snd_25_);
lean_dec_ref(v_x_23_);
v___x_26_ = lean_apply_2(v_f_22_, v_fst_24_, v_snd_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeImage2___redArg(lean_object* v_inst_27_, lean_object* v_f_28_, lean_object* v_hs_29_, lean_object* v_ht_30_){
_start:
{
lean_object* v___f_31_; lean_object* v___x_32_; lean_object* v___x_33_; 
v___f_31_ = lean_alloc_closure((void*)(lp_mathlib_Set_fintypeImage2___redArg___lam__0), 2, 1);
lean_closure_set(v___f_31_, 0, v_f_28_);
v___x_32_ = lp_mathlib_Set_fintypeProd___redArg(v_hs_29_, v_ht_30_);
v___x_33_ = lp_mathlib_Set_fintypeImage___redArg(v_inst_27_, v___f_31_, v___x_32_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeImage2(lean_object* v_00_u03b1_34_, lean_object* v_00_u03b2_35_, lean_object* v_00_u03b3_36_, lean_object* v_inst_37_, lean_object* v_f_38_, lean_object* v_s_39_, lean_object* v_t_40_, lean_object* v_hs_41_, lean_object* v_ht_42_){
_start:
{
lean_object* v___x_43_; 
v___x_43_ = lp_mathlib_Set_fintypeImage2___redArg(v_inst_37_, v_f_38_, v_hs_41_, v_ht_42_);
return v___x_43_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Pi(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Multiset(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_ULift(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_NAry(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Basic_Finite_Prod(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Multiset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_NAry(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Basic_Finite_Prod(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Set_Finite_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Multiset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_ULift(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_NAry(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Basic_Finite_Prod(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Finite_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Multiset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_NAry(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Finite_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Basic_Finite_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Basic_Finite_Prod(builtin);
}
#ifdef __cplusplus
}
#endif
