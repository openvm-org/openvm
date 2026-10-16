// Lean compiler output
// Module: Mathlib.Data.Finset.Prod
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.Card public import Mathlib.Data.Finset.Union public import Mathlib.Data.List.OffDiag public import Mathlib.Data.Nat.Choose.Basic
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
lean_object* lp_mathlib_Multiset_product___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_List_offDiag___redArg(lean_object*);
lean_object* lp_mathlib_Finset_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_product___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_product(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Finset_instSProd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Finset_product, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Finset_instSProd___closed__0 = (const lean_object*)&lp_mathlib_Finset_instSProd___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Finset_instSProd(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Finset_prod___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_Finset_prod___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_Finset_prod___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_Finset_prod___closed__0 = (const lean_object*)&lp_mathlib_Equiv_Finset_prod___closed__0_value;
static const lean_ctor_object lp_mathlib_Equiv_Finset_prod___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_Finset_prod___closed__0_value),((lean_object*)&lp_mathlib_Equiv_Finset_prod___closed__0_value)}};
static const lean_object* lp_mathlib_Equiv_Finset_prod___closed__1 = (const lean_object*)&lp_mathlib_Equiv_Finset_prod___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Finset_prod(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Finset_prod___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_diag___redArg___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Finset_diag___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Finset_diag___redArg___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Finset_diag___redArg___closed__0 = (const lean_object*)&lp_mathlib_Finset_diag___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Finset_diag___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_diag(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_offDiag___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_offDiag(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_product___redArg(lean_object* v_s_1_, lean_object* v_t_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lp_mathlib_Multiset_product___redArg(v_s_1_, v_t_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_product(lean_object* v_00_u03b1_4_, lean_object* v_00_u03b2_5_, lean_object* v_s_6_, lean_object* v_t_7_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lp_mathlib_Multiset_product___redArg(v_s_6_, v_t_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instSProd(lean_object* v_00_u03b1_10_, lean_object* v_00_u03b2_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = ((lean_object*)(lp_mathlib_Finset_instSProd___closed__0));
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Finset_prod___lam__0(lean_object* v_x_13_){
_start:
{
lean_object* v_fst_14_; lean_object* v_snd_15_; lean_object* v___x_17_; uint8_t v_isShared_18_; uint8_t v_isSharedCheck_22_; 
v_fst_14_ = lean_ctor_get(v_x_13_, 0);
v_snd_15_ = lean_ctor_get(v_x_13_, 1);
v_isSharedCheck_22_ = !lean_is_exclusive(v_x_13_);
if (v_isSharedCheck_22_ == 0)
{
v___x_17_ = v_x_13_;
v_isShared_18_ = v_isSharedCheck_22_;
goto v_resetjp_16_;
}
else
{
lean_inc(v_snd_15_);
lean_inc(v_fst_14_);
lean_dec(v_x_13_);
v___x_17_ = lean_box(0);
v_isShared_18_ = v_isSharedCheck_22_;
goto v_resetjp_16_;
}
v_resetjp_16_:
{
lean_object* v___x_20_; 
if (v_isShared_18_ == 0)
{
v___x_20_ = v___x_17_;
goto v_reusejp_19_;
}
else
{
lean_object* v_reuseFailAlloc_21_; 
v_reuseFailAlloc_21_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_21_, 0, v_fst_14_);
lean_ctor_set(v_reuseFailAlloc_21_, 1, v_snd_15_);
v___x_20_ = v_reuseFailAlloc_21_;
goto v_reusejp_19_;
}
v_reusejp_19_:
{
return v___x_20_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Finset_prod(lean_object* v_00_u03b1_26_, lean_object* v_00_u03b2_27_, lean_object* v_s_28_, lean_object* v_t_29_){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = ((lean_object*)(lp_mathlib_Equiv_Finset_prod___closed__1));
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Finset_prod___boxed(lean_object* v_00_u03b1_31_, lean_object* v_00_u03b2_32_, lean_object* v_s_33_, lean_object* v_t_34_){
_start:
{
lean_object* v_res_35_; 
v_res_35_ = lp_mathlib_Equiv_Finset_prod(v_00_u03b1_31_, v_00_u03b2_32_, v_s_33_, v_t_34_);
lean_dec(v_t_34_);
lean_dec(v_s_33_);
return v_res_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_diag___redArg___lam__0(lean_object* v___y_36_){
_start:
{
lean_object* v___x_37_; 
lean_inc(v___y_36_);
v___x_37_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_37_, 0, v___y_36_);
lean_ctor_set(v___x_37_, 1, v___y_36_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_diag___redArg(lean_object* v_s_39_){
_start:
{
lean_object* v___f_40_; lean_object* v___x_41_; 
v___f_40_ = ((lean_object*)(lp_mathlib_Finset_diag___redArg___closed__0));
v___x_41_ = lp_mathlib_Finset_map___redArg(v___f_40_, v_s_39_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_diag(lean_object* v_00_u03b1_42_, lean_object* v_s_43_){
_start:
{
lean_object* v___x_44_; 
v___x_44_ = lp_mathlib_Finset_diag___redArg(v_s_43_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_offDiag___redArg(lean_object* v_s_45_){
_start:
{
lean_object* v___x_46_; 
v___x_46_ = lp_mathlib_List_offDiag___redArg(v_s_45_);
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_offDiag(lean_object* v_00_u03b1_47_, lean_object* v_s_48_){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lp_mathlib_List_offDiag___redArg(v_s_48_);
return v___x_49_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Card(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Union(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_OffDiag(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Choose_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Prod(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Union(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_OffDiag(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Choose_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Finset_Prod(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Finset_Card(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Union(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_OffDiag(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Choose_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_Prod(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Union(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_OffDiag(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Choose_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Finset_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Finset_Prod(builtin);
}
#ifdef __cplusplus
}
#endif
