// Lean compiler output
// Module: Mathlib.Order.OrderIsoNat
// Imports: public import Init public meta import Init public import Mathlib.Basic.Denumerable public import Mathlib.Data.Set.Subsingleton public import Mathlib.Logic.Function.Iterate public import Mathlib.Order.Hom.Basic public import Mathlib.Order.Lattice.Nat
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
lean_object* lp_mathlib_Nat_Subtype_ofNat___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Embedding_subtype___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_Function_Embedding_trans___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_natLT___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_natLT___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_natLT(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_natLT___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_natGT___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_natGT___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_natGT(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_natGT___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Nat_orderEmbeddingOfSet___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Function_Embedding_subtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Nat_orderEmbeddingOfSet___redArg___closed__0 = (const lean_object*)&lp_mathlib_Nat_orderEmbeddingOfSet___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Nat_orderEmbeddingOfSet___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_orderEmbeddingOfSet(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_natLT___redArg(lean_object* v_f_1_){
_start:
{
lean_inc(v_f_1_);
return v_f_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_natLT___redArg___boxed(lean_object* v_f_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_RelEmbedding_natLT___redArg(v_f_2_);
lean_dec(v_f_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_natLT(lean_object* v_00_u03b1_4_, lean_object* v_r_5_, lean_object* v_inst_6_, lean_object* v_f_7_, lean_object* v_H_8_){
_start:
{
lean_inc(v_f_7_);
return v_f_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_natLT___boxed(lean_object* v_00_u03b1_9_, lean_object* v_r_10_, lean_object* v_inst_11_, lean_object* v_f_12_, lean_object* v_H_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_RelEmbedding_natLT(v_00_u03b1_9_, v_r_10_, v_inst_11_, v_f_12_, v_H_13_);
lean_dec(v_f_12_);
return v_res_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_natGT___redArg(lean_object* v_f_15_){
_start:
{
lean_inc(v_f_15_);
return v_f_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_natGT___redArg___boxed(lean_object* v_f_16_){
_start:
{
lean_object* v_res_17_; 
v_res_17_ = lp_mathlib_RelEmbedding_natGT___redArg(v_f_16_);
lean_dec(v_f_16_);
return v_res_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_natGT(lean_object* v_00_u03b1_18_, lean_object* v_r_19_, lean_object* v_inst_20_, lean_object* v_f_21_, lean_object* v_H_22_){
_start:
{
lean_inc(v_f_21_);
return v_f_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_natGT___boxed(lean_object* v_00_u03b1_23_, lean_object* v_r_24_, lean_object* v_inst_25_, lean_object* v_f_26_, lean_object* v_H_27_){
_start:
{
lean_object* v_res_28_; 
v_res_28_ = lp_mathlib_RelEmbedding_natGT(v_00_u03b1_23_, v_r_24_, v_inst_25_, v_f_26_, v_H_27_);
lean_dec(v_f_26_);
return v_res_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_orderEmbeddingOfSet___redArg(lean_object* v_inst_30_){
_start:
{
lean_object* v___x_31_; lean_object* v___f_32_; lean_object* v___f_33_; 
v___x_31_ = lean_alloc_closure((void*)(lp_mathlib_Nat_Subtype_ofNat___boxed), 4, 3);
lean_closure_set(v___x_31_, 0, lean_box(0));
lean_closure_set(v___x_31_, 1, v_inst_30_);
lean_closure_set(v___x_31_, 2, lean_box(0));
v___f_32_ = ((lean_object*)(lp_mathlib_Nat_orderEmbeddingOfSet___redArg___closed__0));
v___f_33_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_trans___redArg___lam__0), 3, 2);
lean_closure_set(v___f_33_, 0, v___x_31_);
lean_closure_set(v___f_33_, 1, v___f_32_);
return v___f_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_orderEmbeddingOfSet(lean_object* v_s_34_, lean_object* v_inst_35_, lean_object* v_inst_36_){
_start:
{
lean_object* v___x_37_; 
v___x_37_ = lp_mathlib_Nat_orderEmbeddingOfSet___redArg(v_inst_36_);
return v___x_37_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_Denumerable(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Subsingleton(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Function_Iterate(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Hom_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Lattice_Nat(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_OrderIsoNat(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Denumerable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Subsingleton(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Function_Iterate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Lattice_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_OrderIsoNat(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Basic_Denumerable(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Subsingleton(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Function_Iterate(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Hom_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Lattice_Nat(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_OrderIsoNat(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_Denumerable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Subsingleton(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Function_Iterate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Lattice_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_OrderIsoNat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_OrderIsoNat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_OrderIsoNat(builtin);
}
#ifdef __cplusplus
}
#endif
