// Lean compiler output
// Module: Mathlib.Logic.Equiv.Nat
// Imports: public import Init public meta import Init public import Mathlib.Data.Nat.Bits public import Mathlib.Data.Nat.Pairing
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
uint8_t l_Nat_testBit(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_bit(uint8_t, lean_object*);
lean_object* lp_mathlib_Equiv_boolProdEquivSum(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
extern lean_object* lp_mathlib_Equiv_intEquivNatSumNat;
lean_object* lp_mathlib_Equiv_prodCongr___redArg(lean_object*, lean_object*);
extern lean_object* lp_mathlib_Nat_pairEquiv;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolProdNatEquivNat___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolProdNatEquivNat___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolProdNatEquivNat___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolProdNatEquivNat___lam__1___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_boolProdNatEquivNat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_boolProdNatEquivNat___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_boolProdNatEquivNat___closed__0 = (const lean_object*)&lp_mathlib_Equiv_boolProdNatEquivNat___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_boolProdNatEquivNat___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_boolProdNatEquivNat___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_boolProdNatEquivNat___closed__1 = (const lean_object*)&lp_mathlib_Equiv_boolProdNatEquivNat___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_boolProdNatEquivNat___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_boolProdNatEquivNat___closed__1_value),((lean_object*)&lp_mathlib_Equiv_boolProdNatEquivNat___closed__0_value)}};
static const lean_object* lp_mathlib_Equiv_boolProdNatEquivNat___closed__2 = (const lean_object*)&lp_mathlib_Equiv_boolProdNatEquivNat___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_Equiv_boolProdNatEquivNat = (const lean_object*)&lp_mathlib_Equiv_boolProdNatEquivNat___closed__2_value;
static lean_once_cell_t lp_mathlib_Equiv_natSumNatEquivNat___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_natSumNatEquivNat___closed__0;
static lean_once_cell_t lp_mathlib_Equiv_natSumNatEquivNat___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_natSumNatEquivNat___closed__1;
static lean_once_cell_t lp_mathlib_Equiv_natSumNatEquivNat___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_natSumNatEquivNat___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_natSumNatEquivNat;
static lean_once_cell_t lp_mathlib_Equiv_intEquivNat___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_intEquivNat___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_intEquivNat;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodEquivOfEquivNat___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodEquivOfEquivNat(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolProdNatEquivNat___lam__0(lean_object* v_n_1_){
_start:
{
lean_object* v___x_2_; uint8_t v___x_3_; lean_object* v___x_4_; lean_object* v___x_5_; lean_object* v___x_6_; lean_object* v___x_7_; 
v___x_2_ = lean_unsigned_to_nat(0u);
v___x_3_ = l_Nat_testBit(v_n_1_, v___x_2_);
v___x_4_ = lean_unsigned_to_nat(1u);
v___x_5_ = lean_nat_shiftr(v_n_1_, v___x_4_);
v___x_6_ = lean_box(v___x_3_);
v___x_7_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_7_, 0, v___x_6_);
lean_ctor_set(v___x_7_, 1, v___x_5_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolProdNatEquivNat___lam__0___boxed(lean_object* v_n_8_){
_start:
{
lean_object* v_res_9_; 
v_res_9_ = lp_mathlib_Equiv_boolProdNatEquivNat___lam__0(v_n_8_);
lean_dec(v_n_8_);
return v_res_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolProdNatEquivNat___lam__1(lean_object* v___y_10_){
_start:
{
lean_object* v_fst_11_; lean_object* v_snd_12_; uint8_t v___x_13_; lean_object* v___x_14_; 
v_fst_11_ = lean_ctor_get(v___y_10_, 0);
v_snd_12_ = lean_ctor_get(v___y_10_, 1);
v___x_13_ = lean_unbox(v_fst_11_);
v___x_14_ = lp_mathlib_Nat_bit(v___x_13_, v_snd_12_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolProdNatEquivNat___lam__1___boxed(lean_object* v___y_15_){
_start:
{
lean_object* v_res_16_; 
v_res_16_ = lp_mathlib_Equiv_boolProdNatEquivNat___lam__1(v___y_15_);
lean_dec_ref(v___y_15_);
return v_res_16_;
}
}
static lean_object* _init_lp_mathlib_Equiv_natSumNatEquivNat___closed__0(void){
_start:
{
lean_object* v___x_23_; 
v___x_23_ = lp_mathlib_Equiv_boolProdEquivSum(lean_box(0));
return v___x_23_;
}
}
static lean_object* _init_lp_mathlib_Equiv_natSumNatEquivNat___closed__1(void){
_start:
{
lean_object* v___x_24_; lean_object* v___x_25_; 
v___x_24_ = lean_obj_once(&lp_mathlib_Equiv_natSumNatEquivNat___closed__0, &lp_mathlib_Equiv_natSumNatEquivNat___closed__0_once, _init_lp_mathlib_Equiv_natSumNatEquivNat___closed__0);
v___x_25_ = lp_mathlib_Equiv_symm___redArg(v___x_24_);
return v___x_25_;
}
}
static lean_object* _init_lp_mathlib_Equiv_natSumNatEquivNat___closed__2(void){
_start:
{
lean_object* v___x_26_; lean_object* v___x_27_; lean_object* v___x_28_; 
v___x_26_ = ((lean_object*)(lp_mathlib_Equiv_boolProdNatEquivNat));
v___x_27_ = lean_obj_once(&lp_mathlib_Equiv_natSumNatEquivNat___closed__1, &lp_mathlib_Equiv_natSumNatEquivNat___closed__1_once, _init_lp_mathlib_Equiv_natSumNatEquivNat___closed__1);
v___x_28_ = lp_mathlib_Equiv_trans___redArg(v___x_27_, v___x_26_);
return v___x_28_;
}
}
static lean_object* _init_lp_mathlib_Equiv_natSumNatEquivNat(void){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lean_obj_once(&lp_mathlib_Equiv_natSumNatEquivNat___closed__2, &lp_mathlib_Equiv_natSumNatEquivNat___closed__2_once, _init_lp_mathlib_Equiv_natSumNatEquivNat___closed__2);
return v___x_29_;
}
}
static lean_object* _init_lp_mathlib_Equiv_intEquivNat___closed__0(void){
_start:
{
lean_object* v___x_30_; lean_object* v___x_31_; lean_object* v___x_32_; 
v___x_30_ = lp_mathlib_Equiv_natSumNatEquivNat;
v___x_31_ = lp_mathlib_Equiv_intEquivNatSumNat;
v___x_32_ = lp_mathlib_Equiv_trans___redArg(v___x_31_, v___x_30_);
return v___x_32_;
}
}
static lean_object* _init_lp_mathlib_Equiv_intEquivNat(void){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = lean_obj_once(&lp_mathlib_Equiv_intEquivNat___closed__0, &lp_mathlib_Equiv_intEquivNat___closed__0_once, _init_lp_mathlib_Equiv_intEquivNat___closed__0);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodEquivOfEquivNat___redArg(lean_object* v_e_34_){
_start:
{
lean_object* v___x_35_; lean_object* v___x_36_; lean_object* v___x_37_; lean_object* v___x_38_; lean_object* v___x_39_; 
lean_inc_ref_n(v_e_34_, 2);
v___x_35_ = lp_mathlib_Equiv_prodCongr___redArg(v_e_34_, v_e_34_);
v___x_36_ = lp_mathlib_Nat_pairEquiv;
v___x_37_ = lp_mathlib_Equiv_trans___redArg(v___x_35_, v___x_36_);
v___x_38_ = lp_mathlib_Equiv_symm___redArg(v_e_34_);
v___x_39_ = lp_mathlib_Equiv_trans___redArg(v___x_37_, v___x_38_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodEquivOfEquivNat(lean_object* v_00_u03b1_40_, lean_object* v_e_41_){
_start:
{
lean_object* v___x_42_; 
v___x_42_ = lp_mathlib_Equiv_prodEquivOfEquivNat___redArg(v_e_41_);
return v___x_42_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Bits(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Pairing(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Nat(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Bits(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Pairing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Equiv_natSumNatEquivNat = _init_lp_mathlib_Equiv_natSumNatEquivNat();
lean_mark_persistent(lp_mathlib_Equiv_natSumNatEquivNat);
lp_mathlib_Equiv_intEquivNat = _init_lp_mathlib_Equiv_intEquivNat();
lean_mark_persistent(lp_mathlib_Equiv_intEquivNat);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Logic_Equiv_Nat(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Nat_Bits(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Pairing(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Nat(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Bits(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Pairing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Logic_Equiv_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Logic_Equiv_Nat(builtin);
}
#ifdef __cplusplus
}
#endif
