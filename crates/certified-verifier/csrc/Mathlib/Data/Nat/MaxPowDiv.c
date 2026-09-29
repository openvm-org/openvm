// Lean compiler output
// Module: Mathlib.Data.Nat.MaxPowDiv
// Imports: public import Init public meta import Init import Mathlib.Data.Nat.Notation
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
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_maxPowDvdDiv_go___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_maxPowDvdDiv_go___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_maxPowDvdDiv_go(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_maxPowDvdDiv_go___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_MaxPowDiv_0__Nat_maxPowDvdDiv_match__1_splitter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_MaxPowDiv_0__Nat_maxPowDvdDiv_match__1_splitter(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_maxPowDvdDiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_maxPowDvdDiv___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_divMaxPow(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_divMaxPow___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_maxPowDvdDiv_go___redArg(lean_object* v_n_1_, lean_object* v_p_2_){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; uint8_t v___x_5_; 
v___x_3_ = lean_nat_mod(v_n_1_, v_p_2_);
v___x_4_ = lean_unsigned_to_nat(0u);
v___x_5_ = lean_nat_dec_eq(v___x_3_, v___x_4_);
lean_dec(v___x_3_);
if (v___x_5_ == 0)
{
lean_object* v___x_6_; 
v___x_6_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_6_, 0, v___x_4_);
lean_ctor_set(v___x_6_, 1, v_n_1_);
return v___x_6_;
}
else
{
lean_object* v___x_7_; lean_object* v___x_8_; lean_object* v_fst_9_; lean_object* v_snd_10_; lean_object* v___x_12_; uint8_t v_isShared_13_; uint8_t v_isSharedCheck_29_; 
v___x_7_ = lean_nat_mul(v_p_2_, v_p_2_);
v___x_8_ = lp_mathlib_Nat_maxPowDvdDiv_go___redArg(v_n_1_, v___x_7_);
lean_dec(v___x_7_);
v_fst_9_ = lean_ctor_get(v___x_8_, 0);
v_snd_10_ = lean_ctor_get(v___x_8_, 1);
v_isSharedCheck_29_ = !lean_is_exclusive(v___x_8_);
if (v_isSharedCheck_29_ == 0)
{
v___x_12_ = v___x_8_;
v_isShared_13_ = v_isSharedCheck_29_;
goto v_resetjp_11_;
}
else
{
lean_inc(v_snd_10_);
lean_inc(v_fst_9_);
lean_dec(v___x_8_);
v___x_12_ = lean_box(0);
v_isShared_13_ = v_isSharedCheck_29_;
goto v_resetjp_11_;
}
v_resetjp_11_:
{
lean_object* v___x_14_; uint8_t v___x_15_; 
v___x_14_ = lean_nat_mod(v_snd_10_, v_p_2_);
v___x_15_ = lean_nat_dec_eq(v___x_14_, v___x_4_);
lean_dec(v___x_14_);
if (v___x_15_ == 0)
{
lean_object* v___x_16_; lean_object* v___x_17_; lean_object* v___x_19_; 
v___x_16_ = lean_unsigned_to_nat(2u);
v___x_17_ = lean_nat_mul(v___x_16_, v_fst_9_);
lean_dec(v_fst_9_);
if (v_isShared_13_ == 0)
{
lean_ctor_set(v___x_12_, 0, v___x_17_);
v___x_19_ = v___x_12_;
goto v_reusejp_18_;
}
else
{
lean_object* v_reuseFailAlloc_20_; 
v_reuseFailAlloc_20_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_20_, 0, v___x_17_);
lean_ctor_set(v_reuseFailAlloc_20_, 1, v_snd_10_);
v___x_19_ = v_reuseFailAlloc_20_;
goto v_reusejp_18_;
}
v_reusejp_18_:
{
return v___x_19_;
}
}
else
{
lean_object* v___x_21_; lean_object* v___x_22_; lean_object* v___x_23_; lean_object* v___x_24_; lean_object* v___x_25_; lean_object* v___x_27_; 
v___x_21_ = lean_unsigned_to_nat(2u);
v___x_22_ = lean_nat_mul(v___x_21_, v_fst_9_);
lean_dec(v_fst_9_);
v___x_23_ = lean_unsigned_to_nat(1u);
v___x_24_ = lean_nat_add(v___x_22_, v___x_23_);
lean_dec(v___x_22_);
v___x_25_ = lean_nat_div(v_snd_10_, v_p_2_);
lean_dec(v_snd_10_);
if (v_isShared_13_ == 0)
{
lean_ctor_set(v___x_12_, 1, v___x_25_);
lean_ctor_set(v___x_12_, 0, v___x_24_);
v___x_27_ = v___x_12_;
goto v_reusejp_26_;
}
else
{
lean_object* v_reuseFailAlloc_28_; 
v_reuseFailAlloc_28_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_28_, 0, v___x_24_);
lean_ctor_set(v_reuseFailAlloc_28_, 1, v___x_25_);
v___x_27_ = v_reuseFailAlloc_28_;
goto v_reusejp_26_;
}
v_reusejp_26_:
{
return v___x_27_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_maxPowDvdDiv_go___redArg___boxed(lean_object* v_n_30_, lean_object* v_p_31_){
_start:
{
lean_object* v_res_32_; 
v_res_32_ = lp_mathlib_Nat_maxPowDvdDiv_go___redArg(v_n_30_, v_p_31_);
lean_dec(v_p_31_);
return v_res_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_maxPowDvdDiv_go(lean_object* v_n_33_, lean_object* v_p_34_, lean_object* v_hp_35_){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lp_mathlib_Nat_maxPowDvdDiv_go___redArg(v_n_33_, v_p_34_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_maxPowDvdDiv_go___boxed(lean_object* v_n_37_, lean_object* v_p_38_, lean_object* v_hp_39_){
_start:
{
lean_object* v_res_40_; 
v_res_40_ = lp_mathlib_Nat_maxPowDvdDiv_go(v_n_37_, v_p_38_, v_hp_39_);
lean_dec(v_p_38_);
return v_res_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_MaxPowDiv_0__Nat_maxPowDvdDiv_match__1_splitter___redArg(lean_object* v_x_41_, lean_object* v_h__1_42_){
_start:
{
lean_object* v_fst_43_; lean_object* v_snd_44_; lean_object* v___x_45_; 
v_fst_43_ = lean_ctor_get(v_x_41_, 0);
lean_inc(v_fst_43_);
v_snd_44_ = lean_ctor_get(v_x_41_, 1);
lean_inc(v_snd_44_);
lean_dec_ref(v_x_41_);
v___x_45_ = lean_apply_2(v_h__1_42_, v_fst_43_, v_snd_44_);
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_MaxPowDiv_0__Nat_maxPowDvdDiv_match__1_splitter(lean_object* v_motive_46_, lean_object* v_x_47_, lean_object* v_h__1_48_){
_start:
{
lean_object* v_fst_49_; lean_object* v_snd_50_; lean_object* v___x_51_; 
v_fst_49_ = lean_ctor_get(v_x_47_, 0);
lean_inc(v_fst_49_);
v_snd_50_ = lean_ctor_get(v_x_47_, 1);
lean_inc(v_snd_50_);
lean_dec_ref(v_x_47_);
v___x_51_ = lean_apply_2(v_h__1_48_, v_fst_49_, v_snd_50_);
return v___x_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_maxPowDvdDiv(lean_object* v_p_52_, lean_object* v_n_53_){
_start:
{
lean_object* v___x_57_; uint8_t v___x_58_; 
v___x_57_ = lean_unsigned_to_nat(1u);
v___x_58_ = lean_nat_dec_lt(v___x_57_, v_p_52_);
if (v___x_58_ == 0)
{
goto v___jp_54_;
}
else
{
lean_object* v___x_59_; uint8_t v___x_60_; 
v___x_59_ = lean_unsigned_to_nat(0u);
v___x_60_ = lean_nat_dec_eq(v_n_53_, v___x_59_);
if (v___x_60_ == 0)
{
if (v___x_58_ == 0)
{
goto v___jp_54_;
}
else
{
lean_object* v___x_61_; 
v___x_61_ = lp_mathlib_Nat_maxPowDvdDiv_go___redArg(v_n_53_, v_p_52_);
return v___x_61_;
}
}
else
{
goto v___jp_54_;
}
}
v___jp_54_:
{
lean_object* v___x_55_; lean_object* v___x_56_; 
v___x_55_ = lean_unsigned_to_nat(0u);
v___x_56_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_56_, 0, v___x_55_);
lean_ctor_set(v___x_56_, 1, v_n_53_);
return v___x_56_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_maxPowDvdDiv___boxed(lean_object* v_p_62_, lean_object* v_n_63_){
_start:
{
lean_object* v_res_64_; 
v_res_64_ = lp_mathlib_Nat_maxPowDvdDiv(v_p_62_, v_n_63_);
lean_dec(v_p_62_);
return v_res_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_divMaxPow(lean_object* v_n_65_, lean_object* v_p_66_){
_start:
{
lean_object* v___x_67_; lean_object* v_snd_68_; 
v___x_67_ = lp_mathlib_Nat_maxPowDvdDiv(v_p_66_, v_n_65_);
v_snd_68_ = lean_ctor_get(v___x_67_, 1);
lean_inc(v_snd_68_);
lean_dec_ref(v___x_67_);
return v_snd_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_divMaxPow___boxed(lean_object* v_n_69_, lean_object* v_p_70_){
_start:
{
lean_object* v_res_71_; 
v_res_71_ = lp_mathlib_Nat_divMaxPow(v_n_69_, v_p_70_);
lean_dec(v_p_70_);
return v_res_71_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Notation(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_MaxPowDiv(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Nat_MaxPowDiv(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Nat_Notation(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Nat_MaxPowDiv(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_MaxPowDiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Nat_MaxPowDiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Nat_MaxPowDiv(builtin);
}
#ifdef __cplusplus
}
#endif
