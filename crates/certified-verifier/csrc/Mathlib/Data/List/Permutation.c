// Lean compiler output
// Module: Mathlib.Data.List.Permutation
// Imports: public import Init public meta import Init public import Mathlib.Data.Nat.Factorial.Basic public import Mathlib.Data.List.Count public import Mathlib.Data.List.Duplicate public import Mathlib.Data.List.InsertIdx public import Mathlib.Data.List.Induction public import Batteries.Data.List.Perm public import Mathlib.Data.List.Perm.Basic public import Mathlib.Tactic.Finiteness.Attr public import Mathlib.Data.Int.Order.Basic public import Mathlib.Order.Basic
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
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Permutation_0__List_permutationsAux2_match__3_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Permutation_0__List_permutationsAux2_match__3_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Permutation_0__List_permutationsAux2_match__1_splitter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Permutation_0__List_permutationsAux2_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Permutation_0__List_permutationsAux_rec_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Permutation_0__List_permutationsAux_rec_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Permutation_0__Nat_factorial_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Permutation_0__Nat_factorial_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Permutation_0__Nat_factorial_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Permutation_0__Nat_factorial_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Permutation_0__List_permutationsAux2_match__3_splitter___redArg(lean_object* v_x_1_, lean_object* v_x_2_, lean_object* v_h__1_3_, lean_object* v_h__2_4_){
_start:
{
if (lean_obj_tag(v_x_1_) == 0)
{
lean_object* v___x_5_; 
lean_dec(v_h__2_4_);
v___x_5_ = lean_apply_1(v_h__1_3_, v_x_2_);
return v___x_5_;
}
else
{
lean_object* v_head_6_; lean_object* v_tail_7_; lean_object* v___x_8_; 
lean_dec(v_h__1_3_);
v_head_6_ = lean_ctor_get(v_x_1_, 0);
lean_inc(v_head_6_);
v_tail_7_ = lean_ctor_get(v_x_1_, 1);
lean_inc(v_tail_7_);
lean_dec_ref_known(v_x_1_, 2);
v___x_8_ = lean_apply_3(v_h__2_4_, v_head_6_, v_tail_7_, v_x_2_);
return v___x_8_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Permutation_0__List_permutationsAux2_match__3_splitter(lean_object* v_00_u03b1_9_, lean_object* v_00_u03b2_10_, lean_object* v_motive_11_, lean_object* v_x_12_, lean_object* v_x_13_, lean_object* v_h__1_14_, lean_object* v_h__2_15_){
_start:
{
if (lean_obj_tag(v_x_12_) == 0)
{
lean_object* v___x_16_; 
lean_dec(v_h__2_15_);
v___x_16_ = lean_apply_1(v_h__1_14_, v_x_13_);
return v___x_16_;
}
else
{
lean_object* v_head_17_; lean_object* v_tail_18_; lean_object* v___x_19_; 
lean_dec(v_h__1_14_);
v_head_17_ = lean_ctor_get(v_x_12_, 0);
lean_inc(v_head_17_);
v_tail_18_ = lean_ctor_get(v_x_12_, 1);
lean_inc(v_tail_18_);
lean_dec_ref_known(v_x_12_, 2);
v___x_19_ = lean_apply_3(v_h__2_15_, v_head_17_, v_tail_18_, v_x_13_);
return v___x_19_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Permutation_0__List_permutationsAux2_match__1_splitter___redArg(lean_object* v_x_20_, lean_object* v_h__1_21_){
_start:
{
lean_object* v_fst_22_; lean_object* v_snd_23_; lean_object* v___x_24_; 
v_fst_22_ = lean_ctor_get(v_x_20_, 0);
lean_inc(v_fst_22_);
v_snd_23_ = lean_ctor_get(v_x_20_, 1);
lean_inc(v_snd_23_);
lean_dec_ref(v_x_20_);
v___x_24_ = lean_apply_2(v_h__1_21_, v_fst_22_, v_snd_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Permutation_0__List_permutationsAux2_match__1_splitter(lean_object* v_00_u03b1_25_, lean_object* v_00_u03b2_26_, lean_object* v_motive_27_, lean_object* v_x_28_, lean_object* v_h__1_29_){
_start:
{
lean_object* v_fst_30_; lean_object* v_snd_31_; lean_object* v___x_32_; 
v_fst_30_ = lean_ctor_get(v_x_28_, 0);
lean_inc(v_fst_30_);
v_snd_31_ = lean_ctor_get(v_x_28_, 1);
lean_inc(v_snd_31_);
lean_dec_ref(v_x_28_);
v___x_32_ = lean_apply_2(v_h__1_29_, v_fst_30_, v_snd_31_);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Permutation_0__List_permutationsAux_rec_match__1_splitter___redArg(lean_object* v_x_33_, lean_object* v_x_34_, lean_object* v_h__1_35_, lean_object* v_h__2_36_){
_start:
{
if (lean_obj_tag(v_x_33_) == 0)
{
lean_object* v___x_37_; 
lean_dec(v_h__2_36_);
v___x_37_ = lean_apply_1(v_h__1_35_, v_x_34_);
return v___x_37_;
}
else
{
lean_object* v_head_38_; lean_object* v_tail_39_; lean_object* v___x_40_; 
lean_dec(v_h__1_35_);
v_head_38_ = lean_ctor_get(v_x_33_, 0);
lean_inc(v_head_38_);
v_tail_39_ = lean_ctor_get(v_x_33_, 1);
lean_inc(v_tail_39_);
lean_dec_ref_known(v_x_33_, 2);
v___x_40_ = lean_apply_3(v_h__2_36_, v_head_38_, v_tail_39_, v_x_34_);
return v___x_40_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Permutation_0__List_permutationsAux_rec_match__1_splitter(lean_object* v_00_u03b1_41_, lean_object* v_motive_42_, lean_object* v_x_43_, lean_object* v_x_44_, lean_object* v_h__1_45_, lean_object* v_h__2_46_){
_start:
{
if (lean_obj_tag(v_x_43_) == 0)
{
lean_object* v___x_47_; 
lean_dec(v_h__2_46_);
v___x_47_ = lean_apply_1(v_h__1_45_, v_x_44_);
return v___x_47_;
}
else
{
lean_object* v_head_48_; lean_object* v_tail_49_; lean_object* v___x_50_; 
lean_dec(v_h__1_45_);
v_head_48_ = lean_ctor_get(v_x_43_, 0);
lean_inc(v_head_48_);
v_tail_49_ = lean_ctor_get(v_x_43_, 1);
lean_inc(v_tail_49_);
lean_dec_ref_known(v_x_43_, 2);
v___x_50_ = lean_apply_3(v_h__2_46_, v_head_48_, v_tail_49_, v_x_44_);
return v___x_50_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Permutation_0__Nat_factorial_match__1_splitter___redArg(lean_object* v_x_51_, lean_object* v_h__1_52_, lean_object* v_h__2_53_){
_start:
{
lean_object* v_zero_54_; uint8_t v_isZero_55_; 
v_zero_54_ = lean_unsigned_to_nat(0u);
v_isZero_55_ = lean_nat_dec_eq(v_x_51_, v_zero_54_);
if (v_isZero_55_ == 1)
{
lean_object* v___x_56_; lean_object* v___x_57_; 
lean_dec(v_h__2_53_);
v___x_56_ = lean_box(0);
v___x_57_ = lean_apply_1(v_h__1_52_, v___x_56_);
return v___x_57_;
}
else
{
lean_object* v_one_58_; lean_object* v_n_59_; lean_object* v___x_60_; 
lean_dec(v_h__1_52_);
v_one_58_ = lean_unsigned_to_nat(1u);
v_n_59_ = lean_nat_sub(v_x_51_, v_one_58_);
v___x_60_ = lean_apply_1(v_h__2_53_, v_n_59_);
return v___x_60_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Permutation_0__Nat_factorial_match__1_splitter___redArg___boxed(lean_object* v_x_61_, lean_object* v_h__1_62_, lean_object* v_h__2_63_){
_start:
{
lean_object* v_res_64_; 
v_res_64_ = lp_mathlib___private_Mathlib_Data_List_Permutation_0__Nat_factorial_match__1_splitter___redArg(v_x_61_, v_h__1_62_, v_h__2_63_);
lean_dec(v_x_61_);
return v_res_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Permutation_0__Nat_factorial_match__1_splitter(lean_object* v_motive_65_, lean_object* v_x_66_, lean_object* v_h__1_67_, lean_object* v_h__2_68_){
_start:
{
lean_object* v_zero_69_; uint8_t v_isZero_70_; 
v_zero_69_ = lean_unsigned_to_nat(0u);
v_isZero_70_ = lean_nat_dec_eq(v_x_66_, v_zero_69_);
if (v_isZero_70_ == 1)
{
lean_object* v___x_71_; lean_object* v___x_72_; 
lean_dec(v_h__2_68_);
v___x_71_ = lean_box(0);
v___x_72_ = lean_apply_1(v_h__1_67_, v___x_71_);
return v___x_72_;
}
else
{
lean_object* v_one_73_; lean_object* v_n_74_; lean_object* v___x_75_; 
lean_dec(v_h__1_67_);
v_one_73_ = lean_unsigned_to_nat(1u);
v_n_74_ = lean_nat_sub(v_x_66_, v_one_73_);
v___x_75_ = lean_apply_1(v_h__2_68_, v_n_74_);
return v___x_75_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Permutation_0__Nat_factorial_match__1_splitter___boxed(lean_object* v_motive_76_, lean_object* v_x_77_, lean_object* v_h__1_78_, lean_object* v_h__2_79_){
_start:
{
lean_object* v_res_80_; 
v_res_80_ = lp_mathlib___private_Mathlib_Data_List_Permutation_0__Nat_factorial_match__1_splitter(v_motive_76_, v_x_77_, v_h__1_78_, v_h__2_79_);
lean_dec(v_x_77_);
return v_res_80_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Factorial_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Count(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Duplicate(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_InsertIdx(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Induction(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Data_List_Perm(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Perm_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Finiteness_Attr(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Order_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Permutation(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Factorial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Count(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Duplicate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_InsertIdx(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Induction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_List_Perm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Perm_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Finiteness_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_List_Permutation(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Nat_Factorial_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Count(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Duplicate(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_InsertIdx(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Induction(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_List_Perm(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Perm_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Finiteness_Attr(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Order_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_List_Permutation(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Factorial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Count(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Duplicate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_InsertIdx(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Induction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_List_Perm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Perm_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Finiteness_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Permutation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_List_Permutation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_List_Permutation(builtin);
}
#ifdef __cplusplus
}
#endif
