// Lean compiler output
// Module: Mathlib.Data.DFinsupp.Lex
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Order.Group.PiLex public import Mathlib.Data.DFinsupp.Order public import Mathlib.Data.DFinsupp.NeLocus public import Mathlib.Order.WellFoundedSet
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
lean_object* lp_mathlib_OrderDual_instLinearOrder___redArg(lean_object*);
lean_object* lp_mathlib_DFinsupp_neLocus___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_min___redArg(lean_object*, lean_object*);
lean_object* l_Or_by__cases___redArg(uint8_t, lean_object*, lean_object*);
lean_object* lp_mathlib_Lex_rec___redArg(lean_object*, lean_object*);
uint8_t lp_mathlib_decidableEqOfDecidableLE___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearOrder_toLattice___redArg(lean_object*);
lean_object* lp_mathlib_Lattice_toSemilatticeInf___redArg(lean_object*);
lean_object* lp_mathlib_decidableEqOfDecidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instLTLex(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instLTLex___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instLTColex(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instLTColex___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_DFinsupp_Lex_partialOrder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_DFinsupp_Lex_partialOrder___closed__0 = (const lean_object*)&lp_mathlib_DFinsupp_Lex_partialOrder___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Lex_partialOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Lex_partialOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Colex_partialOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Colex_partialOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg___lam__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg___lam__2, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_Lex_decidableLE___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Lex_decidableLE___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_Lex_decidableLE___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Lex_decidableLE___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_DFinsupp_Lex_decidableLE___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_DFinsupp_Lex_decidableLE___redArg___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_DFinsupp_Lex_decidableLE___redArg___closed__0 = (const lean_object*)&lp_mathlib_DFinsupp_Lex_decidableLE___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_DFinsupp_Lex_decidableLE___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_DFinsupp_Lex_decidableLE___redArg___lam__2___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_DFinsupp_Lex_decidableLE___redArg___closed__1 = (const lean_object*)&lp_mathlib_DFinsupp_Lex_decidableLE___redArg___closed__1_value;
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_Lex_decidableLE___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Lex_decidableLE___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_Lex_decidableLE(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Lex_decidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_Colex_decidableLE___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Colex_decidableLE___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_Colex_decidableLE(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Colex_decidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_Lex_decidableLT___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Lex_decidableLT___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_Lex_decidableLT(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Lex_decidableLT___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_Colex_decidableLT___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Colex_decidableLT___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_Colex_decidableLT(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Colex_decidableLT___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Lex_linearOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Lex_linearOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Lex_linearOrder___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_Lex_linearOrder___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Lex_linearOrder___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Lex_linearOrder___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Lex_linearOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Colex_linearOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Colex_linearOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_Colex_linearOrder___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Colex_linearOrder___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Colex_linearOrder___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Colex_linearOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Lex_orderBot___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Lex_orderBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Lex_orderBot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Lex_orderBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Colex_orderBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Colex_orderBot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Colex_orderBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instLTLex(lean_object* v_00_u03b9_1_, lean_object* v_00_u03b1_2_, lean_object* v_inst_3_, lean_object* v_inst_4_, lean_object* v_inst_5_){
_start:
{
lean_object* v___x_6_; 
v___x_6_ = lean_box(0);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instLTLex___boxed(lean_object* v_00_u03b9_7_, lean_object* v_00_u03b1_8_, lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_inst_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_mathlib_DFinsupp_instLTLex(v_00_u03b9_7_, v_00_u03b1_8_, v_inst_9_, v_inst_10_, v_inst_11_);
lean_dec_ref(v_inst_11_);
lean_dec(v_inst_9_);
return v_res_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instLTColex(lean_object* v_00_u03b9_13_, lean_object* v_00_u03b1_14_, lean_object* v_inst_15_, lean_object* v_inst_16_, lean_object* v_inst_17_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = lean_box(0);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instLTColex___boxed(lean_object* v_00_u03b9_19_, lean_object* v_00_u03b1_20_, lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_inst_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_mathlib_DFinsupp_instLTColex(v_00_u03b9_19_, v_00_u03b1_20_, v_inst_21_, v_inst_22_, v_inst_23_);
lean_dec_ref(v_inst_23_);
lean_dec(v_inst_21_);
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Lex_partialOrder(lean_object* v_00_u03b9_28_, lean_object* v_00_u03b1_29_, lean_object* v_inst_30_, lean_object* v_inst_31_, lean_object* v_inst_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = ((lean_object*)(lp_mathlib_DFinsupp_Lex_partialOrder___closed__0));
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Lex_partialOrder___boxed(lean_object* v_00_u03b9_34_, lean_object* v_00_u03b1_35_, lean_object* v_inst_36_, lean_object* v_inst_37_, lean_object* v_inst_38_){
_start:
{
lean_object* v_res_39_; 
v_res_39_ = lp_mathlib_DFinsupp_Lex_partialOrder(v_00_u03b9_34_, v_00_u03b1_35_, v_inst_36_, v_inst_37_, v_inst_38_);
lean_dec_ref(v_inst_38_);
lean_dec_ref(v_inst_37_);
lean_dec(v_inst_36_);
return v_res_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Colex_partialOrder(lean_object* v_00_u03b9_40_, lean_object* v_00_u03b1_41_, lean_object* v_inst_42_, lean_object* v_inst_43_, lean_object* v_inst_44_){
_start:
{
lean_object* v___x_45_; 
v___x_45_ = ((lean_object*)(lp_mathlib_DFinsupp_Lex_partialOrder___closed__0));
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Colex_partialOrder___boxed(lean_object* v_00_u03b9_46_, lean_object* v_00_u03b1_47_, lean_object* v_inst_48_, lean_object* v_inst_49_, lean_object* v_inst_50_){
_start:
{
lean_object* v_res_51_; 
v_res_51_ = lp_mathlib_DFinsupp_Colex_partialOrder(v_00_u03b9_46_, v_00_u03b1_47_, v_inst_48_, v_inst_49_, v_inst_50_);
lean_dec_ref(v_inst_50_);
lean_dec_ref(v_inst_49_);
lean_dec(v_inst_48_);
return v_res_51_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg___lam__0(lean_object* v_inst_52_, lean_object* v_a_53_, lean_object* v_b_54_){
_start:
{
lean_object* v_toDecidableEq_55_; lean_object* v___x_56_; uint8_t v___x_57_; 
v_toDecidableEq_55_ = lean_ctor_get(v_inst_52_, 5);
lean_inc_ref(v_toDecidableEq_55_);
lean_dec_ref(v_inst_52_);
v___x_56_ = lean_apply_2(v_toDecidableEq_55_, v_a_53_, v_b_54_);
v___x_57_ = lean_unbox(v___x_56_);
return v___x_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg___lam__0___boxed(lean_object* v_inst_58_, lean_object* v_a_59_, lean_object* v_b_60_){
_start:
{
uint8_t v_res_61_; lean_object* v_r_62_; 
v_res_61_ = lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg___lam__0(v_inst_58_, v_a_59_, v_b_60_);
v_r_62_ = lean_box(v_res_61_);
return v_r_62_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg___lam__1(lean_object* v_inst_63_, lean_object* v_a_64_, lean_object* v_a_65_, lean_object* v_b_66_){
_start:
{
lean_object* v___x_67_; lean_object* v_toDecidableEq_68_; lean_object* v___x_69_; uint8_t v___x_70_; 
v___x_67_ = lean_apply_1(v_inst_63_, v_a_64_);
v_toDecidableEq_68_ = lean_ctor_get(v___x_67_, 5);
lean_inc_ref(v_toDecidableEq_68_);
lean_dec_ref(v___x_67_);
v___x_69_ = lean_apply_2(v_toDecidableEq_68_, v_a_65_, v_b_66_);
v___x_70_ = lean_unbox(v___x_69_);
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg___lam__1___boxed(lean_object* v_inst_71_, lean_object* v_a_72_, lean_object* v_a_73_, lean_object* v_b_74_){
_start:
{
uint8_t v_res_75_; lean_object* v_r_76_; 
v_res_75_ = lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg___lam__1(v_inst_71_, v_a_72_, v_a_73_, v_b_74_);
v_r_76_ = lean_box(v_res_75_);
return v_r_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg___lam__2(lean_object* v_f_77_, lean_object* v___y_78_){
_start:
{
lean_object* v_toFun_79_; lean_object* v___x_80_; 
v_toFun_79_ = lean_ctor_get(v_f_77_, 0);
lean_inc(v_toFun_79_);
lean_dec_ref(v_f_77_);
v___x_80_ = lean_apply_1(v_toFun_79_, v___y_78_);
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg___lam__3(lean_object* v_h__lt_81_, lean_object* v_f_82_, lean_object* v_g_83_, lean_object* v_hwit_84_){
_start:
{
lean_object* v___x_85_; 
v___x_85_ = lean_apply_3(v_h__lt_81_, v_f_82_, v_g_83_, lean_box(0));
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg___lam__4(lean_object* v_h__gt_86_, lean_object* v_f_87_, lean_object* v_g_88_, lean_object* v_hwit_89_){
_start:
{
lean_object* v___x_90_; 
v___x_90_ = lean_apply_3(v_h__gt_86_, v_f_87_, v_g_88_, lean_box(0));
return v___x_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg___lam__5(lean_object* v___f_91_, lean_object* v___f_92_, lean_object* v_inst_93_, lean_object* v_f_94_, lean_object* v_inst_95_, lean_object* v_h__eq_96_, lean_object* v_inst_97_, lean_object* v_h__lt_98_, lean_object* v_h__gt_99_, lean_object* v___f_100_, lean_object* v_g_101_){
_start:
{
lean_object* v___x_102_; lean_object* v___x_103_; 
lean_inc_ref(v_g_101_);
lean_inc_ref(v_f_94_);
v___x_102_ = lp_mathlib_DFinsupp_neLocus___redArg(v___f_91_, v___f_92_, v_inst_93_, v_f_94_, v_g_101_);
v___x_103_ = lp_mathlib_Finset_min___redArg(v_inst_95_, v___x_102_);
if (lean_obj_tag(v___x_103_) == 0)
{
lean_object* v___x_104_; 
lean_dec(v___f_100_);
lean_dec(v_h__gt_99_);
lean_dec(v_h__lt_98_);
lean_dec_ref(v_inst_97_);
v___x_104_ = lean_apply_3(v_h__eq_96_, v_f_94_, v_g_101_, lean_box(0));
return v___x_104_;
}
else
{
lean_object* v_val_105_; lean_object* v___x_106_; lean_object* v_toDecidableLT_107_; lean_object* v___f_108_; lean_object* v___f_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; uint8_t v___x_113_; lean_object* v___x_114_; 
lean_dec(v_h__eq_96_);
v_val_105_ = lean_ctor_get(v___x_103_, 0);
lean_inc_n(v_val_105_, 3);
lean_dec_ref_known(v___x_103_, 1);
v___x_106_ = lean_apply_1(v_inst_97_, v_val_105_);
v_toDecidableLT_107_ = lean_ctor_get(v___x_106_, 6);
lean_inc_ref(v_toDecidableLT_107_);
lean_dec_ref(v___x_106_);
lean_inc_ref_n(v_g_101_, 2);
lean_inc_ref_n(v_f_94_, 2);
v___f_108_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg___lam__3), 4, 3);
lean_closure_set(v___f_108_, 0, v_h__lt_98_);
lean_closure_set(v___f_108_, 1, v_f_94_);
lean_closure_set(v___f_108_, 2, v_g_101_);
v___f_109_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg___lam__4), 4, 3);
lean_closure_set(v___f_109_, 0, v_h__gt_99_);
lean_closure_set(v___f_109_, 1, v_f_94_);
lean_closure_set(v___f_109_, 2, v_g_101_);
lean_inc(v___f_100_);
v___x_110_ = lean_apply_2(v___f_100_, v_f_94_, v_val_105_);
v___x_111_ = lean_apply_2(v___f_100_, v_g_101_, v_val_105_);
v___x_112_ = lean_apply_2(v_toDecidableLT_107_, v___x_110_, v___x_111_);
v___x_113_ = lean_unbox(v___x_112_);
v___x_114_ = l_Or_by__cases___redArg(v___x_113_, v___f_108_, v___f_109_);
return v___x_114_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg___lam__5___boxed(lean_object* v___f_115_, lean_object* v___f_116_, lean_object* v_inst_117_, lean_object* v_f_118_, lean_object* v_inst_119_, lean_object* v_h__eq_120_, lean_object* v_inst_121_, lean_object* v_h__lt_122_, lean_object* v_h__gt_123_, lean_object* v___f_124_, lean_object* v_g_125_){
_start:
{
lean_object* v_res_126_; 
v_res_126_ = lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg___lam__5(v___f_115_, v___f_116_, v_inst_117_, v_f_118_, v_inst_119_, v_h__eq_120_, v_inst_121_, v_h__lt_122_, v_h__gt_123_, v___f_124_, v_g_125_);
lean_dec_ref(v_inst_119_);
return v_res_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg___lam__6(lean_object* v___f_127_, lean_object* v___f_128_, lean_object* v_inst_129_, lean_object* v_inst_130_, lean_object* v_h__eq_131_, lean_object* v_inst_132_, lean_object* v_h__lt_133_, lean_object* v_h__gt_134_, lean_object* v___f_135_, lean_object* v_f_136_, lean_object* v_a_137_){
_start:
{
lean_object* v___f_138_; lean_object* v___x_139_; 
v___f_138_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg___lam__5___boxed), 11, 10);
lean_closure_set(v___f_138_, 0, v___f_127_);
lean_closure_set(v___f_138_, 1, v___f_128_);
lean_closure_set(v___f_138_, 2, v_inst_129_);
lean_closure_set(v___f_138_, 3, v_f_136_);
lean_closure_set(v___f_138_, 4, v_inst_130_);
lean_closure_set(v___f_138_, 5, v_h__eq_131_);
lean_closure_set(v___f_138_, 6, v_inst_132_);
lean_closure_set(v___f_138_, 7, v_h__lt_133_);
lean_closure_set(v___f_138_, 8, v_h__gt_134_);
lean_closure_set(v___f_138_, 9, v___f_135_);
v___x_139_ = lp_mathlib_Lex_rec___redArg(v___f_138_, v_a_137_);
return v___x_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg(lean_object* v_inst_141_, lean_object* v_inst_142_, lean_object* v_inst_143_, lean_object* v_h__lt_144_, lean_object* v_h__eq_145_, lean_object* v_h__gt_146_, lean_object* v_a_147_, lean_object* v_g_148_){
_start:
{
lean_object* v___f_149_; lean_object* v___f_150_; lean_object* v___f_151_; lean_object* v___f_152_; lean_object* v___x_47__overap_153_; lean_object* v___x_154_; 
lean_inc_ref(v_inst_142_);
v___f_149_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_149_, 0, v_inst_142_);
lean_inc_ref(v_inst_143_);
v___f_150_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg___lam__1___boxed), 4, 1);
lean_closure_set(v___f_150_, 0, v_inst_143_);
v___f_151_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg___closed__0));
v___f_152_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg___lam__6), 11, 9);
lean_closure_set(v___f_152_, 0, v___f_149_);
lean_closure_set(v___f_152_, 1, v___f_150_);
lean_closure_set(v___f_152_, 2, v_inst_141_);
lean_closure_set(v___f_152_, 3, v_inst_142_);
lean_closure_set(v___f_152_, 4, v_h__eq_145_);
lean_closure_set(v___f_152_, 5, v_inst_143_);
lean_closure_set(v___f_152_, 6, v_h__lt_144_);
lean_closure_set(v___f_152_, 7, v_h__gt_146_);
lean_closure_set(v___f_152_, 8, v___f_151_);
v___x_47__overap_153_ = lp_mathlib_Lex_rec___redArg(v___f_152_, v_a_147_);
v___x_154_ = lean_apply_1(v___x_47__overap_153_, v_g_148_);
return v___x_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec(lean_object* v_00_u03b9_155_, lean_object* v_00_u03b1_156_, lean_object* v_inst_157_, lean_object* v_inst_158_, lean_object* v_inst_159_, lean_object* v_P_160_, lean_object* v_h__lt_161_, lean_object* v_h__eq_162_, lean_object* v_h__gt_163_, lean_object* v_a_164_, lean_object* v_g_165_){
_start:
{
lean_object* v___x_166_; 
v___x_166_ = lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg(v_inst_157_, v_inst_158_, v_inst_159_, v_h__lt_161_, v_h__eq_162_, v_h__gt_163_, v_a_164_, v_g_165_);
return v___x_166_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_Lex_decidableLE___redArg___lam__0(lean_object* v_f_167_, lean_object* v_g_168_, lean_object* v_h_169_){
_start:
{
uint8_t v___x_170_; 
v___x_170_ = 1;
return v___x_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Lex_decidableLE___redArg___lam__0___boxed(lean_object* v_f_171_, lean_object* v_g_172_, lean_object* v_h_173_){
_start:
{
uint8_t v_res_174_; lean_object* v_r_175_; 
v_res_174_ = lp_mathlib_DFinsupp_Lex_decidableLE___redArg___lam__0(v_f_171_, v_g_172_, v_h_173_);
lean_dec_ref(v_g_172_);
lean_dec_ref(v_f_171_);
v_r_175_ = lean_box(v_res_174_);
return v_r_175_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_Lex_decidableLE___redArg___lam__2(lean_object* v_f_176_, lean_object* v_g_177_, lean_object* v_h_178_){
_start:
{
uint8_t v___x_179_; 
v___x_179_ = 0;
return v___x_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Lex_decidableLE___redArg___lam__2___boxed(lean_object* v_f_180_, lean_object* v_g_181_, lean_object* v_h_182_){
_start:
{
uint8_t v_res_183_; lean_object* v_r_184_; 
v_res_183_ = lp_mathlib_DFinsupp_Lex_decidableLE___redArg___lam__2(v_f_180_, v_g_181_, v_h_182_);
lean_dec_ref(v_g_181_);
lean_dec_ref(v_f_180_);
v_r_184_ = lean_box(v_res_183_);
return v_r_184_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_Lex_decidableLE___redArg(lean_object* v_inst_187_, lean_object* v_inst_188_, lean_object* v_inst_189_, lean_object* v_f_190_, lean_object* v_g_191_){
_start:
{
lean_object* v___f_192_; lean_object* v___f_193_; lean_object* v___x_194_; uint8_t v___x_195_; 
v___f_192_ = ((lean_object*)(lp_mathlib_DFinsupp_Lex_decidableLE___redArg___closed__0));
v___f_193_ = ((lean_object*)(lp_mathlib_DFinsupp_Lex_decidableLE___redArg___closed__1));
v___x_194_ = lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg(v_inst_187_, v_inst_188_, v_inst_189_, v___f_192_, v___f_192_, v___f_193_, v_f_190_, v_g_191_);
v___x_195_ = lean_unbox(v___x_194_);
lean_dec(v___x_194_);
return v___x_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Lex_decidableLE___redArg___boxed(lean_object* v_inst_196_, lean_object* v_inst_197_, lean_object* v_inst_198_, lean_object* v_f_199_, lean_object* v_g_200_){
_start:
{
uint8_t v_res_201_; lean_object* v_r_202_; 
v_res_201_ = lp_mathlib_DFinsupp_Lex_decidableLE___redArg(v_inst_196_, v_inst_197_, v_inst_198_, v_f_199_, v_g_200_);
v_r_202_ = lean_box(v_res_201_);
return v_r_202_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_Lex_decidableLE(lean_object* v_00_u03b9_203_, lean_object* v_00_u03b1_204_, lean_object* v_inst_205_, lean_object* v_inst_206_, lean_object* v_inst_207_, lean_object* v_f_208_, lean_object* v_g_209_){
_start:
{
uint8_t v___x_210_; 
v___x_210_ = lp_mathlib_DFinsupp_Lex_decidableLE___redArg(v_inst_205_, v_inst_206_, v_inst_207_, v_f_208_, v_g_209_);
return v___x_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Lex_decidableLE___boxed(lean_object* v_00_u03b9_211_, lean_object* v_00_u03b1_212_, lean_object* v_inst_213_, lean_object* v_inst_214_, lean_object* v_inst_215_, lean_object* v_f_216_, lean_object* v_g_217_){
_start:
{
uint8_t v_res_218_; lean_object* v_r_219_; 
v_res_218_ = lp_mathlib_DFinsupp_Lex_decidableLE(v_00_u03b9_211_, v_00_u03b1_212_, v_inst_213_, v_inst_214_, v_inst_215_, v_f_216_, v_g_217_);
v_r_219_ = lean_box(v_res_218_);
return v_r_219_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_Colex_decidableLE___redArg(lean_object* v_inst_220_, lean_object* v_inst_221_, lean_object* v_inst_222_, lean_object* v_a_223_, lean_object* v_b_224_){
_start:
{
lean_object* v___x_225_; uint8_t v___x_226_; 
v___x_225_ = lp_mathlib_OrderDual_instLinearOrder___redArg(v_inst_221_);
v___x_226_ = lp_mathlib_DFinsupp_Lex_decidableLE___redArg(v_inst_220_, v___x_225_, v_inst_222_, v_a_223_, v_b_224_);
return v___x_226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Colex_decidableLE___redArg___boxed(lean_object* v_inst_227_, lean_object* v_inst_228_, lean_object* v_inst_229_, lean_object* v_a_230_, lean_object* v_b_231_){
_start:
{
uint8_t v_res_232_; lean_object* v_r_233_; 
v_res_232_ = lp_mathlib_DFinsupp_Colex_decidableLE___redArg(v_inst_227_, v_inst_228_, v_inst_229_, v_a_230_, v_b_231_);
v_r_233_ = lean_box(v_res_232_);
return v_r_233_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_Colex_decidableLE(lean_object* v_00_u03b9_234_, lean_object* v_00_u03b1_235_, lean_object* v_inst_236_, lean_object* v_inst_237_, lean_object* v_inst_238_, lean_object* v_a_239_, lean_object* v_b_240_){
_start:
{
uint8_t v___x_241_; 
v___x_241_ = lp_mathlib_DFinsupp_Colex_decidableLE___redArg(v_inst_236_, v_inst_237_, v_inst_238_, v_a_239_, v_b_240_);
return v___x_241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Colex_decidableLE___boxed(lean_object* v_00_u03b9_242_, lean_object* v_00_u03b1_243_, lean_object* v_inst_244_, lean_object* v_inst_245_, lean_object* v_inst_246_, lean_object* v_a_247_, lean_object* v_b_248_){
_start:
{
uint8_t v_res_249_; lean_object* v_r_250_; 
v_res_249_ = lp_mathlib_DFinsupp_Colex_decidableLE(v_00_u03b9_242_, v_00_u03b1_243_, v_inst_244_, v_inst_245_, v_inst_246_, v_a_247_, v_b_248_);
v_r_250_ = lean_box(v_res_249_);
return v_r_250_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_Lex_decidableLT___redArg(lean_object* v_inst_251_, lean_object* v_inst_252_, lean_object* v_inst_253_, lean_object* v_f_254_, lean_object* v_g_255_){
_start:
{
lean_object* v___f_256_; lean_object* v___f_257_; lean_object* v___x_258_; uint8_t v___x_259_; 
v___f_256_ = ((lean_object*)(lp_mathlib_DFinsupp_Lex_decidableLE___redArg___closed__0));
v___f_257_ = ((lean_object*)(lp_mathlib_DFinsupp_Lex_decidableLE___redArg___closed__1));
v___x_258_ = lp_mathlib___private_Mathlib_Data_DFinsupp_Lex_0__DFinsupp_lt__trichotomy__rec___redArg(v_inst_251_, v_inst_252_, v_inst_253_, v___f_256_, v___f_257_, v___f_257_, v_f_254_, v_g_255_);
v___x_259_ = lean_unbox(v___x_258_);
lean_dec(v___x_258_);
return v___x_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Lex_decidableLT___redArg___boxed(lean_object* v_inst_260_, lean_object* v_inst_261_, lean_object* v_inst_262_, lean_object* v_f_263_, lean_object* v_g_264_){
_start:
{
uint8_t v_res_265_; lean_object* v_r_266_; 
v_res_265_ = lp_mathlib_DFinsupp_Lex_decidableLT___redArg(v_inst_260_, v_inst_261_, v_inst_262_, v_f_263_, v_g_264_);
v_r_266_ = lean_box(v_res_265_);
return v_r_266_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_Lex_decidableLT(lean_object* v_00_u03b9_267_, lean_object* v_00_u03b1_268_, lean_object* v_inst_269_, lean_object* v_inst_270_, lean_object* v_inst_271_, lean_object* v_f_272_, lean_object* v_g_273_){
_start:
{
uint8_t v___x_274_; 
v___x_274_ = lp_mathlib_DFinsupp_Lex_decidableLT___redArg(v_inst_269_, v_inst_270_, v_inst_271_, v_f_272_, v_g_273_);
return v___x_274_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Lex_decidableLT___boxed(lean_object* v_00_u03b9_275_, lean_object* v_00_u03b1_276_, lean_object* v_inst_277_, lean_object* v_inst_278_, lean_object* v_inst_279_, lean_object* v_f_280_, lean_object* v_g_281_){
_start:
{
uint8_t v_res_282_; lean_object* v_r_283_; 
v_res_282_ = lp_mathlib_DFinsupp_Lex_decidableLT(v_00_u03b9_275_, v_00_u03b1_276_, v_inst_277_, v_inst_278_, v_inst_279_, v_f_280_, v_g_281_);
v_r_283_ = lean_box(v_res_282_);
return v_r_283_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_Colex_decidableLT___redArg(lean_object* v_inst_284_, lean_object* v_inst_285_, lean_object* v_inst_286_, lean_object* v_a_287_, lean_object* v_b_288_){
_start:
{
lean_object* v___x_289_; uint8_t v___x_290_; 
v___x_289_ = lp_mathlib_OrderDual_instLinearOrder___redArg(v_inst_285_);
v___x_290_ = lp_mathlib_DFinsupp_Lex_decidableLT___redArg(v_inst_284_, v___x_289_, v_inst_286_, v_a_287_, v_b_288_);
return v___x_290_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Colex_decidableLT___redArg___boxed(lean_object* v_inst_291_, lean_object* v_inst_292_, lean_object* v_inst_293_, lean_object* v_a_294_, lean_object* v_b_295_){
_start:
{
uint8_t v_res_296_; lean_object* v_r_297_; 
v_res_296_ = lp_mathlib_DFinsupp_Colex_decidableLT___redArg(v_inst_291_, v_inst_292_, v_inst_293_, v_a_294_, v_b_295_);
v_r_297_ = lean_box(v_res_296_);
return v_r_297_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_Colex_decidableLT(lean_object* v_00_u03b9_298_, lean_object* v_00_u03b1_299_, lean_object* v_inst_300_, lean_object* v_inst_301_, lean_object* v_inst_302_, lean_object* v_a_303_, lean_object* v_b_304_){
_start:
{
uint8_t v___x_305_; 
v___x_305_ = lp_mathlib_DFinsupp_Colex_decidableLT___redArg(v_inst_300_, v_inst_301_, v_inst_302_, v_a_303_, v_b_304_);
return v___x_305_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Colex_decidableLT___boxed(lean_object* v_00_u03b9_306_, lean_object* v_00_u03b1_307_, lean_object* v_inst_308_, lean_object* v_inst_309_, lean_object* v_inst_310_, lean_object* v_a_311_, lean_object* v_b_312_){
_start:
{
uint8_t v_res_313_; lean_object* v_r_314_; 
v_res_313_ = lp_mathlib_DFinsupp_Colex_decidableLT(v_00_u03b9_306_, v_00_u03b1_307_, v_inst_308_, v_inst_309_, v_inst_310_, v_a_311_, v_b_312_);
v_r_314_ = lean_box(v_res_313_);
return v_r_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Lex_linearOrder___redArg___lam__0(lean_object* v_inst_315_, lean_object* v_inst_316_, lean_object* v_inst_317_, lean_object* v_a_318_, lean_object* v_b_319_){
_start:
{
uint8_t v___x_320_; 
lean_inc_ref(v_b_319_);
lean_inc_ref(v_a_318_);
v___x_320_ = lp_mathlib_DFinsupp_Lex_decidableLE___redArg(v_inst_315_, v_inst_316_, v_inst_317_, v_a_318_, v_b_319_);
if (v___x_320_ == 0)
{
lean_dec_ref(v_b_319_);
return v_a_318_;
}
else
{
lean_dec_ref(v_a_318_);
return v_b_319_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Lex_linearOrder___redArg___lam__1(lean_object* v_inst_321_, lean_object* v_inst_322_, lean_object* v_inst_323_, lean_object* v_a_324_, lean_object* v_b_325_){
_start:
{
uint8_t v___x_326_; 
lean_inc_ref(v_b_325_);
lean_inc_ref(v_a_324_);
v___x_326_ = lp_mathlib_DFinsupp_Lex_decidableLE___redArg(v_inst_321_, v_inst_322_, v_inst_323_, v_a_324_, v_b_325_);
if (v___x_326_ == 0)
{
lean_dec_ref(v_a_324_);
return v_b_325_;
}
else
{
lean_dec_ref(v_b_325_);
return v_a_324_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Lex_linearOrder___redArg___lam__2(lean_object* v_inst_327_, lean_object* v_i_328_){
_start:
{
lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v_toPartialOrder_332_; 
v___x_329_ = lean_apply_1(v_inst_327_, v_i_328_);
v___x_330_ = lp_mathlib_LinearOrder_toLattice___redArg(v___x_329_);
lean_dec_ref(v___x_329_);
v___x_331_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_330_);
v_toPartialOrder_332_ = lean_ctor_get(v___x_331_, 0);
lean_inc_ref(v_toPartialOrder_332_);
lean_dec_ref(v___x_331_);
return v_toPartialOrder_332_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_Lex_linearOrder___redArg___lam__3(lean_object* v_inst_333_, lean_object* v_inst_334_, lean_object* v_inst_335_, lean_object* v_a_336_, lean_object* v_b_337_){
_start:
{
uint8_t v___x_338_; 
lean_inc_ref(v_b_337_);
lean_inc_ref(v_a_336_);
lean_inc_ref(v_inst_335_);
lean_inc_ref(v_inst_334_);
lean_inc(v_inst_333_);
v___x_338_ = lp_mathlib_DFinsupp_Lex_decidableLT___redArg(v_inst_333_, v_inst_334_, v_inst_335_, v_a_336_, v_b_337_);
if (v___x_338_ == 0)
{
lean_object* v___x_339_; uint8_t v___x_340_; 
v___x_339_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_Lex_decidableLE___boxed), 7, 5);
lean_closure_set(v___x_339_, 0, lean_box(0));
lean_closure_set(v___x_339_, 1, lean_box(0));
lean_closure_set(v___x_339_, 2, v_inst_333_);
lean_closure_set(v___x_339_, 3, v_inst_334_);
lean_closure_set(v___x_339_, 4, v_inst_335_);
v___x_340_ = lp_mathlib_decidableEqOfDecidableLE___redArg(v___x_339_, v_a_336_, v_b_337_);
if (v___x_340_ == 0)
{
uint8_t v___x_341_; 
v___x_341_ = 2;
return v___x_341_;
}
else
{
uint8_t v___x_342_; 
v___x_342_ = 1;
return v___x_342_;
}
}
else
{
uint8_t v___x_343_; 
lean_dec_ref(v_b_337_);
lean_dec_ref(v_a_336_);
lean_dec_ref(v_inst_335_);
lean_dec_ref(v_inst_334_);
lean_dec(v_inst_333_);
v___x_343_ = 0;
return v___x_343_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Lex_linearOrder___redArg___lam__3___boxed(lean_object* v_inst_344_, lean_object* v_inst_345_, lean_object* v_inst_346_, lean_object* v_a_347_, lean_object* v_b_348_){
_start:
{
uint8_t v_res_349_; lean_object* v_r_350_; 
v_res_349_ = lp_mathlib_DFinsupp_Lex_linearOrder___redArg___lam__3(v_inst_344_, v_inst_345_, v_inst_346_, v_a_347_, v_b_348_);
v_r_350_ = lean_box(v_res_349_);
return v_r_350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Lex_linearOrder___redArg(lean_object* v_inst_351_, lean_object* v_inst_352_, lean_object* v_inst_353_){
_start:
{
lean_object* v___f_354_; lean_object* v___f_355_; lean_object* v___f_356_; lean_object* v___f_357_; lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; lean_object* v___x_361_; lean_object* v___x_362_; 
lean_inc_ref_n(v_inst_353_, 5);
lean_inc_ref_n(v_inst_352_, 4);
lean_inc_n(v_inst_351_, 4);
v___f_354_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_Lex_linearOrder___redArg___lam__0), 5, 3);
lean_closure_set(v___f_354_, 0, v_inst_351_);
lean_closure_set(v___f_354_, 1, v_inst_352_);
lean_closure_set(v___f_354_, 2, v_inst_353_);
v___f_355_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_Lex_linearOrder___redArg___lam__1), 5, 3);
lean_closure_set(v___f_355_, 0, v_inst_351_);
lean_closure_set(v___f_355_, 1, v_inst_352_);
lean_closure_set(v___f_355_, 2, v_inst_353_);
v___f_356_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_Lex_linearOrder___redArg___lam__2), 2, 1);
lean_closure_set(v___f_356_, 0, v_inst_353_);
v___f_357_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_Lex_linearOrder___redArg___lam__3___boxed), 5, 3);
lean_closure_set(v___f_357_, 0, v_inst_351_);
lean_closure_set(v___f_357_, 1, v_inst_352_);
lean_closure_set(v___f_357_, 2, v_inst_353_);
v___x_358_ = lp_mathlib_DFinsupp_Lex_partialOrder(lean_box(0), lean_box(0), v_inst_351_, v_inst_352_, v___f_356_);
lean_dec_ref(v___f_356_);
v___x_359_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_Lex_decidableLE___boxed), 7, 5);
lean_closure_set(v___x_359_, 0, lean_box(0));
lean_closure_set(v___x_359_, 1, lean_box(0));
lean_closure_set(v___x_359_, 2, v_inst_351_);
lean_closure_set(v___x_359_, 3, v_inst_352_);
lean_closure_set(v___x_359_, 4, v_inst_353_);
lean_inc_ref(v___x_359_);
lean_inc_ref(v___x_358_);
v___x_360_ = lean_alloc_closure((void*)(lp_mathlib_decidableEqOfDecidableLE___boxed), 5, 3);
lean_closure_set(v___x_360_, 0, lean_box(0));
lean_closure_set(v___x_360_, 1, v___x_358_);
lean_closure_set(v___x_360_, 2, v___x_359_);
v___x_361_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_Lex_decidableLT___boxed), 7, 5);
lean_closure_set(v___x_361_, 0, lean_box(0));
lean_closure_set(v___x_361_, 1, lean_box(0));
lean_closure_set(v___x_361_, 2, v_inst_351_);
lean_closure_set(v___x_361_, 3, v_inst_352_);
lean_closure_set(v___x_361_, 4, v_inst_353_);
v___x_362_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_362_, 0, v___x_358_);
lean_ctor_set(v___x_362_, 1, v___f_355_);
lean_ctor_set(v___x_362_, 2, v___f_354_);
lean_ctor_set(v___x_362_, 3, v___f_357_);
lean_ctor_set(v___x_362_, 4, v___x_359_);
lean_ctor_set(v___x_362_, 5, v___x_360_);
lean_ctor_set(v___x_362_, 6, v___x_361_);
return v___x_362_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Lex_linearOrder(lean_object* v_00_u03b9_363_, lean_object* v_00_u03b1_364_, lean_object* v_inst_365_, lean_object* v_inst_366_, lean_object* v_inst_367_){
_start:
{
lean_object* v___x_368_; 
v___x_368_ = lp_mathlib_DFinsupp_Lex_linearOrder___redArg(v_inst_365_, v_inst_366_, v_inst_367_);
return v___x_368_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Colex_linearOrder___redArg___lam__0(lean_object* v_inst_369_, lean_object* v_inst_370_, lean_object* v_inst_371_, lean_object* v_a_372_, lean_object* v_b_373_){
_start:
{
uint8_t v___x_374_; 
lean_inc_ref(v_b_373_);
lean_inc_ref(v_a_372_);
v___x_374_ = lp_mathlib_DFinsupp_Colex_decidableLE___redArg(v_inst_369_, v_inst_370_, v_inst_371_, v_a_372_, v_b_373_);
if (v___x_374_ == 0)
{
lean_dec_ref(v_b_373_);
return v_a_372_;
}
else
{
lean_dec_ref(v_a_372_);
return v_b_373_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Colex_linearOrder___redArg___lam__1(lean_object* v_inst_375_, lean_object* v_inst_376_, lean_object* v_inst_377_, lean_object* v_a_378_, lean_object* v_b_379_){
_start:
{
uint8_t v___x_380_; 
lean_inc_ref(v_b_379_);
lean_inc_ref(v_a_378_);
v___x_380_ = lp_mathlib_DFinsupp_Colex_decidableLE___redArg(v_inst_375_, v_inst_376_, v_inst_377_, v_a_378_, v_b_379_);
if (v___x_380_ == 0)
{
lean_dec_ref(v_a_378_);
return v_b_379_;
}
else
{
lean_dec_ref(v_b_379_);
return v_a_378_;
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_Colex_linearOrder___redArg___lam__3(lean_object* v_inst_381_, lean_object* v_inst_382_, lean_object* v_inst_383_, lean_object* v_a_384_, lean_object* v_b_385_){
_start:
{
uint8_t v___x_386_; 
lean_inc_ref(v_b_385_);
lean_inc_ref(v_a_384_);
lean_inc_ref(v_inst_383_);
lean_inc_ref(v_inst_382_);
lean_inc(v_inst_381_);
v___x_386_ = lp_mathlib_DFinsupp_Colex_decidableLT___redArg(v_inst_381_, v_inst_382_, v_inst_383_, v_a_384_, v_b_385_);
if (v___x_386_ == 0)
{
lean_object* v___x_387_; uint8_t v___x_388_; 
v___x_387_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_Colex_decidableLE___boxed), 7, 5);
lean_closure_set(v___x_387_, 0, lean_box(0));
lean_closure_set(v___x_387_, 1, lean_box(0));
lean_closure_set(v___x_387_, 2, v_inst_381_);
lean_closure_set(v___x_387_, 3, v_inst_382_);
lean_closure_set(v___x_387_, 4, v_inst_383_);
v___x_388_ = lp_mathlib_decidableEqOfDecidableLE___redArg(v___x_387_, v_a_384_, v_b_385_);
if (v___x_388_ == 0)
{
uint8_t v___x_389_; 
v___x_389_ = 2;
return v___x_389_;
}
else
{
uint8_t v___x_390_; 
v___x_390_ = 1;
return v___x_390_;
}
}
else
{
uint8_t v___x_391_; 
lean_dec_ref(v_b_385_);
lean_dec_ref(v_a_384_);
lean_dec_ref(v_inst_383_);
lean_dec_ref(v_inst_382_);
lean_dec(v_inst_381_);
v___x_391_ = 0;
return v___x_391_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Colex_linearOrder___redArg___lam__3___boxed(lean_object* v_inst_392_, lean_object* v_inst_393_, lean_object* v_inst_394_, lean_object* v_a_395_, lean_object* v_b_396_){
_start:
{
uint8_t v_res_397_; lean_object* v_r_398_; 
v_res_397_ = lp_mathlib_DFinsupp_Colex_linearOrder___redArg___lam__3(v_inst_392_, v_inst_393_, v_inst_394_, v_a_395_, v_b_396_);
v_r_398_ = lean_box(v_res_397_);
return v_r_398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Colex_linearOrder___redArg(lean_object* v_inst_399_, lean_object* v_inst_400_, lean_object* v_inst_401_){
_start:
{
lean_object* v___f_402_; lean_object* v___f_403_; lean_object* v___f_404_; lean_object* v___f_405_; lean_object* v___x_406_; lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v___x_410_; 
lean_inc_ref_n(v_inst_401_, 5);
lean_inc_ref_n(v_inst_400_, 4);
lean_inc_n(v_inst_399_, 4);
v___f_402_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_Colex_linearOrder___redArg___lam__0), 5, 3);
lean_closure_set(v___f_402_, 0, v_inst_399_);
lean_closure_set(v___f_402_, 1, v_inst_400_);
lean_closure_set(v___f_402_, 2, v_inst_401_);
v___f_403_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_Colex_linearOrder___redArg___lam__1), 5, 3);
lean_closure_set(v___f_403_, 0, v_inst_399_);
lean_closure_set(v___f_403_, 1, v_inst_400_);
lean_closure_set(v___f_403_, 2, v_inst_401_);
v___f_404_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_Lex_linearOrder___redArg___lam__2), 2, 1);
lean_closure_set(v___f_404_, 0, v_inst_401_);
v___f_405_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_Colex_linearOrder___redArg___lam__3___boxed), 5, 3);
lean_closure_set(v___f_405_, 0, v_inst_399_);
lean_closure_set(v___f_405_, 1, v_inst_400_);
lean_closure_set(v___f_405_, 2, v_inst_401_);
v___x_406_ = lp_mathlib_DFinsupp_Colex_partialOrder(lean_box(0), lean_box(0), v_inst_399_, v_inst_400_, v___f_404_);
lean_dec_ref(v___f_404_);
v___x_407_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_Colex_decidableLE___boxed), 7, 5);
lean_closure_set(v___x_407_, 0, lean_box(0));
lean_closure_set(v___x_407_, 1, lean_box(0));
lean_closure_set(v___x_407_, 2, v_inst_399_);
lean_closure_set(v___x_407_, 3, v_inst_400_);
lean_closure_set(v___x_407_, 4, v_inst_401_);
lean_inc_ref(v___x_407_);
lean_inc_ref(v___x_406_);
v___x_408_ = lean_alloc_closure((void*)(lp_mathlib_decidableEqOfDecidableLE___boxed), 5, 3);
lean_closure_set(v___x_408_, 0, lean_box(0));
lean_closure_set(v___x_408_, 1, v___x_406_);
lean_closure_set(v___x_408_, 2, v___x_407_);
v___x_409_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_Colex_decidableLT___boxed), 7, 5);
lean_closure_set(v___x_409_, 0, lean_box(0));
lean_closure_set(v___x_409_, 1, lean_box(0));
lean_closure_set(v___x_409_, 2, v_inst_399_);
lean_closure_set(v___x_409_, 3, v_inst_400_);
lean_closure_set(v___x_409_, 4, v_inst_401_);
v___x_410_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_410_, 0, v___x_406_);
lean_ctor_set(v___x_410_, 1, v___f_403_);
lean_ctor_set(v___x_410_, 2, v___f_402_);
lean_ctor_set(v___x_410_, 3, v___f_405_);
lean_ctor_set(v___x_410_, 4, v___x_407_);
lean_ctor_set(v___x_410_, 5, v___x_408_);
lean_ctor_set(v___x_410_, 6, v___x_409_);
return v___x_410_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Colex_linearOrder(lean_object* v_00_u03b9_411_, lean_object* v_00_u03b1_412_, lean_object* v_inst_413_, lean_object* v_inst_414_, lean_object* v_inst_415_){
_start:
{
lean_object* v___x_416_; 
v___x_416_ = lp_mathlib_DFinsupp_Colex_linearOrder___redArg(v_inst_413_, v_inst_414_, v_inst_415_);
return v___x_416_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Lex_orderBot___redArg___lam__0(lean_object* v_inst_417_, lean_object* v_x_418_){
_start:
{
lean_object* v___x_419_; lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v_toZero_422_; 
v___x_419_ = lean_apply_1(v_inst_417_, v_x_418_);
v___x_420_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v___x_419_);
lean_dec_ref(v___x_419_);
v___x_421_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_420_);
v_toZero_422_ = lean_ctor_get(v___x_421_, 0);
lean_inc(v_toZero_422_);
lean_dec_ref(v___x_421_);
return v_toZero_422_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Lex_orderBot___redArg(lean_object* v_inst_423_){
_start:
{
lean_object* v___f_424_; lean_object* v___x_425_; lean_object* v___x_426_; 
v___f_424_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_Lex_orderBot___redArg___lam__0), 2, 1);
lean_closure_set(v___f_424_, 0, v_inst_423_);
v___x_425_ = lean_box(0);
v___x_426_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_426_, 0, v___f_424_);
lean_ctor_set(v___x_426_, 1, v___x_425_);
return v___x_426_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Lex_orderBot(lean_object* v_00_u03b9_427_, lean_object* v_00_u03b1_428_, lean_object* v_inst_429_, lean_object* v_inst_430_, lean_object* v_inst_431_, lean_object* v_inst_432_){
_start:
{
lean_object* v___x_433_; 
v___x_433_ = lp_mathlib_DFinsupp_Lex_orderBot___redArg(v_inst_430_);
return v___x_433_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Lex_orderBot___boxed(lean_object* v_00_u03b9_434_, lean_object* v_00_u03b1_435_, lean_object* v_inst_436_, lean_object* v_inst_437_, lean_object* v_inst_438_, lean_object* v_inst_439_){
_start:
{
lean_object* v_res_440_; 
v_res_440_ = lp_mathlib_DFinsupp_Lex_orderBot(v_00_u03b9_434_, v_00_u03b1_435_, v_inst_436_, v_inst_437_, v_inst_438_, v_inst_439_);
lean_dec_ref(v_inst_438_);
lean_dec_ref(v_inst_436_);
return v_res_440_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Colex_orderBot___redArg(lean_object* v_inst_441_){
_start:
{
lean_object* v___f_442_; lean_object* v___x_443_; lean_object* v___x_444_; 
v___f_442_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_Lex_orderBot___redArg___lam__0), 2, 1);
lean_closure_set(v___f_442_, 0, v_inst_441_);
v___x_443_ = lean_box(0);
v___x_444_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_444_, 0, v___f_442_);
lean_ctor_set(v___x_444_, 1, v___x_443_);
return v___x_444_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Colex_orderBot(lean_object* v_00_u03b9_445_, lean_object* v_00_u03b1_446_, lean_object* v_inst_447_, lean_object* v_inst_448_, lean_object* v_inst_449_, lean_object* v_inst_450_){
_start:
{
lean_object* v___x_451_; 
v___x_451_ = lp_mathlib_DFinsupp_Colex_orderBot___redArg(v_inst_448_);
return v___x_451_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_Colex_orderBot___boxed(lean_object* v_00_u03b9_452_, lean_object* v_00_u03b1_453_, lean_object* v_inst_454_, lean_object* v_inst_455_, lean_object* v_inst_456_, lean_object* v_inst_457_){
_start:
{
lean_object* v_res_458_; 
v_res_458_ = lp_mathlib_DFinsupp_Colex_orderBot(v_00_u03b9_452_, v_00_u03b1_453_, v_inst_454_, v_inst_455_, v_inst_456_, v_inst_457_);
lean_dec_ref(v_inst_456_);
lean_dec_ref(v_inst_454_);
return v_res_458_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_PiLex(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_DFinsupp_Order(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_DFinsupp_NeLocus(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_WellFoundedSet(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_DFinsupp_Lex(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_PiLex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_DFinsupp_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_DFinsupp_NeLocus(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_WellFoundedSet(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_DFinsupp_Lex(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_PiLex(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_DFinsupp_Order(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_DFinsupp_NeLocus(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_WellFoundedSet(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_DFinsupp_Lex(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_PiLex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_DFinsupp_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_DFinsupp_NeLocus(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_WellFoundedSet(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_DFinsupp_Lex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_DFinsupp_Lex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_DFinsupp_Lex(builtin);
}
#ifdef __cplusplus
}
#endif
