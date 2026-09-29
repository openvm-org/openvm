// Lean compiler output
// Module: Mathlib.Basic.Denumerable
// Imports: public import Init public meta import Init public import Mathlib.Data.Fintype.EquivFin public import Mathlib.Data.List.MinMax public import Mathlib.Data.Nat.Order.Lemmas public import Mathlib.Logic.Encodable.Basic
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
lean_object* lp_mathlib_Equiv_plift(lean_object*);
lean_object* lp_mathlib_Encodable_ofEquiv___redArg(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_List_range(lean_object*);
lean_object* l_List_countP_go___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_findX___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Nat_encodable;
lean_object* lp_mathlib_Sum_encodable___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_sigmaEquivProd(lean_object*, lean_object*);
lean_object* lp_mathlib_Sigma_encodable___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Option_encodable___redArg(lean_object*);
lean_object* lp_mathlib_Encodable_decidableRangeEncode___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Encodable_equivRangeEncode___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_ulift(lean_object*);
extern lean_object* lp_mathlib_Int_encodable;
extern lean_object* lp_mathlib_PNat_encodable;
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_ofNat___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_ofNat(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_eqv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_eqv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_mk_x27___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_mk_x27___redArg___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_mk_x27___redArg___lam__2(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Denumerable_mk_x27___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Denumerable_mk_x27___redArg___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Denumerable_mk_x27___redArg___closed__0 = (const lean_object*)&lp_mathlib_Denumerable_mk_x27___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_mk_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_mk_x27(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_ofEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_ofEquiv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_equiv_u2082___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_equiv_u2082(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_nat;
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_option___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_option(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Basic_Denumerable_0__Encodable_decodeSum_match__1_splitter___redArg(uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Basic_Denumerable_0__Encodable_decodeSum_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Basic_Denumerable_0__Encodable_decodeSum_match__1_splitter(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Basic_Denumerable_0__Encodable_decodeSum_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_sum___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_sum(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_sigma___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_sigma___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_sigma(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_prod___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_prod___redArg___lam__0___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Denumerable_prod___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Denumerable_prod___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Denumerable_prod___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Denumerable_prod___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_prod___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_prod(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_int;
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_pnat;
static lean_once_cell_t lp_mathlib_Denumerable_ulift___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Denumerable_ulift___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_ulift___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_ulift(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Denumerable_plift___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Denumerable_plift___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_plift___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_plift(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_pair___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_pair(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Nat_Subtype_succ___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_Subtype_succ___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_Subtype_succ___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_Subtype_succ(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_Subtype_ofNat___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_Subtype_ofNat___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_Subtype_ofNat(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_Subtype_ofNat___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Basic_Denumerable_0__Nat_Subtype_toFunAux___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Basic_Denumerable_0__Nat_Subtype_toFunAux___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Basic_Denumerable_0__Nat_Subtype_toFunAux___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Basic_Denumerable_0__Nat_Subtype_toFunAux(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Basic_Denumerable_0__Nat_Subtype_ofNat_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Basic_Denumerable_0__Nat_Subtype_ofNat_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Basic_Denumerable_0__Nat_Subtype_ofNat_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Basic_Denumerable_0__Nat_Subtype_ofNat_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_Subtype_denumerable___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_Subtype_denumerable(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_ofEncodableOfInfinite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_ofEncodableOfInfinite(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_ofNat___redArg(lean_object* v_inst_1_, lean_object* v_n_2_){
_start:
{
lean_object* v_decode_3_; lean_object* v___x_4_; lean_object* v_val_5_; 
v_decode_3_ = lean_ctor_get(v_inst_1_, 1);
lean_inc_ref(v_decode_3_);
lean_dec_ref(v_inst_1_);
v___x_4_ = lean_apply_1(v_decode_3_, v_n_2_);
v_val_5_ = lean_ctor_get(v___x_4_, 0);
lean_inc(v_val_5_);
lean_dec(v___x_4_);
return v_val_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_ofNat(lean_object* v_00_u03b1_6_, lean_object* v_inst_7_, lean_object* v_n_8_){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = lp_mathlib_Denumerable_ofNat___redArg(v_inst_7_, v_n_8_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_eqv___redArg(lean_object* v_inst_10_){
_start:
{
lean_object* v_encode_11_; lean_object* v___x_12_; lean_object* v___x_13_; 
v_encode_11_ = lean_ctor_get(v_inst_10_, 0);
lean_inc_ref(v_encode_11_);
v___x_12_ = lean_alloc_closure((void*)(lp_mathlib_Denumerable_ofNat), 3, 2);
lean_closure_set(v___x_12_, 0, lean_box(0));
lean_closure_set(v___x_12_, 1, v_inst_10_);
v___x_13_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_13_, 0, v_encode_11_);
lean_ctor_set(v___x_13_, 1, v___x_12_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_eqv(lean_object* v_00_u03b1_14_, lean_object* v_inst_15_){
_start:
{
lean_object* v___x_16_; 
v___x_16_ = lp_mathlib_Denumerable_eqv___redArg(v_inst_15_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_mk_x27___redArg___lam__0(lean_object* v_e_17_, lean_object* v___y_18_){
_start:
{
lean_object* v_toFun_19_; lean_object* v___x_20_; 
v_toFun_19_ = lean_ctor_get(v_e_17_, 0);
lean_inc(v_toFun_19_);
lean_dec_ref(v_e_17_);
v___x_20_ = lean_apply_1(v_toFun_19_, v___y_18_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_mk_x27___redArg___lam__1(lean_object* v_val_21_){
_start:
{
lean_object* v___x_22_; 
v___x_22_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_22_, 0, v_val_21_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_mk_x27___redArg___lam__2(lean_object* v___x_23_, lean_object* v___y_24_){
_start:
{
lean_object* v_toFun_25_; lean_object* v___x_26_; 
v_toFun_25_ = lean_ctor_get(v___x_23_, 0);
lean_inc(v_toFun_25_);
lean_dec_ref(v___x_23_);
v___x_26_ = lean_apply_1(v_toFun_25_, v___y_24_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_mk_x27___redArg(lean_object* v_e_28_){
_start:
{
lean_object* v___f_29_; lean_object* v___f_30_; lean_object* v___x_31_; lean_object* v___f_32_; lean_object* v___x_33_; lean_object* v___x_34_; 
lean_inc_ref(v_e_28_);
v___f_29_ = lean_alloc_closure((void*)(lp_mathlib_Denumerable_mk_x27___redArg___lam__0), 2, 1);
lean_closure_set(v___f_29_, 0, v_e_28_);
v___f_30_ = ((lean_object*)(lp_mathlib_Denumerable_mk_x27___redArg___closed__0));
v___x_31_ = lp_mathlib_Equiv_symm___redArg(v_e_28_);
v___f_32_ = lean_alloc_closure((void*)(lp_mathlib_Denumerable_mk_x27___redArg___lam__2), 2, 1);
lean_closure_set(v___f_32_, 0, v___x_31_);
v___x_33_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_33_, 0, lean_box(0));
lean_closure_set(v___x_33_, 1, lean_box(0));
lean_closure_set(v___x_33_, 2, lean_box(0));
lean_closure_set(v___x_33_, 3, v___f_30_);
lean_closure_set(v___x_33_, 4, v___f_32_);
v___x_34_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_34_, 0, v___f_29_);
lean_ctor_set(v___x_34_, 1, v___x_33_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_mk_x27(lean_object* v_00_u03b1_35_, lean_object* v_e_36_){
_start:
{
lean_object* v___x_37_; 
v___x_37_ = lp_mathlib_Denumerable_mk_x27___redArg(v_e_36_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_ofEquiv___redArg(lean_object* v_inst_38_, lean_object* v_e_39_){
_start:
{
lean_object* v___x_40_; 
v___x_40_ = lp_mathlib_Encodable_ofEquiv___redArg(v_inst_38_, v_e_39_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_ofEquiv(lean_object* v_00_u03b1_41_, lean_object* v_00_u03b2_42_, lean_object* v_inst_43_, lean_object* v_e_44_){
_start:
{
lean_object* v___x_45_; 
v___x_45_ = lp_mathlib_Encodable_ofEquiv___redArg(v_inst_43_, v_e_44_);
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_equiv_u2082___redArg(lean_object* v_inst_46_, lean_object* v_inst_47_){
_start:
{
lean_object* v___x_48_; lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; 
v___x_48_ = lp_mathlib_Denumerable_eqv___redArg(v_inst_46_);
v___x_49_ = lp_mathlib_Denumerable_eqv___redArg(v_inst_47_);
v___x_50_ = lp_mathlib_Equiv_symm___redArg(v___x_49_);
v___x_51_ = lp_mathlib_Equiv_trans___redArg(v___x_48_, v___x_50_);
return v___x_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_equiv_u2082(lean_object* v_00_u03b1_52_, lean_object* v_00_u03b2_53_, lean_object* v_inst_54_, lean_object* v_inst_55_){
_start:
{
lean_object* v___x_56_; 
v___x_56_ = lp_mathlib_Denumerable_equiv_u2082___redArg(v_inst_54_, v_inst_55_);
return v___x_56_;
}
}
static lean_object* _init_lp_mathlib_Denumerable_nat(void){
_start:
{
lean_object* v___x_57_; 
v___x_57_ = lp_mathlib_Nat_encodable;
return v___x_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_option___redArg(lean_object* v_inst_58_){
_start:
{
lean_object* v___x_59_; 
v___x_59_ = lp_mathlib_Option_encodable___redArg(v_inst_58_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_option(lean_object* v_00_u03b1_60_, lean_object* v_inst_61_){
_start:
{
lean_object* v___x_62_; 
v___x_62_ = lp_mathlib_Option_encodable___redArg(v_inst_61_);
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Basic_Denumerable_0__Encodable_decodeSum_match__1_splitter___redArg(uint8_t v_x_63_, lean_object* v_x_64_, lean_object* v_h__1_65_, lean_object* v_h__2_66_){
_start:
{
if (v_x_63_ == 0)
{
lean_object* v___x_67_; 
lean_dec(v_h__2_66_);
v___x_67_ = lean_apply_1(v_h__1_65_, v_x_64_);
return v___x_67_;
}
else
{
lean_object* v___x_68_; lean_object* v___x_69_; 
lean_dec(v_h__1_65_);
v___x_68_ = lean_box(v_x_63_);
v___x_69_ = lean_apply_3(v_h__2_66_, v___x_68_, v_x_64_, lean_box(0));
return v___x_69_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Basic_Denumerable_0__Encodable_decodeSum_match__1_splitter___redArg___boxed(lean_object* v_x_70_, lean_object* v_x_71_, lean_object* v_h__1_72_, lean_object* v_h__2_73_){
_start:
{
uint8_t v_x_16__boxed_74_; lean_object* v_res_75_; 
v_x_16__boxed_74_ = lean_unbox(v_x_70_);
v_res_75_ = lp_mathlib___private_Mathlib_Basic_Denumerable_0__Encodable_decodeSum_match__1_splitter___redArg(v_x_16__boxed_74_, v_x_71_, v_h__1_72_, v_h__2_73_);
return v_res_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Basic_Denumerable_0__Encodable_decodeSum_match__1_splitter(lean_object* v_motive_76_, uint8_t v_x_77_, lean_object* v_x_78_, lean_object* v_h__1_79_, lean_object* v_h__2_80_){
_start:
{
if (v_x_77_ == 0)
{
lean_object* v___x_81_; 
lean_dec(v_h__2_80_);
v___x_81_ = lean_apply_1(v_h__1_79_, v_x_78_);
return v___x_81_;
}
else
{
lean_object* v___x_82_; lean_object* v___x_83_; 
lean_dec(v_h__1_79_);
v___x_82_ = lean_box(v_x_77_);
v___x_83_ = lean_apply_3(v_h__2_80_, v___x_82_, v_x_78_, lean_box(0));
return v___x_83_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Basic_Denumerable_0__Encodable_decodeSum_match__1_splitter___boxed(lean_object* v_motive_84_, lean_object* v_x_85_, lean_object* v_x_86_, lean_object* v_h__1_87_, lean_object* v_h__2_88_){
_start:
{
uint8_t v_x_28__boxed_89_; lean_object* v_res_90_; 
v_x_28__boxed_89_ = lean_unbox(v_x_85_);
v_res_90_ = lp_mathlib___private_Mathlib_Basic_Denumerable_0__Encodable_decodeSum_match__1_splitter(v_motive_84_, v_x_28__boxed_89_, v_x_86_, v_h__1_87_, v_h__2_88_);
return v_res_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_sum___redArg(lean_object* v_inst_91_, lean_object* v_inst_92_){
_start:
{
lean_object* v___x_93_; 
v___x_93_ = lp_mathlib_Sum_encodable___redArg(v_inst_91_, v_inst_92_);
return v___x_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_sum(lean_object* v_00_u03b1_94_, lean_object* v_00_u03b2_95_, lean_object* v_inst_96_, lean_object* v_inst_97_){
_start:
{
lean_object* v___x_98_; 
v___x_98_ = lp_mathlib_Sum_encodable___redArg(v_inst_96_, v_inst_97_);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_sigma___redArg___lam__0(lean_object* v_inst_99_, lean_object* v_a_100_){
_start:
{
lean_object* v___x_101_; 
v___x_101_ = lean_apply_1(v_inst_99_, v_a_100_);
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_sigma___redArg(lean_object* v_inst_102_, lean_object* v_inst_103_){
_start:
{
lean_object* v___f_104_; lean_object* v___x_105_; 
v___f_104_ = lean_alloc_closure((void*)(lp_mathlib_Denumerable_sigma___redArg___lam__0), 2, 1);
lean_closure_set(v___f_104_, 0, v_inst_103_);
v___x_105_ = lp_mathlib_Sigma_encodable___redArg(v_inst_102_, v___f_104_);
return v___x_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_sigma(lean_object* v_00_u03b1_106_, lean_object* v_inst_107_, lean_object* v_00_u03b3_108_, lean_object* v_inst_109_){
_start:
{
lean_object* v___x_110_; 
v___x_110_ = lp_mathlib_Denumerable_sigma___redArg(v_inst_107_, v_inst_109_);
return v___x_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_prod___redArg___lam__0(lean_object* v_inst_111_, lean_object* v_a_112_){
_start:
{
lean_inc_ref(v_inst_111_);
return v_inst_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_prod___redArg___lam__0___boxed(lean_object* v_inst_113_, lean_object* v_a_114_){
_start:
{
lean_object* v_res_115_; 
v_res_115_ = lp_mathlib_Denumerable_prod___redArg___lam__0(v_inst_113_, v_a_114_);
lean_dec(v_a_114_);
lean_dec_ref(v_inst_113_);
return v_res_115_;
}
}
static lean_object* _init_lp_mathlib_Denumerable_prod___redArg___closed__0(void){
_start:
{
lean_object* v___x_116_; 
v___x_116_ = lp_mathlib_Equiv_sigmaEquivProd(lean_box(0), lean_box(0));
return v___x_116_;
}
}
static lean_object* _init_lp_mathlib_Denumerable_prod___redArg___closed__1(void){
_start:
{
lean_object* v___x_117_; lean_object* v___x_118_; 
v___x_117_ = lean_obj_once(&lp_mathlib_Denumerable_prod___redArg___closed__0, &lp_mathlib_Denumerable_prod___redArg___closed__0_once, _init_lp_mathlib_Denumerable_prod___redArg___closed__0);
v___x_118_ = lp_mathlib_Equiv_symm___redArg(v___x_117_);
return v___x_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_prod___redArg(lean_object* v_inst_119_, lean_object* v_inst_120_){
_start:
{
lean_object* v___f_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; 
v___f_121_ = lean_alloc_closure((void*)(lp_mathlib_Denumerable_prod___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_121_, 0, v_inst_120_);
v___x_122_ = lp_mathlib_Denumerable_sigma___redArg(v_inst_119_, v___f_121_);
v___x_123_ = lean_obj_once(&lp_mathlib_Denumerable_prod___redArg___closed__1, &lp_mathlib_Denumerable_prod___redArg___closed__1_once, _init_lp_mathlib_Denumerable_prod___redArg___closed__1);
v___x_124_ = lp_mathlib_Encodable_ofEquiv___redArg(v___x_122_, v___x_123_);
return v___x_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_prod(lean_object* v_00_u03b1_125_, lean_object* v_00_u03b2_126_, lean_object* v_inst_127_, lean_object* v_inst_128_){
_start:
{
lean_object* v___x_129_; 
v___x_129_ = lp_mathlib_Denumerable_prod___redArg(v_inst_127_, v_inst_128_);
return v___x_129_;
}
}
static lean_object* _init_lp_mathlib_Denumerable_int(void){
_start:
{
lean_object* v___x_130_; 
v___x_130_ = lp_mathlib_Int_encodable;
return v___x_130_;
}
}
static lean_object* _init_lp_mathlib_Denumerable_pnat(void){
_start:
{
lean_object* v___x_131_; 
v___x_131_ = lp_mathlib_PNat_encodable;
return v___x_131_;
}
}
static lean_object* _init_lp_mathlib_Denumerable_ulift___redArg___closed__0(void){
_start:
{
lean_object* v___x_132_; 
v___x_132_ = lp_mathlib_Equiv_ulift(lean_box(0));
return v___x_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_ulift___redArg(lean_object* v_inst_133_){
_start:
{
lean_object* v___x_134_; lean_object* v___x_135_; 
v___x_134_ = lean_obj_once(&lp_mathlib_Denumerable_ulift___redArg___closed__0, &lp_mathlib_Denumerable_ulift___redArg___closed__0_once, _init_lp_mathlib_Denumerable_ulift___redArg___closed__0);
v___x_135_ = lp_mathlib_Encodable_ofEquiv___redArg(v_inst_133_, v___x_134_);
return v___x_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_ulift(lean_object* v_00_u03b1_136_, lean_object* v_inst_137_){
_start:
{
lean_object* v___x_138_; 
v___x_138_ = lp_mathlib_Denumerable_ulift___redArg(v_inst_137_);
return v___x_138_;
}
}
static lean_object* _init_lp_mathlib_Denumerable_plift___redArg___closed__0(void){
_start:
{
lean_object* v___x_139_; 
v___x_139_ = lp_mathlib_Equiv_plift(lean_box(0));
return v___x_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_plift___redArg(lean_object* v_inst_140_){
_start:
{
lean_object* v___x_141_; lean_object* v___x_142_; 
v___x_141_ = lean_obj_once(&lp_mathlib_Denumerable_plift___redArg___closed__0, &lp_mathlib_Denumerable_plift___redArg___closed__0_once, _init_lp_mathlib_Denumerable_plift___redArg___closed__0);
v___x_142_ = lp_mathlib_Encodable_ofEquiv___redArg(v_inst_140_, v___x_141_);
return v___x_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_plift(lean_object* v_00_u03b1_143_, lean_object* v_inst_144_){
_start:
{
lean_object* v___x_145_; 
v___x_145_ = lp_mathlib_Denumerable_plift___redArg(v_inst_144_);
return v___x_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_pair___redArg(lean_object* v_inst_146_){
_start:
{
lean_object* v___x_147_; lean_object* v___x_148_; 
lean_inc_ref_n(v_inst_146_, 2);
v___x_147_ = lp_mathlib_Denumerable_prod___redArg(v_inst_146_, v_inst_146_);
v___x_148_ = lp_mathlib_Denumerable_equiv_u2082___redArg(v___x_147_, v_inst_146_);
return v___x_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_pair(lean_object* v_00_u03b1_149_, lean_object* v_inst_150_){
_start:
{
lean_object* v___x_151_; 
v___x_151_ = lp_mathlib_Denumerable_pair___redArg(v_inst_150_);
return v___x_151_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Nat_Subtype_succ___redArg___lam__0(lean_object* v_x_152_, lean_object* v_inst_153_, lean_object* v_a_154_){
_start:
{
lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; uint8_t v___x_159_; 
v___x_155_ = lean_nat_add(v_x_152_, v_a_154_);
v___x_156_ = lean_unsigned_to_nat(1u);
v___x_157_ = lean_nat_add(v___x_155_, v___x_156_);
lean_dec(v___x_155_);
v___x_158_ = lean_apply_1(v_inst_153_, v___x_157_);
v___x_159_ = lean_unbox(v___x_158_);
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_Subtype_succ___redArg___lam__0___boxed(lean_object* v_x_160_, lean_object* v_inst_161_, lean_object* v_a_162_){
_start:
{
uint8_t v_res_163_; lean_object* v_r_164_; 
v_res_163_ = lp_mathlib_Nat_Subtype_succ___redArg___lam__0(v_x_160_, v_inst_161_, v_a_162_);
lean_dec(v_a_162_);
lean_dec(v_x_160_);
v_r_164_ = lean_box(v_res_163_);
return v_r_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_Subtype_succ___redArg(lean_object* v_inst_165_, lean_object* v_x_166_){
_start:
{
lean_object* v___f_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; 
lean_inc(v_x_166_);
v___f_167_ = lean_alloc_closure((void*)(lp_mathlib_Nat_Subtype_succ___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_167_, 0, v_x_166_);
lean_closure_set(v___f_167_, 1, v_inst_165_);
v___x_168_ = lp_mathlib_Nat_findX___redArg(v___f_167_);
v___x_169_ = lean_nat_add(v_x_166_, v___x_168_);
lean_dec(v___x_168_);
lean_dec(v_x_166_);
v___x_170_ = lean_unsigned_to_nat(1u);
v___x_171_ = lean_nat_add(v___x_169_, v___x_170_);
lean_dec(v___x_169_);
return v___x_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_Subtype_succ(lean_object* v_s_172_, lean_object* v_inst_173_, lean_object* v_inst_174_, lean_object* v_x_175_){
_start:
{
lean_object* v___x_176_; 
v___x_176_ = lp_mathlib_Nat_Subtype_succ___redArg(v_inst_174_, v_x_175_);
return v___x_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_Subtype_ofNat___redArg(lean_object* v_inst_177_, lean_object* v_x_178_){
_start:
{
lean_object* v_zero_179_; uint8_t v_isZero_180_; 
v_zero_179_ = lean_unsigned_to_nat(0u);
v_isZero_180_ = lean_nat_dec_eq(v_x_178_, v_zero_179_);
if (v_isZero_180_ == 1)
{
lean_object* v___x_181_; 
v___x_181_ = lp_mathlib_Nat_findX___redArg(v_inst_177_);
return v___x_181_;
}
else
{
lean_object* v_one_182_; lean_object* v_n_183_; lean_object* v___x_184_; lean_object* v___x_185_; 
v_one_182_ = lean_unsigned_to_nat(1u);
v_n_183_ = lean_nat_sub(v_x_178_, v_one_182_);
lean_inc_ref(v_inst_177_);
v___x_184_ = lp_mathlib_Nat_Subtype_ofNat___redArg(v_inst_177_, v_n_183_);
lean_dec(v_n_183_);
v___x_185_ = lp_mathlib_Nat_Subtype_succ___redArg(v_inst_177_, v___x_184_);
return v___x_185_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_Subtype_ofNat___redArg___boxed(lean_object* v_inst_186_, lean_object* v_x_187_){
_start:
{
lean_object* v_res_188_; 
v_res_188_ = lp_mathlib_Nat_Subtype_ofNat___redArg(v_inst_186_, v_x_187_);
lean_dec(v_x_187_);
return v_res_188_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_Subtype_ofNat(lean_object* v_s_189_, lean_object* v_inst_190_, lean_object* v_inst_191_, lean_object* v_x_192_){
_start:
{
lean_object* v___x_193_; 
v___x_193_ = lp_mathlib_Nat_Subtype_ofNat___redArg(v_inst_190_, v_x_192_);
return v___x_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_Subtype_ofNat___boxed(lean_object* v_s_194_, lean_object* v_inst_195_, lean_object* v_inst_196_, lean_object* v_x_197_){
_start:
{
lean_object* v_res_198_; 
v_res_198_ = lp_mathlib_Nat_Subtype_ofNat(v_s_194_, v_inst_195_, v_inst_196_, v_x_197_);
lean_dec(v_x_197_);
return v_res_198_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Basic_Denumerable_0__Nat_Subtype_toFunAux___redArg___lam__0(lean_object* v_inst_199_, lean_object* v_x_200_){
_start:
{
lean_object* v___x_201_; uint8_t v___x_202_; 
v___x_201_ = lean_apply_1(v_inst_199_, v_x_200_);
v___x_202_ = lean_unbox(v___x_201_);
return v___x_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Basic_Denumerable_0__Nat_Subtype_toFunAux___redArg___lam__0___boxed(lean_object* v_inst_203_, lean_object* v_x_204_){
_start:
{
uint8_t v_res_205_; lean_object* v_r_206_; 
v_res_205_ = lp_mathlib___private_Mathlib_Basic_Denumerable_0__Nat_Subtype_toFunAux___redArg___lam__0(v_inst_203_, v_x_204_);
v_r_206_ = lean_box(v_res_205_);
return v_r_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Basic_Denumerable_0__Nat_Subtype_toFunAux___redArg(lean_object* v_inst_207_, lean_object* v_x_208_){
_start:
{
lean_object* v___f_209_; lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; 
v___f_209_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Basic_Denumerable_0__Nat_Subtype_toFunAux___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_209_, 0, v_inst_207_);
v___x_210_ = l_List_range(v_x_208_);
v___x_211_ = lean_unsigned_to_nat(0u);
v___x_212_ = l_List_countP_go___redArg(v___f_209_, v___x_210_, v___x_211_);
return v___x_212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Basic_Denumerable_0__Nat_Subtype_toFunAux(lean_object* v_s_213_, lean_object* v_inst_214_, lean_object* v_x_215_){
_start:
{
lean_object* v___x_216_; 
v___x_216_ = lp_mathlib___private_Mathlib_Basic_Denumerable_0__Nat_Subtype_toFunAux___redArg(v_inst_214_, v_x_215_);
return v___x_216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Basic_Denumerable_0__Nat_Subtype_ofNat_match__1_splitter___redArg(lean_object* v_x_217_, lean_object* v_h__1_218_, lean_object* v_h__2_219_){
_start:
{
lean_object* v_zero_220_; uint8_t v_isZero_221_; 
v_zero_220_ = lean_unsigned_to_nat(0u);
v_isZero_221_ = lean_nat_dec_eq(v_x_217_, v_zero_220_);
if (v_isZero_221_ == 1)
{
lean_object* v___x_222_; lean_object* v___x_223_; 
lean_dec(v_h__2_219_);
v___x_222_ = lean_box(0);
v___x_223_ = lean_apply_1(v_h__1_218_, v___x_222_);
return v___x_223_;
}
else
{
lean_object* v_one_224_; lean_object* v_n_225_; lean_object* v___x_226_; 
lean_dec(v_h__1_218_);
v_one_224_ = lean_unsigned_to_nat(1u);
v_n_225_ = lean_nat_sub(v_x_217_, v_one_224_);
v___x_226_ = lean_apply_1(v_h__2_219_, v_n_225_);
return v___x_226_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Basic_Denumerable_0__Nat_Subtype_ofNat_match__1_splitter___redArg___boxed(lean_object* v_x_227_, lean_object* v_h__1_228_, lean_object* v_h__2_229_){
_start:
{
lean_object* v_res_230_; 
v_res_230_ = lp_mathlib___private_Mathlib_Basic_Denumerable_0__Nat_Subtype_ofNat_match__1_splitter___redArg(v_x_227_, v_h__1_228_, v_h__2_229_);
lean_dec(v_x_227_);
return v_res_230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Basic_Denumerable_0__Nat_Subtype_ofNat_match__1_splitter(lean_object* v_motive_231_, lean_object* v_x_232_, lean_object* v_h__1_233_, lean_object* v_h__2_234_){
_start:
{
lean_object* v_zero_235_; uint8_t v_isZero_236_; 
v_zero_235_ = lean_unsigned_to_nat(0u);
v_isZero_236_ = lean_nat_dec_eq(v_x_232_, v_zero_235_);
if (v_isZero_236_ == 1)
{
lean_object* v___x_237_; lean_object* v___x_238_; 
lean_dec(v_h__2_234_);
v___x_237_ = lean_box(0);
v___x_238_ = lean_apply_1(v_h__1_233_, v___x_237_);
return v___x_238_;
}
else
{
lean_object* v_one_239_; lean_object* v_n_240_; lean_object* v___x_241_; 
lean_dec(v_h__1_233_);
v_one_239_ = lean_unsigned_to_nat(1u);
v_n_240_ = lean_nat_sub(v_x_232_, v_one_239_);
v___x_241_ = lean_apply_1(v_h__2_234_, v_n_240_);
return v___x_241_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Basic_Denumerable_0__Nat_Subtype_ofNat_match__1_splitter___boxed(lean_object* v_motive_242_, lean_object* v_x_243_, lean_object* v_h__1_244_, lean_object* v_h__2_245_){
_start:
{
lean_object* v_res_246_; 
v_res_246_ = lp_mathlib___private_Mathlib_Basic_Denumerable_0__Nat_Subtype_ofNat_match__1_splitter(v_motive_242_, v_x_243_, v_h__1_244_, v_h__2_245_);
lean_dec(v_x_243_);
return v_res_246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_Subtype_denumerable___redArg(lean_object* v_inst_247_){
_start:
{
lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; 
v___x_248_ = lp_mathlib_Nat_encodable;
lean_inc_ref(v_inst_247_);
v___x_249_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Basic_Denumerable_0__Nat_Subtype_toFunAux), 3, 2);
lean_closure_set(v___x_249_, 0, lean_box(0));
lean_closure_set(v___x_249_, 1, v_inst_247_);
v___x_250_ = lean_alloc_closure((void*)(lp_mathlib_Nat_Subtype_ofNat___boxed), 4, 3);
lean_closure_set(v___x_250_, 0, lean_box(0));
lean_closure_set(v___x_250_, 1, v_inst_247_);
lean_closure_set(v___x_250_, 2, lean_box(0));
v___x_251_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_251_, 0, v___x_249_);
lean_ctor_set(v___x_251_, 1, v___x_250_);
v___x_252_ = lp_mathlib_Encodable_ofEquiv___redArg(v___x_248_, v___x_251_);
return v___x_252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_Subtype_denumerable(lean_object* v_s_253_, lean_object* v_inst_254_, lean_object* v_inst_255_){
_start:
{
lean_object* v___x_256_; 
v___x_256_ = lp_mathlib_Nat_Subtype_denumerable___redArg(v_inst_254_);
return v___x_256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_ofEncodableOfInfinite___redArg(lean_object* v_inst_257_){
_start:
{
lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; 
lean_inc_ref(v_inst_257_);
v___x_258_ = lean_alloc_closure((void*)(lp_mathlib_Encodable_decidableRangeEncode___boxed), 3, 2);
lean_closure_set(v___x_258_, 0, lean_box(0));
lean_closure_set(v___x_258_, 1, v_inst_257_);
v___x_259_ = lp_mathlib_Nat_Subtype_denumerable___redArg(v___x_258_);
v___x_260_ = lp_mathlib_Encodable_equivRangeEncode___redArg(v_inst_257_);
v___x_261_ = lp_mathlib_Encodable_ofEquiv___redArg(v___x_259_, v___x_260_);
return v___x_261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_ofEncodableOfInfinite(lean_object* v_00_u03b1_262_, lean_object* v_inst_263_, lean_object* v_inst_264_){
_start:
{
lean_object* v___x_265_; 
v___x_265_ = lp_mathlib_Denumerable_ofEncodableOfInfinite___redArg(v_inst_263_);
return v___x_265_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_EquivFin(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_MinMax(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Order_Lemmas(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Encodable_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Basic_Denumerable(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_EquivFin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_MinMax(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Order_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Encodable_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Denumerable_nat = _init_lp_mathlib_Denumerable_nat();
lean_mark_persistent(lp_mathlib_Denumerable_nat);
lp_mathlib_Denumerable_int = _init_lp_mathlib_Denumerable_int();
lean_mark_persistent(lp_mathlib_Denumerable_int);
lp_mathlib_Denumerable_pnat = _init_lp_mathlib_Denumerable_pnat();
lean_mark_persistent(lp_mathlib_Denumerable_pnat);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Basic_Denumerable(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Fintype_EquivFin(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_MinMax(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Order_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Encodable_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Basic_Denumerable(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_EquivFin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_MinMax(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Order_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Encodable_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Denumerable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Basic_Denumerable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Basic_Denumerable(builtin);
}
#ifdef __cplusplus
}
#endif
