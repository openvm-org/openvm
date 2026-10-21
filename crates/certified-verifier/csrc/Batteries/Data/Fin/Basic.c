// Lean compiler output
// Module: Batteries.Data.Fin.Basic
// Imports: public import Init public meta import Init public import Batteries.Data.Nat.Lemmas
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
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Bool_toNat(uint8_t);
lean_object* l_Fin_foldr_loop___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
lean_object* l_Fin_foldl_loop___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_clamp(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_clamp___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_dfoldrM_loop___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_dfoldrM_loop___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_dfoldrM_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_dfoldrM_loop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_dfoldrM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_dfoldrM___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_dfoldrM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_dfoldrM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries_Fin_dfoldr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Fin_dfoldr___redArg___closed__0 = (const lean_object*)&lp_batteries_Fin_dfoldr___redArg___closed__0_value;
static const lean_closure_object lp_batteries_Fin_dfoldr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Fin_dfoldr___redArg___closed__1 = (const lean_object*)&lp_batteries_Fin_dfoldr___redArg___closed__1_value;
static const lean_closure_object lp_batteries_Fin_dfoldr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Fin_dfoldr___redArg___closed__2 = (const lean_object*)&lp_batteries_Fin_dfoldr___redArg___closed__2_value;
static const lean_closure_object lp_batteries_Fin_dfoldr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Fin_dfoldr___redArg___closed__3 = (const lean_object*)&lp_batteries_Fin_dfoldr___redArg___closed__3_value;
static const lean_closure_object lp_batteries_Fin_dfoldr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Fin_dfoldr___redArg___closed__4 = (const lean_object*)&lp_batteries_Fin_dfoldr___redArg___closed__4_value;
static const lean_closure_object lp_batteries_Fin_dfoldr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Fin_dfoldr___redArg___closed__5 = (const lean_object*)&lp_batteries_Fin_dfoldr___redArg___closed__5_value;
static const lean_closure_object lp_batteries_Fin_dfoldr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Fin_dfoldr___redArg___closed__6 = (const lean_object*)&lp_batteries_Fin_dfoldr___redArg___closed__6_value;
static const lean_ctor_object lp_batteries_Fin_dfoldr___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Fin_dfoldr___redArg___closed__0_value),((lean_object*)&lp_batteries_Fin_dfoldr___redArg___closed__1_value)}};
static const lean_object* lp_batteries_Fin_dfoldr___redArg___closed__7 = (const lean_object*)&lp_batteries_Fin_dfoldr___redArg___closed__7_value;
static const lean_ctor_object lp_batteries_Fin_dfoldr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Fin_dfoldr___redArg___closed__7_value),((lean_object*)&lp_batteries_Fin_dfoldr___redArg___closed__2_value),((lean_object*)&lp_batteries_Fin_dfoldr___redArg___closed__3_value),((lean_object*)&lp_batteries_Fin_dfoldr___redArg___closed__4_value),((lean_object*)&lp_batteries_Fin_dfoldr___redArg___closed__5_value)}};
static const lean_object* lp_batteries_Fin_dfoldr___redArg___closed__8 = (const lean_object*)&lp_batteries_Fin_dfoldr___redArg___closed__8_value;
static const lean_ctor_object lp_batteries_Fin_dfoldr___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Fin_dfoldr___redArg___closed__8_value),((lean_object*)&lp_batteries_Fin_dfoldr___redArg___closed__6_value)}};
static const lean_object* lp_batteries_Fin_dfoldr___redArg___closed__9 = (const lean_object*)&lp_batteries_Fin_dfoldr___redArg___closed__9_value;
LEAN_EXPORT lean_object* lp_batteries_Fin_dfoldr___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_dfoldr___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_dfoldr(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_dfoldr___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_dfoldlM_loop___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_dfoldlM_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_dfoldlM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_dfoldlM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_dfoldl___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_dfoldl(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_sum___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_sum___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_sum(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_prod___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_countP___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_countP___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_countP(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_findSome_x3f___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_findSome_x3f___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_findSome_x3f___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_findSome_x3f___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_findSome_x3f(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_findSome_x3f___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_findSomeRev_x3f___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_findSomeRev_x3f___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_findSomeRev_x3f___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_findSomeRev_x3f(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_find_x3f___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_find_x3f___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_find_x3f(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_find_x3f___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_findRev_x3f___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_findRev_x3f___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_findRev_x3f(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_divNat___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_divNat___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_divNat(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_divNat___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_modNat___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_modNat___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_modNat(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_modNat___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_mkDivMod___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_mkDivMod___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_mkDivMod(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_mkDivMod___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Fin_clamp(lean_object* v_n_1_, lean_object* v_m_2_){
_start:
{
uint8_t v___x_3_; 
v___x_3_ = lean_nat_dec_le(v_n_1_, v_m_2_);
if (v___x_3_ == 0)
{
lean_inc(v_m_2_);
return v_m_2_;
}
else
{
lean_inc(v_n_1_);
return v_n_1_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_clamp___boxed(lean_object* v_n_4_, lean_object* v_m_5_){
_start:
{
lean_object* v_res_6_; 
v_res_6_ = lp_batteries_Fin_clamp(v_n_4_, v_m_5_);
lean_dec(v_m_5_);
lean_dec(v_n_4_);
return v_res_6_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_dfoldrM_loop___redArg___boxed(lean_object* v_inst_7_, lean_object* v_f_8_, lean_object* v_i_9_, lean_object* v_x_10_){
_start:
{
lean_object* v_res_11_; 
v_res_11_ = lp_batteries_Fin_dfoldrM_loop___redArg(v_inst_7_, v_f_8_, v_i_9_, v_x_10_);
lean_dec(v_i_9_);
return v_res_11_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_dfoldrM_loop___redArg(lean_object* v_inst_12_, lean_object* v_f_13_, lean_object* v_i_14_, lean_object* v_x_15_){
_start:
{
lean_object* v_toApplicative_16_; lean_object* v_toBind_17_; lean_object* v_toPure_18_; lean_object* v_zero_19_; uint8_t v_isZero_20_; 
v_toApplicative_16_ = lean_ctor_get(v_inst_12_, 0);
v_toBind_17_ = lean_ctor_get(v_inst_12_, 1);
lean_inc(v_toBind_17_);
v_toPure_18_ = lean_ctor_get(v_toApplicative_16_, 1);
v_zero_19_ = lean_unsigned_to_nat(0u);
v_isZero_20_ = lean_nat_dec_eq(v_i_14_, v_zero_19_);
if (v_isZero_20_ == 1)
{
lean_object* v___x_21_; 
lean_inc(v_toPure_18_);
lean_dec(v_toBind_17_);
lean_dec(v_f_13_);
lean_dec_ref(v_inst_12_);
v___x_21_ = lean_apply_2(v_toPure_18_, lean_box(0), v_x_15_);
return v___x_21_;
}
else
{
lean_object* v_one_22_; lean_object* v_n_23_; lean_object* v___x_24_; lean_object* v___x_25_; lean_object* v___x_26_; 
v_one_22_ = lean_unsigned_to_nat(1u);
v_n_23_ = lean_nat_sub(v_i_14_, v_one_22_);
lean_inc(v_f_13_);
lean_inc(v_n_23_);
v___x_24_ = lean_apply_2(v_f_13_, v_n_23_, v_x_15_);
v___x_25_ = lean_alloc_closure((void*)(lp_batteries_Fin_dfoldrM_loop___redArg___boxed), 4, 3);
lean_closure_set(v___x_25_, 0, v_inst_12_);
lean_closure_set(v___x_25_, 1, v_f_13_);
lean_closure_set(v___x_25_, 2, v_n_23_);
v___x_26_ = lean_apply_4(v_toBind_17_, lean_box(0), lean_box(0), v___x_24_, v___x_25_);
return v___x_26_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_dfoldrM_loop(lean_object* v_m_27_, lean_object* v_inst_28_, lean_object* v_n_29_, lean_object* v_00_u03b1_30_, lean_object* v_f_31_, lean_object* v_i_32_, lean_object* v_h_33_, lean_object* v_x_34_){
_start:
{
lean_object* v___x_35_; 
v___x_35_ = lp_batteries_Fin_dfoldrM_loop___redArg(v_inst_28_, v_f_31_, v_i_32_, v_x_34_);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_dfoldrM_loop___boxed(lean_object* v_m_36_, lean_object* v_inst_37_, lean_object* v_n_38_, lean_object* v_00_u03b1_39_, lean_object* v_f_40_, lean_object* v_i_41_, lean_object* v_h_42_, lean_object* v_x_43_){
_start:
{
lean_object* v_res_44_; 
v_res_44_ = lp_batteries_Fin_dfoldrM_loop(v_m_36_, v_inst_37_, v_n_38_, v_00_u03b1_39_, v_f_40_, v_i_41_, v_h_42_, v_x_43_);
lean_dec(v_i_41_);
lean_dec(v_n_38_);
return v_res_44_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_dfoldrM___redArg(lean_object* v_inst_45_, lean_object* v_n_46_, lean_object* v_f_47_, lean_object* v_init_48_){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lp_batteries_Fin_dfoldrM_loop___redArg(v_inst_45_, v_f_47_, v_n_46_, v_init_48_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_dfoldrM___redArg___boxed(lean_object* v_inst_50_, lean_object* v_n_51_, lean_object* v_f_52_, lean_object* v_init_53_){
_start:
{
lean_object* v_res_54_; 
v_res_54_ = lp_batteries_Fin_dfoldrM___redArg(v_inst_50_, v_n_51_, v_f_52_, v_init_53_);
lean_dec(v_n_51_);
return v_res_54_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_dfoldrM(lean_object* v_m_55_, lean_object* v_inst_56_, lean_object* v_n_57_, lean_object* v_00_u03b1_58_, lean_object* v_f_59_, lean_object* v_init_60_){
_start:
{
lean_object* v___x_61_; 
v___x_61_ = lp_batteries_Fin_dfoldrM_loop___redArg(v_inst_56_, v_f_59_, v_n_57_, v_init_60_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_dfoldrM___boxed(lean_object* v_m_62_, lean_object* v_inst_63_, lean_object* v_n_64_, lean_object* v_00_u03b1_65_, lean_object* v_f_66_, lean_object* v_init_67_){
_start:
{
lean_object* v_res_68_; 
v_res_68_ = lp_batteries_Fin_dfoldrM(v_m_62_, v_inst_63_, v_n_64_, v_00_u03b1_65_, v_f_66_, v_init_67_);
lean_dec(v_n_64_);
return v_res_68_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_dfoldr___redArg(lean_object* v_n_88_, lean_object* v_f_89_, lean_object* v_init_90_){
_start:
{
lean_object* v___x_91_; lean_object* v___x_92_; 
v___x_91_ = ((lean_object*)(lp_batteries_Fin_dfoldr___redArg___closed__9));
v___x_92_ = lp_batteries_Fin_dfoldrM_loop___redArg(v___x_91_, v_f_89_, v_n_88_, v_init_90_);
return v___x_92_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_dfoldr___redArg___boxed(lean_object* v_n_93_, lean_object* v_f_94_, lean_object* v_init_95_){
_start:
{
lean_object* v_res_96_; 
v_res_96_ = lp_batteries_Fin_dfoldr___redArg(v_n_93_, v_f_94_, v_init_95_);
lean_dec(v_n_93_);
return v_res_96_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_dfoldr(lean_object* v_n_97_, lean_object* v_00_u03b1_98_, lean_object* v_f_99_, lean_object* v_init_100_){
_start:
{
lean_object* v___x_101_; lean_object* v___x_102_; 
v___x_101_ = ((lean_object*)(lp_batteries_Fin_dfoldr___redArg___closed__9));
v___x_102_ = lp_batteries_Fin_dfoldrM_loop___redArg(v___x_101_, v_f_99_, v_n_97_, v_init_100_);
return v___x_102_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_dfoldr___boxed(lean_object* v_n_103_, lean_object* v_00_u03b1_104_, lean_object* v_f_105_, lean_object* v_init_106_){
_start:
{
lean_object* v_res_107_; 
v_res_107_ = lp_batteries_Fin_dfoldr(v_n_103_, v_00_u03b1_104_, v_f_105_, v_init_106_);
lean_dec(v_n_103_);
return v_res_107_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_dfoldlM_loop___redArg(lean_object* v_inst_108_, lean_object* v_n_109_, lean_object* v_f_110_, lean_object* v_i_111_, lean_object* v_x_112_){
_start:
{
uint8_t v___x_113_; 
v___x_113_ = lean_nat_dec_lt(v_i_111_, v_n_109_);
if (v___x_113_ == 0)
{
lean_object* v_toApplicative_114_; lean_object* v_toPure_115_; lean_object* v___x_116_; 
lean_dec(v_i_111_);
lean_dec(v_f_110_);
lean_dec(v_n_109_);
v_toApplicative_114_ = lean_ctor_get(v_inst_108_, 0);
lean_inc_ref(v_toApplicative_114_);
lean_dec_ref(v_inst_108_);
v_toPure_115_ = lean_ctor_get(v_toApplicative_114_, 1);
lean_inc(v_toPure_115_);
lean_dec_ref(v_toApplicative_114_);
v___x_116_ = lean_apply_2(v_toPure_115_, lean_box(0), v_x_112_);
return v___x_116_;
}
else
{
lean_object* v_toBind_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; 
v_toBind_117_ = lean_ctor_get(v_inst_108_, 1);
lean_inc(v_toBind_117_);
lean_inc(v_f_110_);
lean_inc(v_i_111_);
v___x_118_ = lean_apply_2(v_f_110_, v_i_111_, v_x_112_);
v___x_119_ = lean_unsigned_to_nat(1u);
v___x_120_ = lean_nat_add(v_i_111_, v___x_119_);
lean_dec(v_i_111_);
v___x_121_ = lean_alloc_closure((void*)(lp_batteries_Fin_dfoldlM_loop___redArg), 5, 4);
lean_closure_set(v___x_121_, 0, v_inst_108_);
lean_closure_set(v___x_121_, 1, v_n_109_);
lean_closure_set(v___x_121_, 2, v_f_110_);
lean_closure_set(v___x_121_, 3, v___x_120_);
v___x_122_ = lean_apply_4(v_toBind_117_, lean_box(0), lean_box(0), v___x_118_, v___x_121_);
return v___x_122_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_dfoldlM_loop(lean_object* v_m_123_, lean_object* v_inst_124_, lean_object* v_n_125_, lean_object* v_00_u03b1_126_, lean_object* v_f_127_, lean_object* v_i_128_, lean_object* v_h_129_, lean_object* v_x_130_){
_start:
{
lean_object* v___x_131_; 
v___x_131_ = lp_batteries_Fin_dfoldlM_loop___redArg(v_inst_124_, v_n_125_, v_f_127_, v_i_128_, v_x_130_);
return v___x_131_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_dfoldlM___redArg(lean_object* v_inst_132_, lean_object* v_n_133_, lean_object* v_f_134_, lean_object* v_init_135_){
_start:
{
lean_object* v___x_136_; lean_object* v___x_137_; 
v___x_136_ = lean_unsigned_to_nat(0u);
v___x_137_ = lp_batteries_Fin_dfoldlM_loop___redArg(v_inst_132_, v_n_133_, v_f_134_, v___x_136_, v_init_135_);
return v___x_137_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_dfoldlM(lean_object* v_m_138_, lean_object* v_inst_139_, lean_object* v_n_140_, lean_object* v_00_u03b1_141_, lean_object* v_f_142_, lean_object* v_init_143_){
_start:
{
lean_object* v___x_144_; lean_object* v___x_145_; 
v___x_144_ = lean_unsigned_to_nat(0u);
v___x_145_ = lp_batteries_Fin_dfoldlM_loop___redArg(v_inst_139_, v_n_140_, v_f_142_, v___x_144_, v_init_143_);
return v___x_145_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_dfoldl___redArg(lean_object* v_n_146_, lean_object* v_f_147_, lean_object* v_init_148_){
_start:
{
lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; 
v___x_149_ = ((lean_object*)(lp_batteries_Fin_dfoldr___redArg___closed__9));
v___x_150_ = lean_unsigned_to_nat(0u);
v___x_151_ = lp_batteries_Fin_dfoldlM_loop___redArg(v___x_149_, v_n_146_, v_f_147_, v___x_150_, v_init_148_);
return v___x_151_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_dfoldl(lean_object* v_n_152_, lean_object* v_00_u03b1_153_, lean_object* v_f_154_, lean_object* v_init_155_){
_start:
{
lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; 
v___x_156_ = ((lean_object*)(lp_batteries_Fin_dfoldr___redArg___closed__9));
v___x_157_ = lean_unsigned_to_nat(0u);
v___x_158_ = lp_batteries_Fin_dfoldlM_loop___redArg(v___x_156_, v_n_152_, v_f_154_, v___x_157_, v_init_155_);
return v___x_158_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_sum___redArg___lam__0(lean_object* v_x_159_, lean_object* v_inst_160_, lean_object* v_x1_161_, lean_object* v_x2_162_){
_start:
{
lean_object* v___x_163_; lean_object* v___x_164_; 
v___x_163_ = lean_apply_1(v_x_159_, v_x1_161_);
v___x_164_ = lean_apply_2(v_inst_160_, v___x_163_, v_x2_162_);
return v___x_164_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_sum___redArg(lean_object* v_n_165_, lean_object* v_inst_166_, lean_object* v_inst_167_, lean_object* v_x_168_){
_start:
{
lean_object* v___f_169_; lean_object* v___x_170_; 
v___f_169_ = lean_alloc_closure((void*)(lp_batteries_Fin_sum___redArg___lam__0), 4, 2);
lean_closure_set(v___f_169_, 0, v_x_168_);
lean_closure_set(v___f_169_, 1, v_inst_167_);
v___x_170_ = l_Fin_foldr_loop___redArg(v___f_169_, v_n_165_, v_inst_166_);
return v___x_170_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_sum(lean_object* v_00_u03b1_171_, lean_object* v_n_172_, lean_object* v_inst_173_, lean_object* v_inst_174_, lean_object* v_x_175_){
_start:
{
lean_object* v___f_176_; lean_object* v___x_177_; 
v___f_176_ = lean_alloc_closure((void*)(lp_batteries_Fin_sum___redArg___lam__0), 4, 2);
lean_closure_set(v___f_176_, 0, v_x_175_);
lean_closure_set(v___f_176_, 1, v_inst_174_);
v___x_177_ = l_Fin_foldr_loop___redArg(v___f_176_, v_n_172_, v_inst_173_);
return v___x_177_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_prod___redArg(lean_object* v_n_178_, lean_object* v_inst_179_, lean_object* v_inst_180_, lean_object* v_x_181_){
_start:
{
lean_object* v___f_182_; lean_object* v___x_183_; 
v___f_182_ = lean_alloc_closure((void*)(lp_batteries_Fin_sum___redArg___lam__0), 4, 2);
lean_closure_set(v___f_182_, 0, v_x_181_);
lean_closure_set(v___f_182_, 1, v_inst_180_);
v___x_183_ = l_Fin_foldr_loop___redArg(v___f_182_, v_n_178_, v_inst_179_);
return v___x_183_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_prod(lean_object* v_00_u03b1_184_, lean_object* v_n_185_, lean_object* v_inst_186_, lean_object* v_inst_187_, lean_object* v_x_188_){
_start:
{
lean_object* v___f_189_; lean_object* v___x_190_; 
v___f_189_ = lean_alloc_closure((void*)(lp_batteries_Fin_sum___redArg___lam__0), 4, 2);
lean_closure_set(v___f_189_, 0, v_x_188_);
lean_closure_set(v___f_189_, 1, v_inst_187_);
v___x_190_ = l_Fin_foldr_loop___redArg(v___f_189_, v_n_185_, v_inst_186_);
return v___x_190_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_countP___lam__0(lean_object* v_p_191_, lean_object* v_x1_192_, lean_object* v_x2_193_){
_start:
{
lean_object* v___x_194_; uint8_t v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; 
v___x_194_ = lean_apply_1(v_p_191_, v_x1_192_);
v___x_195_ = lean_unbox(v___x_194_);
v___x_196_ = l_Bool_toNat(v___x_195_);
v___x_197_ = lean_nat_add(v___x_196_, v_x2_193_);
lean_dec(v___x_196_);
return v___x_197_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_countP___lam__0___boxed(lean_object* v_p_198_, lean_object* v_x1_199_, lean_object* v_x2_200_){
_start:
{
lean_object* v_res_201_; 
v_res_201_ = lp_batteries_Fin_countP___lam__0(v_p_198_, v_x1_199_, v_x2_200_);
lean_dec(v_x2_200_);
return v_res_201_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_countP(lean_object* v_n_202_, lean_object* v_p_203_){
_start:
{
lean_object* v___f_204_; lean_object* v___x_205_; lean_object* v___x_206_; 
v___f_204_ = lean_alloc_closure((void*)(lp_batteries_Fin_countP___lam__0___boxed), 3, 1);
lean_closure_set(v___f_204_, 0, v_p_203_);
v___x_205_ = lean_unsigned_to_nat(0u);
v___x_206_ = l_Fin_foldr_loop___redArg(v___f_204_, v_n_202_, v___x_205_);
return v___x_206_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_findSome_x3f___redArg___lam__0(lean_object* v_f_207_, lean_object* v_r_208_, lean_object* v_i_209_){
_start:
{
if (lean_obj_tag(v_r_208_) == 0)
{
lean_object* v___x_210_; 
v___x_210_ = lean_apply_1(v_f_207_, v_i_209_);
return v___x_210_;
}
else
{
lean_dec(v_i_209_);
lean_dec_ref(v_f_207_);
lean_inc_ref(v_r_208_);
return v_r_208_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_findSome_x3f___redArg___lam__0___boxed(lean_object* v_f_211_, lean_object* v_r_212_, lean_object* v_i_213_){
_start:
{
lean_object* v_res_214_; 
v_res_214_ = lp_batteries_Fin_findSome_x3f___redArg___lam__0(v_f_211_, v_r_212_, v_i_213_);
lean_dec(v_r_212_);
return v_res_214_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_findSome_x3f___redArg(lean_object* v_n_215_, lean_object* v_f_216_){
_start:
{
lean_object* v___f_217_; lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; 
v___f_217_ = lean_alloc_closure((void*)(lp_batteries_Fin_findSome_x3f___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_217_, 0, v_f_216_);
v___x_218_ = lean_box(0);
v___x_219_ = lean_unsigned_to_nat(0u);
v___x_220_ = l_Fin_foldl_loop___redArg(v_n_215_, v___f_217_, v___x_218_, v___x_219_);
return v___x_220_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_findSome_x3f___redArg___boxed(lean_object* v_n_221_, lean_object* v_f_222_){
_start:
{
lean_object* v_res_223_; 
v_res_223_ = lp_batteries_Fin_findSome_x3f___redArg(v_n_221_, v_f_222_);
lean_dec(v_n_221_);
return v_res_223_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_findSome_x3f(lean_object* v_n_224_, lean_object* v_00_u03b1_225_, lean_object* v_f_226_){
_start:
{
lean_object* v___f_227_; lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; 
v___f_227_ = lean_alloc_closure((void*)(lp_batteries_Fin_findSome_x3f___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_227_, 0, v_f_226_);
v___x_228_ = lean_box(0);
v___x_229_ = lean_unsigned_to_nat(0u);
v___x_230_ = l_Fin_foldl_loop___redArg(v_n_224_, v___f_227_, v___x_228_, v___x_229_);
return v___x_230_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_findSome_x3f___boxed(lean_object* v_n_231_, lean_object* v_00_u03b1_232_, lean_object* v_f_233_){
_start:
{
lean_object* v_res_234_; 
v_res_234_ = lp_batteries_Fin_findSome_x3f(v_n_231_, v_00_u03b1_232_, v_f_233_);
lean_dec(v_n_231_);
return v_res_234_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_findSomeRev_x3f___redArg___lam__0(lean_object* v_n_235_, lean_object* v_f_236_, lean_object* v_r_237_, lean_object* v_i_238_){
_start:
{
if (lean_obj_tag(v_r_237_) == 0)
{
lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; 
v___x_239_ = lean_unsigned_to_nat(1u);
v___x_240_ = lean_nat_add(v_i_238_, v___x_239_);
v___x_241_ = lean_nat_sub(v_n_235_, v___x_240_);
lean_dec(v___x_240_);
v___x_242_ = lean_apply_1(v_f_236_, v___x_241_);
return v___x_242_;
}
else
{
lean_dec_ref(v_f_236_);
lean_inc_ref(v_r_237_);
return v_r_237_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_findSomeRev_x3f___redArg___lam__0___boxed(lean_object* v_n_243_, lean_object* v_f_244_, lean_object* v_r_245_, lean_object* v_i_246_){
_start:
{
lean_object* v_res_247_; 
v_res_247_ = lp_batteries_Fin_findSomeRev_x3f___redArg___lam__0(v_n_243_, v_f_244_, v_r_245_, v_i_246_);
lean_dec(v_i_246_);
lean_dec(v_r_245_);
lean_dec(v_n_243_);
return v_res_247_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_findSomeRev_x3f___redArg(lean_object* v_n_248_, lean_object* v_f_249_){
_start:
{
lean_object* v___f_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; 
lean_inc(v_n_248_);
v___f_250_ = lean_alloc_closure((void*)(lp_batteries_Fin_findSomeRev_x3f___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_250_, 0, v_n_248_);
lean_closure_set(v___f_250_, 1, v_f_249_);
v___x_251_ = lean_box(0);
v___x_252_ = lean_unsigned_to_nat(0u);
v___x_253_ = l_Fin_foldl_loop___redArg(v_n_248_, v___f_250_, v___x_251_, v___x_252_);
lean_dec(v_n_248_);
return v___x_253_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_findSomeRev_x3f(lean_object* v_n_254_, lean_object* v_00_u03b1_255_, lean_object* v_f_256_){
_start:
{
lean_object* v___f_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; 
lean_inc(v_n_254_);
v___f_257_ = lean_alloc_closure((void*)(lp_batteries_Fin_findSomeRev_x3f___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_257_, 0, v_n_254_);
lean_closure_set(v___f_257_, 1, v_f_256_);
v___x_258_ = lean_box(0);
v___x_259_ = lean_unsigned_to_nat(0u);
v___x_260_ = l_Fin_foldl_loop___redArg(v_n_254_, v___f_257_, v___x_258_, v___x_259_);
lean_dec(v_n_254_);
return v___x_260_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_find_x3f___lam__0(lean_object* v_p_261_, lean_object* v_r_262_, lean_object* v_i_263_){
_start:
{
if (lean_obj_tag(v_r_262_) == 0)
{
lean_object* v___x_264_; uint8_t v___x_265_; 
lean_inc(v_i_263_);
v___x_264_ = lean_apply_1(v_p_261_, v_i_263_);
v___x_265_ = lean_unbox(v___x_264_);
if (v___x_265_ == 0)
{
lean_dec(v_i_263_);
return v_r_262_;
}
else
{
lean_object* v___x_266_; 
v___x_266_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_266_, 0, v_i_263_);
return v___x_266_;
}
}
else
{
lean_dec(v_i_263_);
lean_dec_ref(v_p_261_);
lean_inc_ref(v_r_262_);
return v_r_262_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_find_x3f___lam__0___boxed(lean_object* v_p_267_, lean_object* v_r_268_, lean_object* v_i_269_){
_start:
{
lean_object* v_res_270_; 
v_res_270_ = lp_batteries_Fin_find_x3f___lam__0(v_p_267_, v_r_268_, v_i_269_);
lean_dec(v_r_268_);
return v_res_270_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_find_x3f(lean_object* v_n_271_, lean_object* v_p_272_){
_start:
{
lean_object* v___f_273_; lean_object* v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; 
v___f_273_ = lean_alloc_closure((void*)(lp_batteries_Fin_find_x3f___lam__0___boxed), 3, 1);
lean_closure_set(v___f_273_, 0, v_p_272_);
v___x_274_ = lean_box(0);
v___x_275_ = lean_unsigned_to_nat(0u);
v___x_276_ = l_Fin_foldl_loop___redArg(v_n_271_, v___f_273_, v___x_274_, v___x_275_);
return v___x_276_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_find_x3f___boxed(lean_object* v_n_277_, lean_object* v_p_278_){
_start:
{
lean_object* v_res_279_; 
v_res_279_ = lp_batteries_Fin_find_x3f(v_n_277_, v_p_278_);
lean_dec(v_n_277_);
return v_res_279_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_findRev_x3f___lam__0(lean_object* v_n_280_, lean_object* v_p_281_, lean_object* v_r_282_, lean_object* v_i_283_){
_start:
{
if (lean_obj_tag(v_r_282_) == 0)
{
lean_object* v___x_284_; lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; uint8_t v___x_288_; 
v___x_284_ = lean_unsigned_to_nat(1u);
v___x_285_ = lean_nat_add(v_i_283_, v___x_284_);
v___x_286_ = lean_nat_sub(v_n_280_, v___x_285_);
lean_dec(v___x_285_);
lean_inc(v___x_286_);
v___x_287_ = lean_apply_1(v_p_281_, v___x_286_);
v___x_288_ = lean_unbox(v___x_287_);
if (v___x_288_ == 0)
{
lean_dec(v___x_286_);
return v_r_282_;
}
else
{
lean_object* v___x_289_; 
v___x_289_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_289_, 0, v___x_286_);
return v___x_289_;
}
}
else
{
lean_dec_ref(v_p_281_);
lean_inc_ref(v_r_282_);
return v_r_282_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_findRev_x3f___lam__0___boxed(lean_object* v_n_290_, lean_object* v_p_291_, lean_object* v_r_292_, lean_object* v_i_293_){
_start:
{
lean_object* v_res_294_; 
v_res_294_ = lp_batteries_Fin_findRev_x3f___lam__0(v_n_290_, v_p_291_, v_r_292_, v_i_293_);
lean_dec(v_i_293_);
lean_dec(v_r_292_);
lean_dec(v_n_290_);
return v_res_294_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_findRev_x3f(lean_object* v_n_295_, lean_object* v_p_296_){
_start:
{
lean_object* v___f_297_; lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; 
lean_inc(v_n_295_);
v___f_297_ = lean_alloc_closure((void*)(lp_batteries_Fin_findRev_x3f___lam__0___boxed), 4, 2);
lean_closure_set(v___f_297_, 0, v_n_295_);
lean_closure_set(v___f_297_, 1, v_p_296_);
v___x_298_ = lean_box(0);
v___x_299_ = lean_unsigned_to_nat(0u);
v___x_300_ = l_Fin_foldl_loop___redArg(v_n_295_, v___f_297_, v___x_298_, v___x_299_);
lean_dec(v_n_295_);
return v___x_300_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_divNat___redArg(lean_object* v_n_301_, lean_object* v_i_302_){
_start:
{
lean_object* v___x_303_; 
v___x_303_ = lean_nat_div(v_i_302_, v_n_301_);
return v___x_303_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_divNat___redArg___boxed(lean_object* v_n_304_, lean_object* v_i_305_){
_start:
{
lean_object* v_res_306_; 
v_res_306_ = lp_batteries_Fin_divNat___redArg(v_n_304_, v_i_305_);
lean_dec(v_i_305_);
lean_dec(v_n_304_);
return v_res_306_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_divNat(lean_object* v_m_307_, lean_object* v_n_308_, lean_object* v_i_309_){
_start:
{
lean_object* v___x_310_; 
v___x_310_ = lean_nat_div(v_i_309_, v_n_308_);
return v___x_310_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_divNat___boxed(lean_object* v_m_311_, lean_object* v_n_312_, lean_object* v_i_313_){
_start:
{
lean_object* v_res_314_; 
v_res_314_ = lp_batteries_Fin_divNat(v_m_311_, v_n_312_, v_i_313_);
lean_dec(v_i_313_);
lean_dec(v_n_312_);
lean_dec(v_m_311_);
return v_res_314_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_modNat___redArg(lean_object* v_n_315_, lean_object* v_i_316_){
_start:
{
lean_object* v___x_317_; 
v___x_317_ = lean_nat_mod(v_i_316_, v_n_315_);
return v___x_317_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_modNat___redArg___boxed(lean_object* v_n_318_, lean_object* v_i_319_){
_start:
{
lean_object* v_res_320_; 
v_res_320_ = lp_batteries_Fin_modNat___redArg(v_n_318_, v_i_319_);
lean_dec(v_i_319_);
lean_dec(v_n_318_);
return v_res_320_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_modNat(lean_object* v_m_321_, lean_object* v_n_322_, lean_object* v_i_323_){
_start:
{
lean_object* v___x_324_; 
v___x_324_ = lean_nat_mod(v_i_323_, v_n_322_);
return v___x_324_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_modNat___boxed(lean_object* v_m_325_, lean_object* v_n_326_, lean_object* v_i_327_){
_start:
{
lean_object* v_res_328_; 
v_res_328_ = lp_batteries_Fin_modNat(v_m_325_, v_n_326_, v_i_327_);
lean_dec(v_i_327_);
lean_dec(v_n_326_);
lean_dec(v_m_325_);
return v_res_328_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_mkDivMod___redArg(lean_object* v_n_329_, lean_object* v_i_330_, lean_object* v_j_331_){
_start:
{
lean_object* v___x_332_; lean_object* v___x_333_; 
v___x_332_ = lean_nat_mul(v_n_329_, v_i_330_);
v___x_333_ = lean_nat_add(v___x_332_, v_j_331_);
lean_dec(v___x_332_);
return v___x_333_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_mkDivMod___redArg___boxed(lean_object* v_n_334_, lean_object* v_i_335_, lean_object* v_j_336_){
_start:
{
lean_object* v_res_337_; 
v_res_337_ = lp_batteries_Fin_mkDivMod___redArg(v_n_334_, v_i_335_, v_j_336_);
lean_dec(v_j_336_);
lean_dec(v_i_335_);
lean_dec(v_n_334_);
return v_res_337_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_mkDivMod(lean_object* v_m_338_, lean_object* v_n_339_, lean_object* v_i_340_, lean_object* v_j_341_){
_start:
{
lean_object* v___x_342_; 
v___x_342_ = lp_batteries_Fin_mkDivMod___redArg(v_n_339_, v_i_340_, v_j_341_);
return v___x_342_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Fin_mkDivMod___boxed(lean_object* v_m_343_, lean_object* v_n_344_, lean_object* v_i_345_, lean_object* v_j_346_){
_start:
{
lean_object* v_res_347_; 
v_res_347_ = lp_batteries_Fin_mkDivMod(v_m_343_, v_n_344_, v_i_345_, v_j_346_);
lean_dec(v_j_346_);
lean_dec(v_i_345_);
lean_dec(v_n_344_);
lean_dec(v_m_343_);
return v_res_347_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Data_Nat_Lemmas(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Data_Fin_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_Nat_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Data_Fin_Basic(uint8_t builtin) {
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
lean_object* initialize_batteries_Batteries_Data_Nat_Lemmas(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Data_Fin_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_Nat_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Data_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Data_Fin_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
