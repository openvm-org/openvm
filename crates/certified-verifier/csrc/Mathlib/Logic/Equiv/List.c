// Lean compiler output
// Module: Mathlib.Logic.Equiv.List
// Imports: public import Init public meta import Init public import Mathlib.Basic.Denumerable
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
lean_object* lp_mathlib_Nat_unpair(lean_object*);
lean_object* l_List_get_x3fInternal___redArg(lean_object*, lean_object*);
lean_object* l_instBEqOfDecidableEq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_List_idxOf___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_pair(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_List_replicateTR___redArg(lean_object*, lean_object*);
lean_object* l_List_lengthTR___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Equiv_listEquivOfEquiv_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_listEquivOfEquiv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_listEquivOfEquiv___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_listEquivOfEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_listEquivOfEquiv(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Equiv_listEquivOfEquiv_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encodeList___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encodeList(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Encodable_decodeList___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Encodable_decodeList___redArg___closed__0 = (const lean_object*)&lp_mathlib_Encodable_decodeList___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decodeList___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decodeList___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decodeList(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decodeList___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Equiv_List_0__Encodable_decodeList_match__3_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Equiv_List_0__Encodable_decodeList_match__3_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Equiv_List_0__Encodable_decodeList_match__3_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Equiv_List_0__Encodable_decodeList_match__3_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Equiv_List_0__Encodable_decodeList_match__1_splitter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Equiv_List_0__Encodable_decodeList_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Equiv_List_0__Encodable_decodeList_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Equiv_List_0__Encodable_encodeList_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Equiv_List_0__Encodable_encodeList_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_encodable___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_encodable(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encodableOfList___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encodableOfList___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encodableOfList___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encodableOfList___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encodableOfList(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncEncodable___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncEncodable(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_denumerableList___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_denumerableList(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_listUniqueEquiv___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_listUniqueEquiv___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_List_lengthTR___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Equiv_listUniqueEquiv___redArg___closed__0 = (const lean_object*)&lp_mathlib_Equiv_listUniqueEquiv___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_listUniqueEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_listUniqueEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encodeList___at___00Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encodeList___at___00Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0_spec__0___boxed(lean_object*);
static const lean_ctor_object lp_mathlib_Encodable_decodeList___at___00Denumerable_ofNat___at___00Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0_spec__1_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Encodable_decodeList___at___00Denumerable_ofNat___at___00Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0_spec__1_spec__2___closed__0 = (const lean_object*)&lp_mathlib_Encodable_decodeList___at___00Denumerable_ofNat___at___00Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0_spec__1_spec__2___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decodeList___at___00Denumerable_ofNat___at___00Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0_spec__1_spec__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decodeList___at___00Denumerable_ofNat___at___00Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0_spec__1_spec__2___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_ofNat___at___00Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_ofNat___at___00Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0_spec__1___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Encodable_encodeList___at___00Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0_spec__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0___closed__0_value;
static const lean_closure_object lp_mathlib_Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Denumerable_ofNat___at___00Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0_spec__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0___closed__1 = (const lean_object*)&lp_mathlib_Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0___closed__1_value;
static const lean_ctor_object lp_mathlib_Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0___closed__0_value),((lean_object*)&lp_mathlib_Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0___closed__1_value)}};
static const lean_object* lp_mathlib_Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0___closed__2 = (const lean_object*)&lp_mathlib_Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0 = (const lean_object*)&lp_mathlib_Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_Equiv_listNatEquivNat = (const lean_object*)&lp_mathlib_Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_listEquivSelfOfEquivNat___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_listEquivSelfOfEquivNat(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Equiv_listEquivOfEquiv_spec__0___redArg(lean_object* v_e_1_, lean_object* v_a_2_, lean_object* v_a_3_){
_start:
{
if (lean_obj_tag(v_a_2_) == 0)
{
lean_object* v___x_4_; 
lean_dec_ref(v_e_1_);
v___x_4_ = l_List_reverse___redArg(v_a_3_);
return v___x_4_;
}
else
{
lean_object* v_head_5_; lean_object* v_tail_6_; lean_object* v___x_8_; uint8_t v_isShared_9_; uint8_t v_isSharedCheck_16_; 
v_head_5_ = lean_ctor_get(v_a_2_, 0);
v_tail_6_ = lean_ctor_get(v_a_2_, 1);
v_isSharedCheck_16_ = !lean_is_exclusive(v_a_2_);
if (v_isSharedCheck_16_ == 0)
{
v___x_8_ = v_a_2_;
v_isShared_9_ = v_isSharedCheck_16_;
goto v_resetjp_7_;
}
else
{
lean_inc(v_tail_6_);
lean_inc(v_head_5_);
lean_dec(v_a_2_);
v___x_8_ = lean_box(0);
v_isShared_9_ = v_isSharedCheck_16_;
goto v_resetjp_7_;
}
v_resetjp_7_:
{
lean_object* v_toFun_10_; lean_object* v___x_11_; lean_object* v___x_13_; 
v_toFun_10_ = lean_ctor_get(v_e_1_, 0);
lean_inc(v_toFun_10_);
v___x_11_ = lean_apply_1(v_toFun_10_, v_head_5_);
if (v_isShared_9_ == 0)
{
lean_ctor_set(v___x_8_, 1, v_a_3_);
lean_ctor_set(v___x_8_, 0, v___x_11_);
v___x_13_ = v___x_8_;
goto v_reusejp_12_;
}
else
{
lean_object* v_reuseFailAlloc_15_; 
v_reuseFailAlloc_15_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_15_, 0, v___x_11_);
lean_ctor_set(v_reuseFailAlloc_15_, 1, v_a_3_);
v___x_13_ = v_reuseFailAlloc_15_;
goto v_reusejp_12_;
}
v_reusejp_12_:
{
v_a_2_ = v_tail_6_;
v_a_3_ = v___x_13_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_listEquivOfEquiv___redArg___lam__0(lean_object* v_e_17_, lean_object* v___y_18_){
_start:
{
lean_object* v___x_19_; lean_object* v___x_20_; 
v___x_19_ = lean_box(0);
v___x_20_ = lp_mathlib_List_mapTR_loop___at___00Equiv_listEquivOfEquiv_spec__0___redArg(v_e_17_, v___y_18_, v___x_19_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_listEquivOfEquiv___redArg___lam__1(lean_object* v___x_21_, lean_object* v___y_22_){
_start:
{
lean_object* v___x_23_; lean_object* v___x_24_; 
v___x_23_ = lean_box(0);
v___x_24_ = lp_mathlib_List_mapTR_loop___at___00Equiv_listEquivOfEquiv_spec__0___redArg(v___x_21_, v___y_22_, v___x_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_listEquivOfEquiv___redArg(lean_object* v_e_25_){
_start:
{
lean_object* v___f_26_; lean_object* v___x_27_; lean_object* v___f_28_; lean_object* v___x_29_; 
lean_inc_ref(v_e_25_);
v___f_26_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_listEquivOfEquiv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_26_, 0, v_e_25_);
v___x_27_ = lp_mathlib_Equiv_symm___redArg(v_e_25_);
v___f_28_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_listEquivOfEquiv___redArg___lam__1), 2, 1);
lean_closure_set(v___f_28_, 0, v___x_27_);
v___x_29_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_29_, 0, v___f_26_);
lean_ctor_set(v___x_29_, 1, v___f_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_listEquivOfEquiv(lean_object* v_00_u03b1_30_, lean_object* v_00_u03b2_31_, lean_object* v_e_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = lp_mathlib_Equiv_listEquivOfEquiv___redArg(v_e_32_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Equiv_listEquivOfEquiv_spec__0(lean_object* v_00_u03b1_34_, lean_object* v_00_u03b2_35_, lean_object* v_e_36_, lean_object* v_a_37_, lean_object* v_a_38_){
_start:
{
lean_object* v___x_39_; 
v___x_39_ = lp_mathlib_List_mapTR_loop___at___00Equiv_listEquivOfEquiv_spec__0___redArg(v_e_36_, v_a_37_, v_a_38_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encodeList___redArg(lean_object* v_inst_40_, lean_object* v_x_41_){
_start:
{
if (lean_obj_tag(v_x_41_) == 0)
{
lean_object* v___x_42_; 
lean_dec_ref(v_inst_40_);
v___x_42_ = lean_unsigned_to_nat(0u);
return v___x_42_;
}
else
{
lean_object* v_head_43_; lean_object* v_tail_44_; lean_object* v_encode_45_; lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; lean_object* v___x_49_; lean_object* v___x_50_; 
v_head_43_ = lean_ctor_get(v_x_41_, 0);
lean_inc(v_head_43_);
v_tail_44_ = lean_ctor_get(v_x_41_, 1);
lean_inc(v_tail_44_);
lean_dec_ref_known(v_x_41_, 2);
v_encode_45_ = lean_ctor_get(v_inst_40_, 0);
lean_inc_ref(v_encode_45_);
v___x_46_ = lean_apply_1(v_encode_45_, v_head_43_);
v___x_47_ = lp_mathlib_Encodable_encodeList___redArg(v_inst_40_, v_tail_44_);
v___x_48_ = lp_mathlib_Nat_pair(v___x_46_, v___x_47_);
lean_dec(v___x_47_);
lean_dec(v___x_46_);
v___x_49_ = lean_unsigned_to_nat(1u);
v___x_50_ = lean_nat_add(v___x_48_, v___x_49_);
lean_dec(v___x_48_);
return v___x_50_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encodeList(lean_object* v_00_u03b1_51_, lean_object* v_inst_52_, lean_object* v_x_53_){
_start:
{
lean_object* v___x_54_; 
v___x_54_ = lp_mathlib_Encodable_encodeList___redArg(v_inst_52_, v_x_53_);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decodeList___redArg(lean_object* v_inst_57_, lean_object* v_x_58_){
_start:
{
lean_object* v_zero_59_; uint8_t v_isZero_60_; 
v_zero_59_ = lean_unsigned_to_nat(0u);
v_isZero_60_ = lean_nat_dec_eq(v_x_58_, v_zero_59_);
if (v_isZero_60_ == 1)
{
lean_object* v___x_61_; 
lean_dec_ref(v_inst_57_);
v___x_61_ = ((lean_object*)(lp_mathlib_Encodable_decodeList___redArg___closed__0));
return v___x_61_;
}
else
{
lean_object* v_one_62_; lean_object* v_n_63_; lean_object* v___x_64_; lean_object* v_fst_65_; lean_object* v_snd_66_; lean_object* v___x_68_; uint8_t v_isShared_69_; uint8_t v_isSharedCheck_86_; 
v_one_62_ = lean_unsigned_to_nat(1u);
v_n_63_ = lean_nat_sub(v_x_58_, v_one_62_);
v___x_64_ = lp_mathlib_Nat_unpair(v_n_63_);
lean_dec(v_n_63_);
v_fst_65_ = lean_ctor_get(v___x_64_, 0);
v_snd_66_ = lean_ctor_get(v___x_64_, 1);
v_isSharedCheck_86_ = !lean_is_exclusive(v___x_64_);
if (v_isSharedCheck_86_ == 0)
{
v___x_68_ = v___x_64_;
v_isShared_69_ = v_isSharedCheck_86_;
goto v_resetjp_67_;
}
else
{
lean_inc(v_snd_66_);
lean_inc(v_fst_65_);
lean_dec(v___x_64_);
v___x_68_ = lean_box(0);
v_isShared_69_ = v_isSharedCheck_86_;
goto v_resetjp_67_;
}
v_resetjp_67_:
{
lean_object* v_decode_70_; lean_object* v___x_71_; 
v_decode_70_ = lean_ctor_get(v_inst_57_, 1);
lean_inc_ref(v_decode_70_);
v___x_71_ = lean_apply_1(v_decode_70_, v_fst_65_);
if (lean_obj_tag(v___x_71_) == 0)
{
lean_object* v___x_72_; 
lean_del_object(v___x_68_);
lean_dec(v_snd_66_);
lean_dec_ref(v_inst_57_);
v___x_72_ = lean_box(0);
return v___x_72_;
}
else
{
lean_object* v_val_73_; lean_object* v___x_74_; 
v_val_73_ = lean_ctor_get(v___x_71_, 0);
lean_inc(v_val_73_);
lean_dec_ref_known(v___x_71_, 1);
v___x_74_ = lp_mathlib_Encodable_decodeList___redArg(v_inst_57_, v_snd_66_);
lean_dec(v_snd_66_);
if (lean_obj_tag(v___x_74_) == 0)
{
lean_dec(v_val_73_);
lean_del_object(v___x_68_);
return v___x_74_;
}
else
{
lean_object* v_val_75_; lean_object* v___x_77_; uint8_t v_isShared_78_; uint8_t v_isSharedCheck_85_; 
v_val_75_ = lean_ctor_get(v___x_74_, 0);
v_isSharedCheck_85_ = !lean_is_exclusive(v___x_74_);
if (v_isSharedCheck_85_ == 0)
{
v___x_77_ = v___x_74_;
v_isShared_78_ = v_isSharedCheck_85_;
goto v_resetjp_76_;
}
else
{
lean_inc(v_val_75_);
lean_dec(v___x_74_);
v___x_77_ = lean_box(0);
v_isShared_78_ = v_isSharedCheck_85_;
goto v_resetjp_76_;
}
v_resetjp_76_:
{
lean_object* v___x_80_; 
if (v_isShared_69_ == 0)
{
lean_ctor_set_tag(v___x_68_, 1);
lean_ctor_set(v___x_68_, 1, v_val_75_);
lean_ctor_set(v___x_68_, 0, v_val_73_);
v___x_80_ = v___x_68_;
goto v_reusejp_79_;
}
else
{
lean_object* v_reuseFailAlloc_84_; 
v_reuseFailAlloc_84_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_84_, 0, v_val_73_);
lean_ctor_set(v_reuseFailAlloc_84_, 1, v_val_75_);
v___x_80_ = v_reuseFailAlloc_84_;
goto v_reusejp_79_;
}
v_reusejp_79_:
{
lean_object* v___x_82_; 
if (v_isShared_78_ == 0)
{
lean_ctor_set(v___x_77_, 0, v___x_80_);
v___x_82_ = v___x_77_;
goto v_reusejp_81_;
}
else
{
lean_object* v_reuseFailAlloc_83_; 
v_reuseFailAlloc_83_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_83_, 0, v___x_80_);
v___x_82_ = v_reuseFailAlloc_83_;
goto v_reusejp_81_;
}
v_reusejp_81_:
{
return v___x_82_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decodeList___redArg___boxed(lean_object* v_inst_87_, lean_object* v_x_88_){
_start:
{
lean_object* v_res_89_; 
v_res_89_ = lp_mathlib_Encodable_decodeList___redArg(v_inst_87_, v_x_88_);
lean_dec(v_x_88_);
return v_res_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decodeList(lean_object* v_00_u03b1_90_, lean_object* v_inst_91_, lean_object* v_x_92_){
_start:
{
lean_object* v___x_93_; 
v___x_93_ = lp_mathlib_Encodable_decodeList___redArg(v_inst_91_, v_x_92_);
return v___x_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decodeList___boxed(lean_object* v_00_u03b1_94_, lean_object* v_inst_95_, lean_object* v_x_96_){
_start:
{
lean_object* v_res_97_; 
v_res_97_ = lp_mathlib_Encodable_decodeList(v_00_u03b1_94_, v_inst_95_, v_x_96_);
lean_dec(v_x_96_);
return v_res_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Equiv_List_0__Encodable_decodeList_match__3_splitter___redArg(lean_object* v_x_98_, lean_object* v_h__1_99_, lean_object* v_h__2_100_){
_start:
{
lean_object* v_zero_101_; uint8_t v_isZero_102_; 
v_zero_101_ = lean_unsigned_to_nat(0u);
v_isZero_102_ = lean_nat_dec_eq(v_x_98_, v_zero_101_);
if (v_isZero_102_ == 1)
{
lean_object* v___x_103_; lean_object* v___x_104_; 
lean_dec(v_h__2_100_);
v___x_103_ = lean_box(0);
v___x_104_ = lean_apply_1(v_h__1_99_, v___x_103_);
return v___x_104_;
}
else
{
lean_object* v_one_105_; lean_object* v_n_106_; lean_object* v___x_107_; 
lean_dec(v_h__1_99_);
v_one_105_ = lean_unsigned_to_nat(1u);
v_n_106_ = lean_nat_sub(v_x_98_, v_one_105_);
v___x_107_ = lean_apply_1(v_h__2_100_, v_n_106_);
return v___x_107_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Equiv_List_0__Encodable_decodeList_match__3_splitter___redArg___boxed(lean_object* v_x_108_, lean_object* v_h__1_109_, lean_object* v_h__2_110_){
_start:
{
lean_object* v_res_111_; 
v_res_111_ = lp_mathlib___private_Mathlib_Logic_Equiv_List_0__Encodable_decodeList_match__3_splitter___redArg(v_x_108_, v_h__1_109_, v_h__2_110_);
lean_dec(v_x_108_);
return v_res_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Equiv_List_0__Encodable_decodeList_match__3_splitter(lean_object* v_motive_112_, lean_object* v_x_113_, lean_object* v_h__1_114_, lean_object* v_h__2_115_){
_start:
{
lean_object* v_zero_116_; uint8_t v_isZero_117_; 
v_zero_116_ = lean_unsigned_to_nat(0u);
v_isZero_117_ = lean_nat_dec_eq(v_x_113_, v_zero_116_);
if (v_isZero_117_ == 1)
{
lean_object* v___x_118_; lean_object* v___x_119_; 
lean_dec(v_h__2_115_);
v___x_118_ = lean_box(0);
v___x_119_ = lean_apply_1(v_h__1_114_, v___x_118_);
return v___x_119_;
}
else
{
lean_object* v_one_120_; lean_object* v_n_121_; lean_object* v___x_122_; 
lean_dec(v_h__1_114_);
v_one_120_ = lean_unsigned_to_nat(1u);
v_n_121_ = lean_nat_sub(v_x_113_, v_one_120_);
v___x_122_ = lean_apply_1(v_h__2_115_, v_n_121_);
return v___x_122_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Equiv_List_0__Encodable_decodeList_match__3_splitter___boxed(lean_object* v_motive_123_, lean_object* v_x_124_, lean_object* v_h__1_125_, lean_object* v_h__2_126_){
_start:
{
lean_object* v_res_127_; 
v_res_127_ = lp_mathlib___private_Mathlib_Logic_Equiv_List_0__Encodable_decodeList_match__3_splitter(v_motive_123_, v_x_124_, v_h__1_125_, v_h__2_126_);
lean_dec(v_x_124_);
return v_res_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Equiv_List_0__Encodable_decodeList_match__1_splitter___redArg(lean_object* v_x_128_, lean_object* v_h__1_129_){
_start:
{
lean_object* v_fst_130_; lean_object* v_snd_131_; lean_object* v___x_132_; 
v_fst_130_ = lean_ctor_get(v_x_128_, 0);
lean_inc(v_fst_130_);
v_snd_131_ = lean_ctor_get(v_x_128_, 1);
lean_inc(v_snd_131_);
lean_dec_ref(v_x_128_);
v___x_132_ = lean_apply_3(v_h__1_129_, v_fst_130_, v_snd_131_, lean_box(0));
return v___x_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Equiv_List_0__Encodable_decodeList_match__1_splitter(lean_object* v_v_133_, lean_object* v_motive_134_, lean_object* v_x_135_, lean_object* v_x_136_, lean_object* v_h__1_137_){
_start:
{
lean_object* v_fst_138_; lean_object* v_snd_139_; lean_object* v___x_140_; 
v_fst_138_ = lean_ctor_get(v_x_135_, 0);
lean_inc(v_fst_138_);
v_snd_139_ = lean_ctor_get(v_x_135_, 1);
lean_inc(v_snd_139_);
lean_dec_ref(v_x_135_);
v___x_140_ = lean_apply_3(v_h__1_137_, v_fst_138_, v_snd_139_, lean_box(0));
return v___x_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Equiv_List_0__Encodable_decodeList_match__1_splitter___boxed(lean_object* v_v_141_, lean_object* v_motive_142_, lean_object* v_x_143_, lean_object* v_x_144_, lean_object* v_h__1_145_){
_start:
{
lean_object* v_res_146_; 
v_res_146_ = lp_mathlib___private_Mathlib_Logic_Equiv_List_0__Encodable_decodeList_match__1_splitter(v_v_141_, v_motive_142_, v_x_143_, v_x_144_, v_h__1_145_);
lean_dec(v_v_141_);
return v_res_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Equiv_List_0__Encodable_encodeList_match__1_splitter___redArg(lean_object* v_x_147_, lean_object* v_h__1_148_, lean_object* v_h__2_149_){
_start:
{
if (lean_obj_tag(v_x_147_) == 0)
{
lean_object* v___x_150_; lean_object* v___x_151_; 
lean_dec(v_h__2_149_);
v___x_150_ = lean_box(0);
v___x_151_ = lean_apply_1(v_h__1_148_, v___x_150_);
return v___x_151_;
}
else
{
lean_object* v_head_152_; lean_object* v_tail_153_; lean_object* v___x_154_; 
lean_dec(v_h__1_148_);
v_head_152_ = lean_ctor_get(v_x_147_, 0);
lean_inc(v_head_152_);
v_tail_153_ = lean_ctor_get(v_x_147_, 1);
lean_inc(v_tail_153_);
lean_dec_ref_known(v_x_147_, 2);
v___x_154_ = lean_apply_2(v_h__2_149_, v_head_152_, v_tail_153_);
return v___x_154_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Equiv_List_0__Encodable_encodeList_match__1_splitter(lean_object* v_00_u03b1_155_, lean_object* v_motive_156_, lean_object* v_x_157_, lean_object* v_h__1_158_, lean_object* v_h__2_159_){
_start:
{
if (lean_obj_tag(v_x_157_) == 0)
{
lean_object* v___x_160_; lean_object* v___x_161_; 
lean_dec(v_h__2_159_);
v___x_160_ = lean_box(0);
v___x_161_ = lean_apply_1(v_h__1_158_, v___x_160_);
return v___x_161_;
}
else
{
lean_object* v_head_162_; lean_object* v_tail_163_; lean_object* v___x_164_; 
lean_dec(v_h__1_158_);
v_head_162_ = lean_ctor_get(v_x_157_, 0);
lean_inc(v_head_162_);
v_tail_163_ = lean_ctor_get(v_x_157_, 1);
lean_inc(v_tail_163_);
lean_dec_ref_known(v_x_157_, 2);
v___x_164_ = lean_apply_2(v_h__2_159_, v_head_162_, v_tail_163_);
return v___x_164_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_encodable___redArg(lean_object* v_inst_165_){
_start:
{
lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; 
lean_inc_ref(v_inst_165_);
v___x_166_ = lean_alloc_closure((void*)(lp_mathlib_Encodable_encodeList), 3, 2);
lean_closure_set(v___x_166_, 0, lean_box(0));
lean_closure_set(v___x_166_, 1, v_inst_165_);
v___x_167_ = lean_alloc_closure((void*)(lp_mathlib_Encodable_decodeList___boxed), 3, 2);
lean_closure_set(v___x_167_, 0, lean_box(0));
lean_closure_set(v___x_167_, 1, v_inst_165_);
v___x_168_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_168_, 0, v___x_166_);
lean_ctor_set(v___x_168_, 1, v___x_167_);
return v___x_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_encodable(lean_object* v_00_u03b1_169_, lean_object* v_inst_170_){
_start:
{
lean_object* v___x_171_; 
v___x_171_ = lp_mathlib_List_encodable___redArg(v_inst_170_);
return v___x_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encodableOfList___redArg___lam__0(lean_object* v_l_172_, lean_object* v_x_173_){
_start:
{
lean_object* v___x_174_; 
v___x_174_ = l_List_get_x3fInternal___redArg(v_l_172_, v_x_173_);
return v___x_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encodableOfList___redArg___lam__0___boxed(lean_object* v_l_175_, lean_object* v_x_176_){
_start:
{
lean_object* v_res_177_; 
v_res_177_ = lp_mathlib_Encodable_encodableOfList___redArg___lam__0(v_l_175_, v_x_176_);
lean_dec(v_l_175_);
return v_res_177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encodableOfList___redArg___lam__1(lean_object* v___f_178_, lean_object* v_l_179_, lean_object* v_a_180_){
_start:
{
lean_object* v___x_181_; 
v___x_181_ = l_List_idxOf___redArg(v___f_178_, v_a_180_, v_l_179_);
return v___x_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encodableOfList___redArg(lean_object* v_inst_182_, lean_object* v_l_183_){
_start:
{
lean_object* v___f_184_; lean_object* v___f_185_; lean_object* v___f_186_; lean_object* v___x_187_; 
lean_inc(v_l_183_);
v___f_184_ = lean_alloc_closure((void*)(lp_mathlib_Encodable_encodableOfList___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_184_, 0, v_l_183_);
v___f_185_ = lean_alloc_closure((void*)(l_instBEqOfDecidableEq___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_185_, 0, v_inst_182_);
v___f_186_ = lean_alloc_closure((void*)(lp_mathlib_Encodable_encodableOfList___redArg___lam__1), 3, 2);
lean_closure_set(v___f_186_, 0, v___f_185_);
lean_closure_set(v___f_186_, 1, v_l_183_);
v___x_187_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_187_, 0, v___f_186_);
lean_ctor_set(v___x_187_, 1, v___f_184_);
return v___x_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encodableOfList(lean_object* v_00_u03b1_188_, lean_object* v_inst_189_, lean_object* v_l_190_, lean_object* v_H_191_){
_start:
{
lean_object* v___x_192_; 
v___x_192_ = lp_mathlib_Encodable_encodableOfList___redArg(v_inst_189_, v_l_190_);
return v___x_192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncEncodable___redArg(lean_object* v_inst_193_, lean_object* v_inst_194_){
_start:
{
lean_object* v___x_195_; 
v___x_195_ = lp_mathlib_Encodable_encodableOfList___redArg(v_inst_193_, v_inst_194_);
return v___x_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncEncodable(lean_object* v_00_u03b1_196_, lean_object* v_inst_197_, lean_object* v_inst_198_){
_start:
{
lean_object* v___x_199_; 
v___x_199_ = lp_mathlib_Encodable_encodableOfList___redArg(v_inst_197_, v_inst_198_);
return v___x_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_denumerableList___redArg(lean_object* v_inst_200_){
_start:
{
lean_object* v___x_201_; 
v___x_201_ = lp_mathlib_List_encodable___redArg(v_inst_200_);
return v___x_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_denumerableList(lean_object* v_00_u03b1_202_, lean_object* v_inst_203_){
_start:
{
lean_object* v___x_204_; 
v___x_204_ = lp_mathlib_List_encodable___redArg(v_inst_203_);
return v___x_204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_listUniqueEquiv___redArg___lam__0(lean_object* v_inst_205_, lean_object* v_n_206_){
_start:
{
lean_object* v___x_207_; 
v___x_207_ = l_List_replicateTR___redArg(v_n_206_, v_inst_205_);
return v___x_207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_listUniqueEquiv___redArg(lean_object* v_inst_209_){
_start:
{
lean_object* v___f_210_; lean_object* v___x_211_; lean_object* v___x_212_; 
v___f_210_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_listUniqueEquiv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_210_, 0, v_inst_209_);
v___x_211_ = ((lean_object*)(lp_mathlib_Equiv_listUniqueEquiv___redArg___closed__0));
v___x_212_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_212_, 0, v___x_211_);
lean_ctor_set(v___x_212_, 1, v___f_210_);
return v___x_212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_listUniqueEquiv(lean_object* v_00_u03b1_213_, lean_object* v_inst_214_){
_start:
{
lean_object* v___x_215_; 
v___x_215_ = lp_mathlib_Equiv_listUniqueEquiv___redArg(v_inst_214_);
return v___x_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encodeList___at___00Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0_spec__0(lean_object* v_x_216_){
_start:
{
if (lean_obj_tag(v_x_216_) == 0)
{
lean_object* v___x_217_; 
v___x_217_ = lean_unsigned_to_nat(0u);
return v___x_217_;
}
else
{
lean_object* v_head_218_; lean_object* v_tail_219_; lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; 
v_head_218_ = lean_ctor_get(v_x_216_, 0);
v_tail_219_ = lean_ctor_get(v_x_216_, 1);
v___x_220_ = lp_mathlib_Encodable_encodeList___at___00Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0_spec__0(v_tail_219_);
v___x_221_ = lp_mathlib_Nat_pair(v_head_218_, v___x_220_);
lean_dec(v___x_220_);
v___x_222_ = lean_unsigned_to_nat(1u);
v___x_223_ = lean_nat_add(v___x_221_, v___x_222_);
lean_dec(v___x_221_);
return v___x_223_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encodeList___at___00Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0_spec__0___boxed(lean_object* v_x_224_){
_start:
{
lean_object* v_res_225_; 
v_res_225_ = lp_mathlib_Encodable_encodeList___at___00Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0_spec__0(v_x_224_);
lean_dec(v_x_224_);
return v_res_225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decodeList___at___00Denumerable_ofNat___at___00Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0_spec__1_spec__2(lean_object* v_x_228_){
_start:
{
lean_object* v_zero_229_; uint8_t v_isZero_230_; 
v_zero_229_ = lean_unsigned_to_nat(0u);
v_isZero_230_ = lean_nat_dec_eq(v_x_228_, v_zero_229_);
if (v_isZero_230_ == 1)
{
lean_object* v___x_231_; 
v___x_231_ = ((lean_object*)(lp_mathlib_Encodable_decodeList___at___00Denumerable_ofNat___at___00Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0_spec__1_spec__2___closed__0));
return v___x_231_;
}
else
{
lean_object* v_one_232_; lean_object* v_n_233_; lean_object* v___x_234_; lean_object* v_fst_235_; lean_object* v_snd_236_; lean_object* v___x_238_; uint8_t v_isShared_239_; uint8_t v_isSharedCheck_252_; 
v_one_232_ = lean_unsigned_to_nat(1u);
v_n_233_ = lean_nat_sub(v_x_228_, v_one_232_);
v___x_234_ = lp_mathlib_Nat_unpair(v_n_233_);
lean_dec(v_n_233_);
v_fst_235_ = lean_ctor_get(v___x_234_, 0);
v_snd_236_ = lean_ctor_get(v___x_234_, 1);
v_isSharedCheck_252_ = !lean_is_exclusive(v___x_234_);
if (v_isSharedCheck_252_ == 0)
{
v___x_238_ = v___x_234_;
v_isShared_239_ = v_isSharedCheck_252_;
goto v_resetjp_237_;
}
else
{
lean_inc(v_snd_236_);
lean_inc(v_fst_235_);
lean_dec(v___x_234_);
v___x_238_ = lean_box(0);
v_isShared_239_ = v_isSharedCheck_252_;
goto v_resetjp_237_;
}
v_resetjp_237_:
{
lean_object* v___x_240_; lean_object* v_val_241_; lean_object* v___x_243_; uint8_t v_isShared_244_; uint8_t v_isSharedCheck_251_; 
v___x_240_ = lp_mathlib_Encodable_decodeList___at___00Denumerable_ofNat___at___00Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0_spec__1_spec__2(v_snd_236_);
lean_dec(v_snd_236_);
v_val_241_ = lean_ctor_get(v___x_240_, 0);
v_isSharedCheck_251_ = !lean_is_exclusive(v___x_240_);
if (v_isSharedCheck_251_ == 0)
{
v___x_243_ = v___x_240_;
v_isShared_244_ = v_isSharedCheck_251_;
goto v_resetjp_242_;
}
else
{
lean_inc(v_val_241_);
lean_dec(v___x_240_);
v___x_243_ = lean_box(0);
v_isShared_244_ = v_isSharedCheck_251_;
goto v_resetjp_242_;
}
v_resetjp_242_:
{
lean_object* v___x_246_; 
if (v_isShared_239_ == 0)
{
lean_ctor_set_tag(v___x_238_, 1);
lean_ctor_set(v___x_238_, 1, v_val_241_);
v___x_246_ = v___x_238_;
goto v_reusejp_245_;
}
else
{
lean_object* v_reuseFailAlloc_250_; 
v_reuseFailAlloc_250_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_250_, 0, v_fst_235_);
lean_ctor_set(v_reuseFailAlloc_250_, 1, v_val_241_);
v___x_246_ = v_reuseFailAlloc_250_;
goto v_reusejp_245_;
}
v_reusejp_245_:
{
lean_object* v___x_248_; 
if (v_isShared_244_ == 0)
{
lean_ctor_set(v___x_243_, 0, v___x_246_);
v___x_248_ = v___x_243_;
goto v_reusejp_247_;
}
else
{
lean_object* v_reuseFailAlloc_249_; 
v_reuseFailAlloc_249_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_249_, 0, v___x_246_);
v___x_248_ = v_reuseFailAlloc_249_;
goto v_reusejp_247_;
}
v_reusejp_247_:
{
return v___x_248_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decodeList___at___00Denumerable_ofNat___at___00Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0_spec__1_spec__2___boxed(lean_object* v_x_253_){
_start:
{
lean_object* v_res_254_; 
v_res_254_ = lp_mathlib_Encodable_decodeList___at___00Denumerable_ofNat___at___00Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0_spec__1_spec__2(v_x_253_);
lean_dec(v_x_253_);
return v_res_254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_ofNat___at___00Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0_spec__1(lean_object* v_n_255_){
_start:
{
lean_object* v___x_256_; lean_object* v_val_257_; 
v___x_256_ = lp_mathlib_Encodable_decodeList___at___00Denumerable_ofNat___at___00Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0_spec__1_spec__2(v_n_255_);
v_val_257_ = lean_ctor_get(v___x_256_, 0);
lean_inc(v_val_257_);
lean_dec(v___x_256_);
return v_val_257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_ofNat___at___00Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0_spec__1___boxed(lean_object* v_n_258_){
_start:
{
lean_object* v_res_259_; 
v_res_259_ = lp_mathlib_Denumerable_ofNat___at___00Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0_spec__1(v_n_258_);
lean_dec(v_n_258_);
return v_res_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_listEquivSelfOfEquivNat___redArg(lean_object* v_e_267_){
_start:
{
lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; 
lean_inc_ref(v_e_267_);
v___x_268_ = lp_mathlib_Equiv_listEquivOfEquiv___redArg(v_e_267_);
v___x_269_ = ((lean_object*)(lp_mathlib_Denumerable_eqv___at___00Equiv_listNatEquivNat_spec__0));
v___x_270_ = lp_mathlib_Equiv_trans___redArg(v___x_268_, v___x_269_);
v___x_271_ = lp_mathlib_Equiv_symm___redArg(v_e_267_);
v___x_272_ = lp_mathlib_Equiv_trans___redArg(v___x_270_, v___x_271_);
return v___x_272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_listEquivSelfOfEquivNat(lean_object* v_00_u03b1_273_, lean_object* v_e_274_){
_start:
{
lean_object* v___x_275_; 
v___x_275_ = lp_mathlib_Equiv_listEquivSelfOfEquivNat___redArg(v_e_274_);
return v___x_275_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_Denumerable(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_List(uint8_t builtin) {
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
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Logic_Equiv_List(uint8_t builtin) {
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
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Logic_Equiv_List(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_List(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Logic_Equiv_List(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Logic_Equiv_List(builtin);
}
#ifdef __cplusplus
}
#endif
