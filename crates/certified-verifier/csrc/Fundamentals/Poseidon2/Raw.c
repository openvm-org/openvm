// Lean compiler output
// Module: Fundamentals.Poseidon2.Raw
// Imports: public import Init public meta import Init public import Init public import Fundamentals.BabyBear.Raw public import Fundamentals.BabyBearExt4.Raw public import Fundamentals.Poseidon2.Generic public import Fundamentals.Poseidon2.Sponge
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
lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_BabyBear_FBB_Raw_inv(lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_BabyBear_FBB_Raw_sub___boxed(lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_BabyBear_FBB_Raw_pow(lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_BabyBear_FBB_Raw_mul___boxed(lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_BabyBear_FBB_Raw_add___boxed(lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_BabyBear_FBB_Raw_ofNat___boxed(lean_object*);
lean_object* l_List_get_x3fInternal___redArg(lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_BabyBearExt4_Raw_ofCoeffs(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_BabyBearExt4_Raw_extToWords(lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_compressDigest___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Generic_permute___redArg(lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_hashSlice___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_WIDTH;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_RATE;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2drbr_x2dformal_Fundamentals_BabyBear_FBB_Raw_ofNat___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__0_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2drbr_x2dformal_Fundamentals_BabyBear_FBB_Raw_add___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__1_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2drbr_x2dformal_Fundamentals_BabyBear_FBB_Raw_mul___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__2 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__2_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2drbr_x2dformal_Fundamentals_BabyBear_FBB_Raw_pow, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__3 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__3_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*6 + 0, .m_other = 6, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__0_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__1_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__2_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__3_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__4 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__4_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2drbr_x2dformal_Fundamentals_BabyBear_FBB_Raw_sub___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__5 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__5_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__4_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__5_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__6 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__6_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2drbr_x2dformal_Fundamentals_BabyBear_FBB_Raw_inv, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__7 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__7_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__6_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__7_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__8 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__8_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__8_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___lam__2___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___closed__0_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___lam__1___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__8_value)} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___closed__1_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___lam__2___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___closed__1_value)} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___closed__2 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___closed__2_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2drbr_x2dformal_Fundamentals_BabyBearExt4_Raw_extToWords, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___closed__3 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___closed__3_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO___closed__8_value),((lean_object*)(((size_t)(4) << 1) | 1)),((lean_object*)&lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___closed__3_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___closed__2_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___closed__0_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___closed__4 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___closed__4_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___closed__4_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_permute(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_hashSlice(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_compressDigest(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_List_beq___at___00Fundamentals_Poseidon2_Sponge_digestEq___at___00Fundamentals_Poseidon2_Sponge_merkleVerify___at___00Fundamentals_Poseidon2_Raw_merkleVerify_spec__0_spec__1_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_beq___at___00Fundamentals_Poseidon2_Sponge_digestEq___at___00Fundamentals_Poseidon2_Sponge_merkleVerify___at___00Fundamentals_Poseidon2_Raw_merkleVerify_spec__0_spec__1_spec__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_digestEq___at___00Fundamentals_Poseidon2_Sponge_merkleVerify___at___00Fundamentals_Poseidon2_Raw_merkleVerify_spec__0_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_digestEq___at___00Fundamentals_Poseidon2_Sponge_merkleVerify___at___00Fundamentals_Poseidon2_Raw_merkleVerify_spec__0_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Fundamentals_Poseidon2_Sponge_merkleVerify___at___00Fundamentals_Poseidon2_Raw_merkleVerify_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_merkleVerify___at___00Fundamentals_Poseidon2_Raw_merkleVerify_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_merkleVerify___at___00Fundamentals_Poseidon2_Raw_merkleVerify_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_merkleVerify(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_merkleVerify___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_merkleVerify___at___00Fundamentals_Poseidon2_Raw_merkleVerify_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_merkleVerify___at___00Fundamentals_Poseidon2_Raw_merkleVerify_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Fundamentals_Poseidon2_Sponge_merkleVerify___at___00Fundamentals_Poseidon2_Raw_merkleVerify_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_WIDTH(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lean_unsigned_to_nat(16u);
return v___x_1_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_RATE(void){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(8u);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___lam__0(lean_object* v_w_23_){
_start:
{
lean_inc(v_w_23_);
return v_w_23_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___lam__0___boxed(lean_object* v_w_24_){
_start:
{
lean_object* v_res_25_; 
v_res_25_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___lam__0(v_w_24_);
lean_dec(v_w_24_);
return v_res_25_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___lam__1(lean_object* v___x_26_, lean_object* v_00___27_){
_start:
{
lean_object* v_toRingOps_28_; lean_object* v_toSemiringOps_29_; lean_object* v_zero_30_; 
v_toRingOps_28_ = lean_ctor_get(v___x_26_, 0);
v_toSemiringOps_29_ = lean_ctor_get(v_toRingOps_28_, 0);
v_zero_30_ = lean_ctor_get(v_toSemiringOps_29_, 0);
lean_inc(v_zero_30_);
return v_zero_30_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___lam__1___boxed(lean_object* v___x_31_, lean_object* v_00___32_){
_start:
{
lean_object* v_res_33_; 
v_res_33_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___lam__1(v___x_31_, v_00___32_);
lean_dec_ref(v___x_31_);
return v_res_33_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___lam__2(lean_object* v___f_34_, lean_object* v_ws_35_){
_start:
{
lean_object* v___y_37_; lean_object* v___y_38_; lean_object* v___y_39_; lean_object* v___y_48_; lean_object* v___y_49_; lean_object* v___y_56_; lean_object* v___x_62_; lean_object* v___x_63_; 
v___x_62_ = lean_unsigned_to_nat(0u);
v___x_63_ = l_List_get_x3fInternal___redArg(v_ws_35_, v___x_62_);
if (lean_obj_tag(v___x_63_) == 0)
{
lean_object* v___x_64_; lean_object* v___x_65_; 
v___x_64_ = lean_box(0);
lean_inc_ref(v___f_34_);
v___x_65_ = lean_apply_1(v___f_34_, v___x_64_);
v___y_56_ = v___x_65_;
goto v___jp_55_;
}
else
{
lean_object* v_val_66_; 
v_val_66_ = lean_ctor_get(v___x_63_, 0);
lean_inc(v_val_66_);
lean_dec_ref_known(v___x_63_, 1);
v___y_56_ = v_val_66_;
goto v___jp_55_;
}
v___jp_36_:
{
lean_object* v___x_40_; lean_object* v___x_41_; 
v___x_40_ = lean_unsigned_to_nat(3u);
v___x_41_ = l_List_get_x3fInternal___redArg(v_ws_35_, v___x_40_);
if (lean_obj_tag(v___x_41_) == 0)
{
lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___x_44_; 
v___x_42_ = lean_box(0);
v___x_43_ = lean_apply_1(v___f_34_, v___x_42_);
v___x_44_ = lp_swirl_x2drbr_x2dformal_Fundamentals_BabyBearExt4_Raw_ofCoeffs(v___y_38_, v___y_37_, v___y_39_, v___x_43_);
return v___x_44_;
}
else
{
lean_object* v_val_45_; lean_object* v___x_46_; 
lean_dec_ref(v___f_34_);
v_val_45_ = lean_ctor_get(v___x_41_, 0);
lean_inc(v_val_45_);
lean_dec_ref_known(v___x_41_, 1);
v___x_46_ = lp_swirl_x2drbr_x2dformal_Fundamentals_BabyBearExt4_Raw_ofCoeffs(v___y_38_, v___y_37_, v___y_39_, v_val_45_);
return v___x_46_;
}
}
v___jp_47_:
{
lean_object* v___x_50_; lean_object* v___x_51_; 
v___x_50_ = lean_unsigned_to_nat(2u);
v___x_51_ = l_List_get_x3fInternal___redArg(v_ws_35_, v___x_50_);
if (lean_obj_tag(v___x_51_) == 0)
{
lean_object* v___x_52_; lean_object* v___x_53_; 
v___x_52_ = lean_box(0);
lean_inc_ref(v___f_34_);
v___x_53_ = lean_apply_1(v___f_34_, v___x_52_);
v___y_37_ = v___y_49_;
v___y_38_ = v___y_48_;
v___y_39_ = v___x_53_;
goto v___jp_36_;
}
else
{
lean_object* v_val_54_; 
v_val_54_ = lean_ctor_get(v___x_51_, 0);
lean_inc(v_val_54_);
lean_dec_ref_known(v___x_51_, 1);
v___y_37_ = v___y_49_;
v___y_38_ = v___y_48_;
v___y_39_ = v_val_54_;
goto v___jp_36_;
}
}
v___jp_55_:
{
lean_object* v___x_57_; lean_object* v___x_58_; 
v___x_57_ = lean_unsigned_to_nat(1u);
v___x_58_ = l_List_get_x3fInternal___redArg(v_ws_35_, v___x_57_);
if (lean_obj_tag(v___x_58_) == 0)
{
lean_object* v___x_59_; lean_object* v___x_60_; 
v___x_59_ = lean_box(0);
lean_inc_ref(v___f_34_);
v___x_60_ = lean_apply_1(v___f_34_, v___x_59_);
v___y_48_ = v___y_56_;
v___y_49_ = v___x_60_;
goto v___jp_47_;
}
else
{
lean_object* v_val_61_; 
v_val_61_ = lean_ctor_get(v___x_58_, 0);
lean_inc(v_val_61_);
lean_dec_ref_known(v___x_58_, 1);
v___y_48_ = v___y_56_;
v___y_49_ = v_val_61_;
goto v___jp_47_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___lam__2___boxed(lean_object* v___f_67_, lean_object* v_ws_68_){
_start:
{
lean_object* v_res_69_; 
v_res_69_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO___lam__2(v___f_67_, v_ws_68_);
lean_dec(v_ws_68_);
return v_res_69_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_permute(lean_object* v_state_83_){
_start:
{
lean_object* v___x_84_; lean_object* v___x_85_; 
v___x_84_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawFO));
v___x_85_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Generic_permute___redArg(v___x_84_, v_state_83_);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_hashSlice(lean_object* v_values_86_){
_start:
{
lean_object* v___x_87_; lean_object* v___x_88_; 
v___x_87_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO));
v___x_88_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_hashSlice___redArg(v___x_87_, v_values_86_);
return v___x_88_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_compressDigest(lean_object* v_left_89_, lean_object* v_right_90_){
_start:
{
lean_object* v___x_91_; lean_object* v___x_92_; 
v___x_91_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO));
v___x_92_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_compressDigest___redArg(v___x_91_, v_left_89_, v_right_90_);
return v___x_92_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_List_beq___at___00Fundamentals_Poseidon2_Sponge_digestEq___at___00Fundamentals_Poseidon2_Sponge_merkleVerify___at___00Fundamentals_Poseidon2_Raw_merkleVerify_spec__0_spec__1_spec__2(lean_object* v_x_93_, lean_object* v_x_94_){
_start:
{
if (lean_obj_tag(v_x_93_) == 0)
{
if (lean_obj_tag(v_x_94_) == 0)
{
uint8_t v___x_95_; 
v___x_95_ = 1;
return v___x_95_;
}
else
{
uint8_t v___x_96_; 
v___x_96_ = 0;
return v___x_96_;
}
}
else
{
if (lean_obj_tag(v_x_94_) == 0)
{
uint8_t v___x_97_; 
v___x_97_ = 0;
return v___x_97_;
}
else
{
lean_object* v_head_98_; lean_object* v_tail_99_; lean_object* v_head_100_; lean_object* v_tail_101_; uint8_t v___x_102_; 
v_head_98_ = lean_ctor_get(v_x_93_, 0);
v_tail_99_ = lean_ctor_get(v_x_93_, 1);
v_head_100_ = lean_ctor_get(v_x_94_, 0);
v_tail_101_ = lean_ctor_get(v_x_94_, 1);
v___x_102_ = lean_nat_dec_eq(v_head_98_, v_head_100_);
if (v___x_102_ == 0)
{
return v___x_102_;
}
else
{
v_x_93_ = v_tail_99_;
v_x_94_ = v_tail_101_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_beq___at___00Fundamentals_Poseidon2_Sponge_digestEq___at___00Fundamentals_Poseidon2_Sponge_merkleVerify___at___00Fundamentals_Poseidon2_Raw_merkleVerify_spec__0_spec__1_spec__2___boxed(lean_object* v_x_104_, lean_object* v_x_105_){
_start:
{
uint8_t v_res_106_; lean_object* v_r_107_; 
v_res_106_ = lp_swirl_x2drbr_x2dformal_List_beq___at___00Fundamentals_Poseidon2_Sponge_digestEq___at___00Fundamentals_Poseidon2_Sponge_merkleVerify___at___00Fundamentals_Poseidon2_Raw_merkleVerify_spec__0_spec__1_spec__2(v_x_104_, v_x_105_);
lean_dec(v_x_105_);
lean_dec(v_x_104_);
v_r_107_ = lean_box(v_res_106_);
return v_r_107_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_digestEq___at___00Fundamentals_Poseidon2_Sponge_merkleVerify___at___00Fundamentals_Poseidon2_Raw_merkleVerify_spec__0_spec__1(lean_object* v_a_108_, lean_object* v_b_109_){
_start:
{
lean_object* v___x_110_; lean_object* v___x_111_; uint8_t v___x_112_; 
v___x_110_ = lean_array_to_list(v_a_108_);
v___x_111_ = lean_array_to_list(v_b_109_);
v___x_112_ = lp_swirl_x2drbr_x2dformal_List_beq___at___00Fundamentals_Poseidon2_Sponge_digestEq___at___00Fundamentals_Poseidon2_Sponge_merkleVerify___at___00Fundamentals_Poseidon2_Raw_merkleVerify_spec__0_spec__1_spec__2(v___x_110_, v___x_111_);
lean_dec(v___x_111_);
lean_dec(v___x_110_);
return v___x_112_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_digestEq___at___00Fundamentals_Poseidon2_Sponge_merkleVerify___at___00Fundamentals_Poseidon2_Raw_merkleVerify_spec__0_spec__1___boxed(lean_object* v_a_113_, lean_object* v_b_114_){
_start:
{
uint8_t v_res_115_; lean_object* v_r_116_; 
v_res_115_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_digestEq___at___00Fundamentals_Poseidon2_Sponge_merkleVerify___at___00Fundamentals_Poseidon2_Raw_merkleVerify_spec__0_spec__1(v_a_113_, v_b_114_);
v_r_116_ = lean_box(v_res_115_);
return v_r_116_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Fundamentals_Poseidon2_Sponge_merkleVerify___at___00Fundamentals_Poseidon2_Raw_merkleVerify_spec__0_spec__0___redArg(lean_object* v_to_117_, lean_object* v_x_118_, lean_object* v_x_119_){
_start:
{
if (lean_obj_tag(v_x_119_) == 0)
{
lean_dec_ref(v_to_117_);
return v_x_118_;
}
else
{
lean_object* v_head_120_; lean_object* v_tail_121_; lean_object* v_fst_122_; lean_object* v_snd_123_; lean_object* v___x_125_; uint8_t v_isShared_126_; uint8_t v_isSharedCheck_141_; 
v_head_120_ = lean_ctor_get(v_x_119_, 0);
lean_inc(v_head_120_);
v_tail_121_ = lean_ctor_get(v_x_119_, 1);
lean_inc(v_tail_121_);
lean_dec_ref_known(v_x_119_, 2);
v_fst_122_ = lean_ctor_get(v_x_118_, 0);
v_snd_123_ = lean_ctor_get(v_x_118_, 1);
v_isSharedCheck_141_ = !lean_is_exclusive(v_x_118_);
if (v_isSharedCheck_141_ == 0)
{
v___x_125_ = v_x_118_;
v_isShared_126_ = v_isSharedCheck_141_;
goto v_resetjp_124_;
}
else
{
lean_inc(v_snd_123_);
lean_inc(v_fst_122_);
lean_dec(v_x_118_);
v___x_125_ = lean_box(0);
v_isShared_126_ = v_isSharedCheck_141_;
goto v_resetjp_124_;
}
v_resetjp_124_:
{
lean_object* v___y_128_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; uint8_t v___x_138_; 
v___x_135_ = lean_unsigned_to_nat(2u);
v___x_136_ = lean_nat_mod(v_snd_123_, v___x_135_);
v___x_137_ = lean_unsigned_to_nat(0u);
v___x_138_ = lean_nat_dec_eq(v___x_136_, v___x_137_);
lean_dec(v___x_136_);
if (v___x_138_ == 0)
{
lean_object* v___x_139_; 
lean_inc_ref(v_to_117_);
v___x_139_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_compressDigest___redArg(v_to_117_, v_head_120_, v_fst_122_);
v___y_128_ = v___x_139_;
goto v___jp_127_;
}
else
{
lean_object* v___x_140_; 
lean_inc_ref(v_to_117_);
v___x_140_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_compressDigest___redArg(v_to_117_, v_fst_122_, v_head_120_);
v___y_128_ = v___x_140_;
goto v___jp_127_;
}
v___jp_127_:
{
lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_132_; 
v___x_129_ = lean_unsigned_to_nat(1u);
v___x_130_ = lean_nat_shiftr(v_snd_123_, v___x_129_);
lean_dec(v_snd_123_);
if (v_isShared_126_ == 0)
{
lean_ctor_set(v___x_125_, 1, v___x_130_);
lean_ctor_set(v___x_125_, 0, v___y_128_);
v___x_132_ = v___x_125_;
goto v_reusejp_131_;
}
else
{
lean_object* v_reuseFailAlloc_134_; 
v_reuseFailAlloc_134_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_134_, 0, v___y_128_);
lean_ctor_set(v_reuseFailAlloc_134_, 1, v___x_130_);
v___x_132_ = v_reuseFailAlloc_134_;
goto v_reusejp_131_;
}
v_reusejp_131_:
{
v_x_118_ = v___x_132_;
v_x_119_ = v_tail_121_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_merkleVerify___at___00Fundamentals_Poseidon2_Raw_merkleVerify_spec__0___redArg(lean_object* v_to_142_, lean_object* v_root_143_, lean_object* v_idx_144_, lean_object* v_leaf_145_, lean_object* v_merkleProof_146_){
_start:
{
lean_object* v___x_147_; lean_object* v_final_148_; lean_object* v_fst_149_; uint8_t v___x_150_; 
v___x_147_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_147_, 0, v_leaf_145_);
lean_ctor_set(v___x_147_, 1, v_idx_144_);
v_final_148_ = lp_swirl_x2drbr_x2dformal_List_foldl___at___00Fundamentals_Poseidon2_Sponge_merkleVerify___at___00Fundamentals_Poseidon2_Raw_merkleVerify_spec__0_spec__0___redArg(v_to_142_, v___x_147_, v_merkleProof_146_);
v_fst_149_ = lean_ctor_get(v_final_148_, 0);
lean_inc(v_fst_149_);
lean_dec_ref(v_final_148_);
v___x_150_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_digestEq___at___00Fundamentals_Poseidon2_Sponge_merkleVerify___at___00Fundamentals_Poseidon2_Raw_merkleVerify_spec__0_spec__1(v_fst_149_, v_root_143_);
return v___x_150_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_merkleVerify___at___00Fundamentals_Poseidon2_Raw_merkleVerify_spec__0___redArg___boxed(lean_object* v_to_151_, lean_object* v_root_152_, lean_object* v_idx_153_, lean_object* v_leaf_154_, lean_object* v_merkleProof_155_){
_start:
{
uint8_t v_res_156_; lean_object* v_r_157_; 
v_res_156_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_merkleVerify___at___00Fundamentals_Poseidon2_Raw_merkleVerify_spec__0___redArg(v_to_151_, v_root_152_, v_idx_153_, v_leaf_154_, v_merkleProof_155_);
v_r_157_ = lean_box(v_res_156_);
return v_r_157_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_merkleVerify(lean_object* v_root_158_, lean_object* v_idx_159_, lean_object* v_leaf_160_, lean_object* v_merkleProof_161_){
_start:
{
lean_object* v___x_162_; uint8_t v___x_163_; 
v___x_162_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_rawTO));
v___x_163_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_merkleVerify___at___00Fundamentals_Poseidon2_Raw_merkleVerify_spec__0___redArg(v___x_162_, v_root_158_, v_idx_159_, v_leaf_160_, v_merkleProof_161_);
return v___x_163_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_merkleVerify___boxed(lean_object* v_root_164_, lean_object* v_idx_165_, lean_object* v_leaf_166_, lean_object* v_merkleProof_167_){
_start:
{
uint8_t v_res_168_; lean_object* v_r_169_; 
v_res_168_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_merkleVerify(v_root_164_, v_idx_165_, v_leaf_166_, v_merkleProof_167_);
v_r_169_ = lean_box(v_res_168_);
return v_r_169_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_merkleVerify___at___00Fundamentals_Poseidon2_Raw_merkleVerify_spec__0(lean_object* v_EF_170_, lean_object* v_to_171_, lean_object* v_root_172_, lean_object* v_idx_173_, lean_object* v_leaf_174_, lean_object* v_merkleProof_175_){
_start:
{
uint8_t v___x_176_; 
v___x_176_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_merkleVerify___at___00Fundamentals_Poseidon2_Raw_merkleVerify_spec__0___redArg(v_to_171_, v_root_172_, v_idx_173_, v_leaf_174_, v_merkleProof_175_);
return v___x_176_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_merkleVerify___at___00Fundamentals_Poseidon2_Raw_merkleVerify_spec__0___boxed(lean_object* v_EF_177_, lean_object* v_to_178_, lean_object* v_root_179_, lean_object* v_idx_180_, lean_object* v_leaf_181_, lean_object* v_merkleProof_182_){
_start:
{
uint8_t v_res_183_; lean_object* v_r_184_; 
v_res_183_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_merkleVerify___at___00Fundamentals_Poseidon2_Raw_merkleVerify_spec__0(v_EF_177_, v_to_178_, v_root_179_, v_idx_180_, v_leaf_181_, v_merkleProof_182_);
v_r_184_ = lean_box(v_res_183_);
return v_r_184_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Fundamentals_Poseidon2_Sponge_merkleVerify___at___00Fundamentals_Poseidon2_Raw_merkleVerify_spec__0_spec__0(lean_object* v_EF_185_, lean_object* v_to_186_, lean_object* v_x_187_, lean_object* v_x_188_){
_start:
{
lean_object* v___x_189_; 
v___x_189_ = lp_swirl_x2drbr_x2dformal_List_foldl___at___00Fundamentals_Poseidon2_Sponge_merkleVerify___at___00Fundamentals_Poseidon2_Raw_merkleVerify_spec__0_spec__0___redArg(v_to_186_, v_x_187_, v_x_188_);
return v___x_189_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_swirl_x2drbr_x2dformal_Fundamentals_BabyBear_Raw(uint8_t builtin);
lean_object* initialize_swirl_x2drbr_x2dformal_Fundamentals_BabyBearExt4_Raw(uint8_t builtin);
lean_object* initialize_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Generic(uint8_t builtin);
lean_object* initialize_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
lean_initialize_runtime_module();
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2drbr_x2dformal_Fundamentals_BabyBear_Raw(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2drbr_x2dformal_Fundamentals_BabyBearExt4_Raw(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Generic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_WIDTH = _init_lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_WIDTH();
lean_mark_persistent(lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_WIDTH);
lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_RATE = _init_lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_RATE();
lean_mark_persistent(lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_RATE);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
