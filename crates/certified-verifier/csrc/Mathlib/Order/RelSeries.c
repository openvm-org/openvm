// Lean compiler output
// Module: Mathlib.Order.RelSeries
// Imports: public import Init public meta import Init public import Mathlib.Algebra.GroupWithZero.Nat public import Mathlib.Algebra.Order.Group.Nat public import Mathlib.Algebra.Order.Monoid.NatCast public import Mathlib.Basic.Rel public import Mathlib.Data.Fin.VecNotation public import Mathlib.Data.Fintype.Pi public import Mathlib.Data.Fintype.Pigeonhole public import Mathlib.Data.Fintype.Sigma public import Mathlib.Order.OrderIsoNat
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
lean_object* lean_nat_add(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t l_Nat_decidableForallFin___redArg(lean_object*, lean_object*);
lean_object* l_instDecidableEqFin___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_List_finRange(lean_object*);
lean_object* lp_mathlib_Fintype_piFinset___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* lp_mathlib_Finset_sigma___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_filter___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_pmap___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_map___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Fin_succAboveCases___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
lean_object* l_Fin_succ___redArg(lean_object*);
lean_object* l_Fin_addCases___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_get___redArg(lean_object*, lean_object*);
lean_object* l_List_ofFn___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Fin_tail___redArg(lean_object*, lean_object*);
lean_object* l_Nat_recCompiled___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_instCoeFunForallFinHAddNatLengthOfNat___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_RelSeries_instCoeFunForallFinHAddNatLengthOfNat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RelSeries_instCoeFunForallFinHAddNatLengthOfNat___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RelSeries_instCoeFunForallFinHAddNatLengthOfNat___closed__0 = (const lean_object*)&lp_mathlib_RelSeries_instCoeFunForallFinHAddNatLengthOfNat___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_instCoeFunForallFinHAddNatLengthOfNat(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_singleton___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_singleton___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_singleton___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_singleton(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_instInhabited(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_ofLE___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_ofLE(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_toList___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_toList(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_fromListIsChain___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_fromListIsChain___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_fromListIsChain___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_fromListIsChain(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_RelSeries_Equiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RelSeries_toList___redArg, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RelSeries_Equiv___closed__0 = (const lean_object*)&lp_mathlib_RelSeries_Equiv___closed__0_value;
static const lean_closure_object lp_mathlib_RelSeries_Equiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RelSeries_fromListIsChain___redArg, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RelSeries_Equiv___closed__1 = (const lean_object*)&lp_mathlib_RelSeries_Equiv___closed__1_value;
static const lean_ctor_object lp_mathlib_RelSeries_Equiv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_RelSeries_Equiv___closed__0_value),((lean_object*)&lp_mathlib_RelSeries_Equiv___closed__1_value)}};
static const lean_object* lp_mathlib_RelSeries_Equiv___closed__2 = (const lean_object*)&lp_mathlib_RelSeries_Equiv___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_Equiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_membership(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_head___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_head(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_last___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_last(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_append___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_append___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_append___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_append(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_map___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_insertNth___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_insertNth___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_insertNth___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_insertNth___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_insertNth(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_insertNth___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_reverse___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_reverse___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_reverse___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_reverse(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_cons___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_cons(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_snoc___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_snoc(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_tail___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_tail___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_tail___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_tail(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_inductionOn___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_inductionOn___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_inductionOn___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_inductionOn___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_inductionOn(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_eraseLast___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_eraseLast___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_eraseLast(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_inductionOn_x27___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_inductionOn_x27___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_inductionOn_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_inductionOn_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_smash___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_smash___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_smash___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_smash___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_smash(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_take___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_take___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_take(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_drop___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_drop___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_drop___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_drop(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_mk___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_mk(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_mk___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_injStrictMono___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_LTSeries_injStrictMono___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LTSeries_injStrictMono___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LTSeries_injStrictMono___closed__0 = (const lean_object*)&lp_mathlib_LTSeries_injStrictMono___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_injStrictMono(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_injStrictMono___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_range___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_range___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_LTSeries_range___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LTSeries_range___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LTSeries_range___closed__0 = (const lean_object*)&lp_mathlib_LTSeries_range___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_range(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__3___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__4___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__5___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__5___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___closed__0 = (const lean_object*)&lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_instCoeFunForallFinHAddNatLengthOfNat___lam__0(lean_object* v_self_1_, lean_object* v___y_2_){
_start:
{
lean_object* v_toFun_3_; lean_object* v___x_4_; 
v_toFun_3_ = lean_ctor_get(v_self_1_, 1);
lean_inc(v_toFun_3_);
lean_dec_ref(v_self_1_);
v___x_4_ = lean_apply_1(v_toFun_3_, v___y_2_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_instCoeFunForallFinHAddNatLengthOfNat(lean_object* v_00_u03b1_6_, lean_object* v_r_7_){
_start:
{
lean_object* v___f_8_; 
v___f_8_ = ((lean_object*)(lp_mathlib_RelSeries_instCoeFunForallFinHAddNatLengthOfNat___closed__0));
return v___f_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_singleton___redArg___lam__0(lean_object* v_a_9_, lean_object* v_x_10_){
_start:
{
lean_inc(v_a_9_);
return v_a_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_singleton___redArg___lam__0___boxed(lean_object* v_a_11_, lean_object* v_x_12_){
_start:
{
lean_object* v_res_13_; 
v_res_13_ = lp_mathlib_RelSeries_singleton___redArg___lam__0(v_a_11_, v_x_12_);
lean_dec(v_x_12_);
lean_dec(v_a_11_);
return v_res_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_singleton___redArg(lean_object* v_a_14_){
_start:
{
lean_object* v___f_15_; lean_object* v___x_16_; lean_object* v___x_17_; 
v___f_15_ = lean_alloc_closure((void*)(lp_mathlib_RelSeries_singleton___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_15_, 0, v_a_14_);
v___x_16_ = lean_unsigned_to_nat(0u);
v___x_17_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_17_, 0, v___x_16_);
lean_ctor_set(v___x_17_, 1, v___f_15_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_singleton(lean_object* v_00_u03b1_18_, lean_object* v_r_19_, lean_object* v_a_20_){
_start:
{
lean_object* v___x_21_; 
v___x_21_ = lp_mathlib_RelSeries_singleton___redArg(v_a_20_);
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_instInhabited___redArg(lean_object* v_inst_22_){
_start:
{
lean_object* v___x_23_; 
v___x_23_ = lp_mathlib_RelSeries_singleton___redArg(v_inst_22_);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_instInhabited(lean_object* v_00_u03b1_24_, lean_object* v_r_25_, lean_object* v_inst_26_){
_start:
{
lean_object* v___x_27_; 
v___x_27_ = lp_mathlib_RelSeries_singleton___redArg(v_inst_26_);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_ofLE___redArg(lean_object* v_x_28_){
_start:
{
lean_object* v_length_29_; lean_object* v_toFun_30_; lean_object* v___x_32_; uint8_t v_isShared_33_; uint8_t v_isSharedCheck_37_; 
v_length_29_ = lean_ctor_get(v_x_28_, 0);
v_toFun_30_ = lean_ctor_get(v_x_28_, 1);
v_isSharedCheck_37_ = !lean_is_exclusive(v_x_28_);
if (v_isSharedCheck_37_ == 0)
{
v___x_32_ = v_x_28_;
v_isShared_33_ = v_isSharedCheck_37_;
goto v_resetjp_31_;
}
else
{
lean_inc(v_toFun_30_);
lean_inc(v_length_29_);
lean_dec(v_x_28_);
v___x_32_ = lean_box(0);
v_isShared_33_ = v_isSharedCheck_37_;
goto v_resetjp_31_;
}
v_resetjp_31_:
{
lean_object* v___x_35_; 
if (v_isShared_33_ == 0)
{
v___x_35_ = v___x_32_;
goto v_reusejp_34_;
}
else
{
lean_object* v_reuseFailAlloc_36_; 
v_reuseFailAlloc_36_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_36_, 0, v_length_29_);
lean_ctor_set(v_reuseFailAlloc_36_, 1, v_toFun_30_);
v___x_35_ = v_reuseFailAlloc_36_;
goto v_reusejp_34_;
}
v_reusejp_34_:
{
return v___x_35_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_ofLE(lean_object* v_00_u03b1_38_, lean_object* v_r_39_, lean_object* v_x_40_, lean_object* v_s_41_, lean_object* v_h_42_){
_start:
{
lean_object* v___x_43_; 
v___x_43_ = lp_mathlib_RelSeries_ofLE___redArg(v_x_40_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_toList___redArg(lean_object* v_x_44_){
_start:
{
lean_object* v_length_45_; lean_object* v_toFun_46_; lean_object* v___x_47_; lean_object* v___x_48_; lean_object* v___x_49_; 
v_length_45_ = lean_ctor_get(v_x_44_, 0);
lean_inc(v_length_45_);
v_toFun_46_ = lean_ctor_get(v_x_44_, 1);
lean_inc(v_toFun_46_);
lean_dec_ref(v_x_44_);
v___x_47_ = lean_unsigned_to_nat(1u);
v___x_48_ = lean_nat_add(v_length_45_, v___x_47_);
lean_dec(v_length_45_);
v___x_49_ = l_List_ofFn___redArg(v___x_48_, v_toFun_46_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_toList(lean_object* v_00_u03b1_50_, lean_object* v_r_51_, lean_object* v_x_52_){
_start:
{
lean_object* v___x_53_; 
v___x_53_ = lp_mathlib_RelSeries_toList___redArg(v_x_52_);
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_fromListIsChain___redArg___lam__0(lean_object* v_x_54_, lean_object* v_i_55_){
_start:
{
lean_object* v___x_56_; 
v___x_56_ = l_List_get___redArg(v_x_54_, v_i_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_fromListIsChain___redArg___lam__0___boxed(lean_object* v_x_57_, lean_object* v_i_58_){
_start:
{
lean_object* v_res_59_; 
v_res_59_ = lp_mathlib_RelSeries_fromListIsChain___redArg___lam__0(v_x_57_, v_i_58_);
lean_dec(v_x_57_);
return v_res_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_fromListIsChain___redArg(lean_object* v_x_60_){
_start:
{
lean_object* v___f_61_; lean_object* v___x_62_; lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; 
lean_inc(v_x_60_);
v___f_61_ = lean_alloc_closure((void*)(lp_mathlib_RelSeries_fromListIsChain___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_61_, 0, v_x_60_);
v___x_62_ = l_List_lengthTR___redArg(v_x_60_);
lean_dec(v_x_60_);
v___x_63_ = lean_unsigned_to_nat(1u);
v___x_64_ = lean_nat_sub(v___x_62_, v___x_63_);
lean_dec(v___x_62_);
v___x_65_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_65_, 0, v___x_64_);
lean_ctor_set(v___x_65_, 1, v___f_61_);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_fromListIsChain(lean_object* v_00_u03b1_66_, lean_object* v_r_67_, lean_object* v_x_68_, lean_object* v_x__ne__nil_69_, lean_object* v_hx_70_){
_start:
{
lean_object* v___x_71_; 
v___x_71_ = lp_mathlib_RelSeries_fromListIsChain___redArg(v_x_68_);
return v___x_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_Equiv(lean_object* v_00_u03b1_77_, lean_object* v_r_78_){
_start:
{
lean_object* v___x_79_; 
v___x_79_ = ((lean_object*)(lp_mathlib_RelSeries_Equiv___closed__2));
return v___x_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_membership(lean_object* v_00_u03b1_80_, lean_object* v_r_81_){
_start:
{
lean_object* v___x_82_; 
v___x_82_ = lean_box(0);
return v___x_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_head___redArg(lean_object* v_x_83_){
_start:
{
lean_object* v_length_84_; lean_object* v_toFun_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; 
v_length_84_ = lean_ctor_get(v_x_83_, 0);
lean_inc(v_length_84_);
v_toFun_85_ = lean_ctor_get(v_x_83_, 1);
lean_inc(v_toFun_85_);
lean_dec_ref(v_x_83_);
v___x_86_ = lean_unsigned_to_nat(1u);
v___x_87_ = lean_nat_add(v_length_84_, v___x_86_);
lean_dec(v_length_84_);
v___x_88_ = lean_unsigned_to_nat(0u);
v___x_89_ = lean_nat_mod(v___x_88_, v___x_87_);
lean_dec(v___x_87_);
v___x_90_ = lean_apply_1(v_toFun_85_, v___x_89_);
return v___x_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_head(lean_object* v_00_u03b1_91_, lean_object* v_r_92_, lean_object* v_x_93_){
_start:
{
lean_object* v___x_94_; 
v___x_94_ = lp_mathlib_RelSeries_head___redArg(v_x_93_);
return v___x_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_last___redArg(lean_object* v_x_95_){
_start:
{
lean_object* v_length_96_; lean_object* v_toFun_97_; lean_object* v___x_98_; 
v_length_96_ = lean_ctor_get(v_x_95_, 0);
lean_inc(v_length_96_);
v_toFun_97_ = lean_ctor_get(v_x_95_, 1);
lean_inc(v_toFun_97_);
lean_dec_ref(v_x_95_);
v___x_98_ = lean_apply_1(v_toFun_97_, v_length_96_);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_last(lean_object* v_00_u03b1_99_, lean_object* v_r_100_, lean_object* v_x_101_){
_start:
{
lean_object* v___x_102_; 
v___x_102_ = lp_mathlib_RelSeries_last___redArg(v_x_101_);
return v___x_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_append___redArg___lam__0(lean_object* v___x_103_, lean_object* v_toFun_104_, lean_object* v_toFun_105_, lean_object* v___y_106_){
_start:
{
lean_object* v___x_107_; 
v___x_107_ = l_Fin_addCases___redArg(v___x_103_, v_toFun_104_, v_toFun_105_, v___y_106_);
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_append___redArg___lam__0___boxed(lean_object* v___x_108_, lean_object* v_toFun_109_, lean_object* v_toFun_110_, lean_object* v___y_111_){
_start:
{
lean_object* v_res_112_; 
v_res_112_ = lp_mathlib_RelSeries_append___redArg___lam__0(v___x_108_, v_toFun_109_, v_toFun_110_, v___y_111_);
lean_dec(v___x_108_);
return v_res_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_append___redArg(lean_object* v_p_113_, lean_object* v_q_114_){
_start:
{
lean_object* v_length_115_; lean_object* v_toFun_116_; lean_object* v_length_117_; lean_object* v_toFun_118_; lean_object* v___x_120_; uint8_t v_isShared_121_; uint8_t v_isSharedCheck_130_; 
v_length_115_ = lean_ctor_get(v_p_113_, 0);
lean_inc(v_length_115_);
v_toFun_116_ = lean_ctor_get(v_p_113_, 1);
lean_inc(v_toFun_116_);
lean_dec_ref(v_p_113_);
v_length_117_ = lean_ctor_get(v_q_114_, 0);
v_toFun_118_ = lean_ctor_get(v_q_114_, 1);
v_isSharedCheck_130_ = !lean_is_exclusive(v_q_114_);
if (v_isSharedCheck_130_ == 0)
{
v___x_120_ = v_q_114_;
v_isShared_121_ = v_isSharedCheck_130_;
goto v_resetjp_119_;
}
else
{
lean_inc(v_toFun_118_);
lean_inc(v_length_117_);
lean_dec(v_q_114_);
v___x_120_ = lean_box(0);
v_isShared_121_ = v_isSharedCheck_130_;
goto v_resetjp_119_;
}
v_resetjp_119_:
{
lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___f_126_; lean_object* v___x_128_; 
v___x_122_ = lean_nat_add(v_length_115_, v_length_117_);
lean_dec(v_length_117_);
v___x_123_ = lean_unsigned_to_nat(1u);
v___x_124_ = lean_nat_add(v___x_122_, v___x_123_);
lean_dec(v___x_122_);
v___x_125_ = lean_nat_add(v_length_115_, v___x_123_);
lean_dec(v_length_115_);
v___f_126_ = lean_alloc_closure((void*)(lp_mathlib_RelSeries_append___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_126_, 0, v___x_125_);
lean_closure_set(v___f_126_, 1, v_toFun_116_);
lean_closure_set(v___f_126_, 2, v_toFun_118_);
if (v_isShared_121_ == 0)
{
lean_ctor_set(v___x_120_, 1, v___f_126_);
lean_ctor_set(v___x_120_, 0, v___x_124_);
v___x_128_ = v___x_120_;
goto v_reusejp_127_;
}
else
{
lean_object* v_reuseFailAlloc_129_; 
v_reuseFailAlloc_129_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_129_, 0, v___x_124_);
lean_ctor_set(v_reuseFailAlloc_129_, 1, v___f_126_);
v___x_128_ = v_reuseFailAlloc_129_;
goto v_reusejp_127_;
}
v_reusejp_127_:
{
return v___x_128_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_append(lean_object* v_00_u03b1_131_, lean_object* v_r_132_, lean_object* v_p_133_, lean_object* v_q_134_, lean_object* v_connect_135_){
_start:
{
lean_object* v___x_136_; 
v___x_136_ = lp_mathlib_RelSeries_append___redArg(v_p_133_, v_q_134_);
return v___x_136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_map___redArg___lam__0(lean_object* v_toFun_137_, lean_object* v_f_138_, lean_object* v___y_139_){
_start:
{
lean_object* v___x_140_; lean_object* v___x_141_; 
v___x_140_ = lean_apply_1(v_toFun_137_, v___y_139_);
v___x_141_ = lean_apply_1(v_f_138_, v___x_140_);
return v___x_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_map___redArg(lean_object* v_p_142_, lean_object* v_f_143_){
_start:
{
lean_object* v_length_144_; lean_object* v_toFun_145_; lean_object* v___x_147_; uint8_t v_isShared_148_; uint8_t v_isSharedCheck_153_; 
v_length_144_ = lean_ctor_get(v_p_142_, 0);
v_toFun_145_ = lean_ctor_get(v_p_142_, 1);
v_isSharedCheck_153_ = !lean_is_exclusive(v_p_142_);
if (v_isSharedCheck_153_ == 0)
{
v___x_147_ = v_p_142_;
v_isShared_148_ = v_isSharedCheck_153_;
goto v_resetjp_146_;
}
else
{
lean_inc(v_toFun_145_);
lean_inc(v_length_144_);
lean_dec(v_p_142_);
v___x_147_ = lean_box(0);
v_isShared_148_ = v_isSharedCheck_153_;
goto v_resetjp_146_;
}
v_resetjp_146_:
{
lean_object* v___f_149_; lean_object* v___x_151_; 
v___f_149_ = lean_alloc_closure((void*)(lp_mathlib_RelSeries_map___redArg___lam__0), 3, 2);
lean_closure_set(v___f_149_, 0, v_toFun_145_);
lean_closure_set(v___f_149_, 1, v_f_143_);
if (v_isShared_148_ == 0)
{
lean_ctor_set(v___x_147_, 1, v___f_149_);
v___x_151_ = v___x_147_;
goto v_reusejp_150_;
}
else
{
lean_object* v_reuseFailAlloc_152_; 
v_reuseFailAlloc_152_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_152_, 0, v_length_144_);
lean_ctor_set(v_reuseFailAlloc_152_, 1, v___f_149_);
v___x_151_ = v_reuseFailAlloc_152_;
goto v_reusejp_150_;
}
v_reusejp_150_:
{
return v___x_151_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_map(lean_object* v_00_u03b1_154_, lean_object* v_r_155_, lean_object* v_00_u03b2_156_, lean_object* v_s_157_, lean_object* v_p_158_, lean_object* v_f_159_){
_start:
{
lean_object* v___x_160_; 
v___x_160_ = lp_mathlib_RelSeries_map___redArg(v_p_158_, v_f_159_);
return v___x_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_insertNth___redArg___lam__0(lean_object* v___x_161_, lean_object* v_a_162_, lean_object* v_toFun_163_, lean_object* v___y_164_){
_start:
{
lean_object* v___x_165_; 
v___x_165_ = lp_mathlib_Fin_succAboveCases___redArg(v___x_161_, v_a_162_, v_toFun_163_, v___y_164_);
return v___x_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_insertNth___redArg___lam__0___boxed(lean_object* v___x_166_, lean_object* v_a_167_, lean_object* v_toFun_168_, lean_object* v___y_169_){
_start:
{
lean_object* v_res_170_; 
v_res_170_ = lp_mathlib_RelSeries_insertNth___redArg___lam__0(v___x_166_, v_a_167_, v_toFun_168_, v___y_169_);
lean_dec(v_a_167_);
lean_dec(v___x_166_);
return v_res_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_insertNth___redArg(lean_object* v_p_171_, lean_object* v_i_172_, lean_object* v_a_173_){
_start:
{
lean_object* v_length_174_; lean_object* v_toFun_175_; lean_object* v___x_177_; uint8_t v_isShared_178_; uint8_t v_isSharedCheck_186_; 
v_length_174_ = lean_ctor_get(v_p_171_, 0);
v_toFun_175_ = lean_ctor_get(v_p_171_, 1);
v_isSharedCheck_186_ = !lean_is_exclusive(v_p_171_);
if (v_isSharedCheck_186_ == 0)
{
v___x_177_ = v_p_171_;
v_isShared_178_ = v_isSharedCheck_186_;
goto v_resetjp_176_;
}
else
{
lean_inc(v_toFun_175_);
lean_inc(v_length_174_);
lean_dec(v_p_171_);
v___x_177_ = lean_box(0);
v_isShared_178_ = v_isSharedCheck_186_;
goto v_resetjp_176_;
}
v_resetjp_176_:
{
lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___f_182_; lean_object* v___x_184_; 
v___x_179_ = lean_unsigned_to_nat(1u);
v___x_180_ = lean_nat_add(v_length_174_, v___x_179_);
lean_dec(v_length_174_);
v___x_181_ = l_Fin_succ___redArg(v_i_172_);
v___f_182_ = lean_alloc_closure((void*)(lp_mathlib_RelSeries_insertNth___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_182_, 0, v___x_181_);
lean_closure_set(v___f_182_, 1, v_a_173_);
lean_closure_set(v___f_182_, 2, v_toFun_175_);
if (v_isShared_178_ == 0)
{
lean_ctor_set(v___x_177_, 1, v___f_182_);
lean_ctor_set(v___x_177_, 0, v___x_180_);
v___x_184_ = v___x_177_;
goto v_reusejp_183_;
}
else
{
lean_object* v_reuseFailAlloc_185_; 
v_reuseFailAlloc_185_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_185_, 0, v___x_180_);
lean_ctor_set(v_reuseFailAlloc_185_, 1, v___f_182_);
v___x_184_ = v_reuseFailAlloc_185_;
goto v_reusejp_183_;
}
v_reusejp_183_:
{
return v___x_184_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_insertNth___redArg___boxed(lean_object* v_p_187_, lean_object* v_i_188_, lean_object* v_a_189_){
_start:
{
lean_object* v_res_190_; 
v_res_190_ = lp_mathlib_RelSeries_insertNth___redArg(v_p_187_, v_i_188_, v_a_189_);
lean_dec(v_i_188_);
return v_res_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_insertNth(lean_object* v_00_u03b1_191_, lean_object* v_r_192_, lean_object* v_p_193_, lean_object* v_i_194_, lean_object* v_a_195_, lean_object* v_prev__connect_196_, lean_object* v_connect__next_197_){
_start:
{
lean_object* v___x_198_; 
v___x_198_ = lp_mathlib_RelSeries_insertNth___redArg(v_p_193_, v_i_194_, v_a_195_);
return v___x_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_insertNth___boxed(lean_object* v_00_u03b1_199_, lean_object* v_r_200_, lean_object* v_p_201_, lean_object* v_i_202_, lean_object* v_a_203_, lean_object* v_prev__connect_204_, lean_object* v_connect__next_205_){
_start:
{
lean_object* v_res_206_; 
v_res_206_ = lp_mathlib_RelSeries_insertNth(v_00_u03b1_199_, v_r_200_, v_p_201_, v_i_202_, v_a_203_, v_prev__connect_204_, v_connect__next_205_);
lean_dec(v_i_202_);
return v_res_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_reverse___redArg___lam__0(lean_object* v___x_207_, lean_object* v___x_208_, lean_object* v_toFun_209_, lean_object* v___y_210_){
_start:
{
lean_object* v___x_211_; lean_object* v___x_212_; lean_object* v___x_213_; 
v___x_211_ = lean_nat_add(v___y_210_, v___x_207_);
v___x_212_ = lean_nat_sub(v___x_208_, v___x_211_);
lean_dec(v___x_211_);
v___x_213_ = lean_apply_1(v_toFun_209_, v___x_212_);
return v___x_213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_reverse___redArg___lam__0___boxed(lean_object* v___x_214_, lean_object* v___x_215_, lean_object* v_toFun_216_, lean_object* v___y_217_){
_start:
{
lean_object* v_res_218_; 
v_res_218_ = lp_mathlib_RelSeries_reverse___redArg___lam__0(v___x_214_, v___x_215_, v_toFun_216_, v___y_217_);
lean_dec(v___y_217_);
lean_dec(v___x_215_);
lean_dec(v___x_214_);
return v_res_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_reverse___redArg(lean_object* v_p_219_){
_start:
{
lean_object* v_length_220_; lean_object* v_toFun_221_; lean_object* v___x_223_; uint8_t v_isShared_224_; uint8_t v_isSharedCheck_231_; 
v_length_220_ = lean_ctor_get(v_p_219_, 0);
v_toFun_221_ = lean_ctor_get(v_p_219_, 1);
v_isSharedCheck_231_ = !lean_is_exclusive(v_p_219_);
if (v_isSharedCheck_231_ == 0)
{
v___x_223_ = v_p_219_;
v_isShared_224_ = v_isSharedCheck_231_;
goto v_resetjp_222_;
}
else
{
lean_inc(v_toFun_221_);
lean_inc(v_length_220_);
lean_dec(v_p_219_);
v___x_223_ = lean_box(0);
v_isShared_224_ = v_isSharedCheck_231_;
goto v_resetjp_222_;
}
v_resetjp_222_:
{
lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___f_227_; lean_object* v___x_229_; 
v___x_225_ = lean_unsigned_to_nat(1u);
v___x_226_ = lean_nat_add(v_length_220_, v___x_225_);
v___f_227_ = lean_alloc_closure((void*)(lp_mathlib_RelSeries_reverse___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_227_, 0, v___x_225_);
lean_closure_set(v___f_227_, 1, v___x_226_);
lean_closure_set(v___f_227_, 2, v_toFun_221_);
if (v_isShared_224_ == 0)
{
lean_ctor_set(v___x_223_, 1, v___f_227_);
v___x_229_ = v___x_223_;
goto v_reusejp_228_;
}
else
{
lean_object* v_reuseFailAlloc_230_; 
v_reuseFailAlloc_230_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_230_, 0, v_length_220_);
lean_ctor_set(v_reuseFailAlloc_230_, 1, v___f_227_);
v___x_229_ = v_reuseFailAlloc_230_;
goto v_reusejp_228_;
}
v_reusejp_228_:
{
return v___x_229_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_reverse(lean_object* v_00_u03b1_232_, lean_object* v_r_233_, lean_object* v_p_234_){
_start:
{
lean_object* v___x_235_; 
v___x_235_ = lp_mathlib_RelSeries_reverse___redArg(v_p_234_);
return v___x_235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_cons___redArg(lean_object* v_p_236_, lean_object* v_newHead_237_){
_start:
{
lean_object* v___x_238_; lean_object* v___x_239_; 
v___x_238_ = lp_mathlib_RelSeries_singleton___redArg(v_newHead_237_);
v___x_239_ = lp_mathlib_RelSeries_append___redArg(v___x_238_, v_p_236_);
return v___x_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_cons(lean_object* v_00_u03b1_240_, lean_object* v_r_241_, lean_object* v_p_242_, lean_object* v_newHead_243_, lean_object* v_rel_244_){
_start:
{
lean_object* v___x_245_; 
v___x_245_ = lp_mathlib_RelSeries_cons___redArg(v_p_242_, v_newHead_243_);
return v___x_245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_snoc___redArg(lean_object* v_p_246_, lean_object* v_newLast_247_){
_start:
{
lean_object* v___x_248_; lean_object* v___x_249_; 
v___x_248_ = lp_mathlib_RelSeries_singleton___redArg(v_newLast_247_);
v___x_249_ = lp_mathlib_RelSeries_append___redArg(v_p_246_, v___x_248_);
return v___x_249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_snoc(lean_object* v_00_u03b1_250_, lean_object* v_r_251_, lean_object* v_p_252_, lean_object* v_newLast_253_, lean_object* v_rel_254_){
_start:
{
lean_object* v___x_255_; 
v___x_255_ = lp_mathlib_RelSeries_snoc___redArg(v_p_252_, v_newLast_253_);
return v___x_255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_tail___redArg___lam__0(lean_object* v_toFun_256_, lean_object* v___y_257_){
_start:
{
lean_object* v___x_258_; 
v___x_258_ = lp_mathlib_Fin_tail___redArg(v_toFun_256_, v___y_257_);
return v___x_258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_tail___redArg___lam__0___boxed(lean_object* v_toFun_259_, lean_object* v___y_260_){
_start:
{
lean_object* v_res_261_; 
v_res_261_ = lp_mathlib_RelSeries_tail___redArg___lam__0(v_toFun_259_, v___y_260_);
lean_dec(v___y_260_);
return v_res_261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_tail___redArg(lean_object* v_p_262_){
_start:
{
lean_object* v_length_263_; lean_object* v_toFun_264_; lean_object* v___x_266_; uint8_t v_isShared_267_; uint8_t v_isSharedCheck_274_; 
v_length_263_ = lean_ctor_get(v_p_262_, 0);
v_toFun_264_ = lean_ctor_get(v_p_262_, 1);
v_isSharedCheck_274_ = !lean_is_exclusive(v_p_262_);
if (v_isSharedCheck_274_ == 0)
{
v___x_266_ = v_p_262_;
v_isShared_267_ = v_isSharedCheck_274_;
goto v_resetjp_265_;
}
else
{
lean_inc(v_toFun_264_);
lean_inc(v_length_263_);
lean_dec(v_p_262_);
v___x_266_ = lean_box(0);
v_isShared_267_ = v_isSharedCheck_274_;
goto v_resetjp_265_;
}
v_resetjp_265_:
{
lean_object* v___f_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_272_; 
v___f_268_ = lean_alloc_closure((void*)(lp_mathlib_RelSeries_tail___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_268_, 0, v_toFun_264_);
v___x_269_ = lean_unsigned_to_nat(1u);
v___x_270_ = lean_nat_sub(v_length_263_, v___x_269_);
lean_dec(v_length_263_);
if (v_isShared_267_ == 0)
{
lean_ctor_set(v___x_266_, 1, v___f_268_);
lean_ctor_set(v___x_266_, 0, v___x_270_);
v___x_272_ = v___x_266_;
goto v_reusejp_271_;
}
else
{
lean_object* v_reuseFailAlloc_273_; 
v_reuseFailAlloc_273_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_273_, 0, v___x_270_);
lean_ctor_set(v_reuseFailAlloc_273_, 1, v___f_268_);
v___x_272_ = v_reuseFailAlloc_273_;
goto v_reusejp_271_;
}
v_reusejp_271_:
{
return v___x_272_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_tail(lean_object* v_00_u03b1_275_, lean_object* v_r_276_, lean_object* v_p_277_, lean_object* v_len__pos_278_){
_start:
{
lean_object* v___x_279_; 
v___x_279_ = lp_mathlib_RelSeries_tail___redArg(v_p_277_);
return v___x_279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_inductionOn___redArg___lam__0(lean_object* v_singleton_280_, lean_object* v_p_281_, lean_object* v_heq_282_){
_start:
{
lean_object* v___x_283_; lean_object* v___x_284_; 
v___x_283_ = lp_mathlib_RelSeries_head___redArg(v_p_281_);
v___x_284_ = lean_apply_1(v_singleton_280_, v___x_283_);
return v___x_284_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_inductionOn___redArg___lam__1(lean_object* v_cons_285_, lean_object* v_d_286_, lean_object* v_hd_287_, lean_object* v_p_288_, lean_object* v_heq_289_){
_start:
{
lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; 
lean_inc_ref(v_p_288_);
v___x_290_ = lp_mathlib_RelSeries_tail___redArg(v_p_288_);
v___x_291_ = lp_mathlib_RelSeries_head___redArg(v_p_288_);
lean_inc_ref(v___x_290_);
v___x_292_ = lean_apply_2(v_hd_287_, v___x_290_, lean_box(0));
v___x_293_ = lean_apply_4(v_cons_285_, v___x_290_, v___x_291_, lean_box(0), v___x_292_);
return v___x_293_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_inductionOn___redArg___lam__1___boxed(lean_object* v_cons_294_, lean_object* v_d_295_, lean_object* v_hd_296_, lean_object* v_p_297_, lean_object* v_heq_298_){
_start:
{
lean_object* v_res_299_; 
v_res_299_ = lp_mathlib_RelSeries_inductionOn___redArg___lam__1(v_cons_294_, v_d_295_, v_hd_296_, v_p_297_, v_heq_298_);
lean_dec(v_d_295_);
return v_res_299_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_inductionOn___redArg(lean_object* v_singleton_300_, lean_object* v_cons_301_, lean_object* v_p_302_){
_start:
{
lean_object* v_length_303_; lean_object* v___f_304_; lean_object* v___f_305_; lean_object* v___x_25__overap_306_; lean_object* v___x_307_; 
v_length_303_ = lean_ctor_get(v_p_302_, 0);
v___f_304_ = lean_alloc_closure((void*)(lp_mathlib_RelSeries_inductionOn___redArg___lam__0), 3, 1);
lean_closure_set(v___f_304_, 0, v_singleton_300_);
v___f_305_ = lean_alloc_closure((void*)(lp_mathlib_RelSeries_inductionOn___redArg___lam__1___boxed), 5, 1);
lean_closure_set(v___f_305_, 0, v_cons_301_);
v___x_25__overap_306_ = l_Nat_recCompiled___redArg(v___f_304_, v___f_305_, v_length_303_);
lean_dec_ref(v___f_304_);
v___x_307_ = lean_apply_2(v___x_25__overap_306_, v_p_302_, lean_box(0));
return v___x_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_inductionOn(lean_object* v_00_u03b1_308_, lean_object* v_r_309_, lean_object* v_motive_310_, lean_object* v_singleton_311_, lean_object* v_cons_312_, lean_object* v_p_313_){
_start:
{
lean_object* v___x_314_; 
v___x_314_ = lp_mathlib_RelSeries_inductionOn___redArg(v_singleton_311_, v_cons_312_, v_p_313_);
return v___x_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_eraseLast___redArg___lam__0(lean_object* v_toFun_315_, lean_object* v_i_316_){
_start:
{
lean_object* v___x_317_; 
v___x_317_ = lean_apply_1(v_toFun_315_, v_i_316_);
return v___x_317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_eraseLast___redArg(lean_object* v_p_318_){
_start:
{
lean_object* v_length_319_; lean_object* v_toFun_320_; lean_object* v___x_322_; uint8_t v_isShared_323_; uint8_t v_isSharedCheck_330_; 
v_length_319_ = lean_ctor_get(v_p_318_, 0);
v_toFun_320_ = lean_ctor_get(v_p_318_, 1);
v_isSharedCheck_330_ = !lean_is_exclusive(v_p_318_);
if (v_isSharedCheck_330_ == 0)
{
v___x_322_ = v_p_318_;
v_isShared_323_ = v_isSharedCheck_330_;
goto v_resetjp_321_;
}
else
{
lean_inc(v_toFun_320_);
lean_inc(v_length_319_);
lean_dec(v_p_318_);
v___x_322_ = lean_box(0);
v_isShared_323_ = v_isSharedCheck_330_;
goto v_resetjp_321_;
}
v_resetjp_321_:
{
lean_object* v___f_324_; lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_328_; 
v___f_324_ = lean_alloc_closure((void*)(lp_mathlib_RelSeries_eraseLast___redArg___lam__0), 2, 1);
lean_closure_set(v___f_324_, 0, v_toFun_320_);
v___x_325_ = lean_unsigned_to_nat(1u);
v___x_326_ = lean_nat_sub(v_length_319_, v___x_325_);
lean_dec(v_length_319_);
if (v_isShared_323_ == 0)
{
lean_ctor_set(v___x_322_, 1, v___f_324_);
lean_ctor_set(v___x_322_, 0, v___x_326_);
v___x_328_ = v___x_322_;
goto v_reusejp_327_;
}
else
{
lean_object* v_reuseFailAlloc_329_; 
v_reuseFailAlloc_329_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_329_, 0, v___x_326_);
lean_ctor_set(v_reuseFailAlloc_329_, 1, v___f_324_);
v___x_328_ = v_reuseFailAlloc_329_;
goto v_reusejp_327_;
}
v_reusejp_327_:
{
return v___x_328_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_eraseLast(lean_object* v_00_u03b1_331_, lean_object* v_r_332_, lean_object* v_p_333_){
_start:
{
lean_object* v___x_334_; 
v___x_334_ = lp_mathlib_RelSeries_eraseLast___redArg(v_p_333_);
return v___x_334_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_inductionOn_x27___redArg___lam__1(lean_object* v_snoc_335_, lean_object* v_d_336_, lean_object* v_hd_337_, lean_object* v_p_338_, lean_object* v_heq_339_){
_start:
{
lean_object* v___x_340_; lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; 
lean_inc_ref(v_p_338_);
v___x_340_ = lp_mathlib_RelSeries_eraseLast___redArg(v_p_338_);
v___x_341_ = lp_mathlib_RelSeries_last___redArg(v_p_338_);
lean_inc_ref(v___x_340_);
v___x_342_ = lean_apply_2(v_hd_337_, v___x_340_, lean_box(0));
v___x_343_ = lean_apply_4(v_snoc_335_, v___x_340_, v___x_341_, lean_box(0), v___x_342_);
return v___x_343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_inductionOn_x27___redArg___lam__1___boxed(lean_object* v_snoc_344_, lean_object* v_d_345_, lean_object* v_hd_346_, lean_object* v_p_347_, lean_object* v_heq_348_){
_start:
{
lean_object* v_res_349_; 
v_res_349_ = lp_mathlib_RelSeries_inductionOn_x27___redArg___lam__1(v_snoc_344_, v_d_345_, v_hd_346_, v_p_347_, v_heq_348_);
lean_dec(v_d_345_);
return v_res_349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_inductionOn_x27___redArg(lean_object* v_singleton_350_, lean_object* v_snoc_351_, lean_object* v_p_352_){
_start:
{
lean_object* v_length_353_; lean_object* v___f_354_; lean_object* v___f_355_; lean_object* v___x_25__overap_356_; lean_object* v___x_357_; 
v_length_353_ = lean_ctor_get(v_p_352_, 0);
v___f_354_ = lean_alloc_closure((void*)(lp_mathlib_RelSeries_inductionOn___redArg___lam__0), 3, 1);
lean_closure_set(v___f_354_, 0, v_singleton_350_);
v___f_355_ = lean_alloc_closure((void*)(lp_mathlib_RelSeries_inductionOn_x27___redArg___lam__1___boxed), 5, 1);
lean_closure_set(v___f_355_, 0, v_snoc_351_);
v___x_25__overap_356_ = l_Nat_recCompiled___redArg(v___f_354_, v___f_355_, v_length_353_);
lean_dec_ref(v___f_354_);
v___x_357_ = lean_apply_2(v___x_25__overap_356_, v_p_352_, lean_box(0));
return v___x_357_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_inductionOn_x27(lean_object* v_00_u03b1_358_, lean_object* v_r_359_, lean_object* v_motive_360_, lean_object* v_singleton_361_, lean_object* v_snoc_362_, lean_object* v_p_363_){
_start:
{
lean_object* v___x_364_; 
v___x_364_ = lp_mathlib_RelSeries_inductionOn_x27___redArg(v_singleton_361_, v_snoc_362_, v_p_363_);
return v___x_364_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_smash___redArg___lam__0(lean_object* v_toFun_365_, lean_object* v___y_366_){
_start:
{
lean_object* v___x_367_; 
v___x_367_ = lean_apply_1(v_toFun_365_, v___y_366_);
return v___x_367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_smash___redArg___lam__1(lean_object* v_length_368_, lean_object* v___f_369_, lean_object* v_toFun_370_, lean_object* v_i_371_){
_start:
{
lean_object* v___x_372_; 
v___x_372_ = l_Fin_addCases___redArg(v_length_368_, v___f_369_, v_toFun_370_, v_i_371_);
return v___x_372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_smash___redArg___lam__1___boxed(lean_object* v_length_373_, lean_object* v___f_374_, lean_object* v_toFun_375_, lean_object* v_i_376_){
_start:
{
lean_object* v_res_377_; 
v_res_377_ = lp_mathlib_RelSeries_smash___redArg___lam__1(v_length_373_, v___f_374_, v_toFun_375_, v_i_376_);
lean_dec(v_length_373_);
return v_res_377_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_smash___redArg(lean_object* v_p_378_, lean_object* v_q_379_){
_start:
{
lean_object* v_length_380_; lean_object* v_toFun_381_; lean_object* v_length_382_; lean_object* v_toFun_383_; lean_object* v___x_385_; uint8_t v_isShared_386_; uint8_t v_isSharedCheck_393_; 
v_length_380_ = lean_ctor_get(v_p_378_, 0);
lean_inc(v_length_380_);
v_toFun_381_ = lean_ctor_get(v_p_378_, 1);
lean_inc(v_toFun_381_);
lean_dec_ref(v_p_378_);
v_length_382_ = lean_ctor_get(v_q_379_, 0);
v_toFun_383_ = lean_ctor_get(v_q_379_, 1);
v_isSharedCheck_393_ = !lean_is_exclusive(v_q_379_);
if (v_isSharedCheck_393_ == 0)
{
v___x_385_ = v_q_379_;
v_isShared_386_ = v_isSharedCheck_393_;
goto v_resetjp_384_;
}
else
{
lean_inc(v_toFun_383_);
lean_inc(v_length_382_);
lean_dec(v_q_379_);
v___x_385_ = lean_box(0);
v_isShared_386_ = v_isSharedCheck_393_;
goto v_resetjp_384_;
}
v_resetjp_384_:
{
lean_object* v___f_387_; lean_object* v___f_388_; lean_object* v___x_389_; lean_object* v___x_391_; 
v___f_387_ = lean_alloc_closure((void*)(lp_mathlib_RelSeries_smash___redArg___lam__0), 2, 1);
lean_closure_set(v___f_387_, 0, v_toFun_381_);
lean_inc(v_length_380_);
v___f_388_ = lean_alloc_closure((void*)(lp_mathlib_RelSeries_smash___redArg___lam__1___boxed), 4, 3);
lean_closure_set(v___f_388_, 0, v_length_380_);
lean_closure_set(v___f_388_, 1, v___f_387_);
lean_closure_set(v___f_388_, 2, v_toFun_383_);
v___x_389_ = lean_nat_add(v_length_380_, v_length_382_);
lean_dec(v_length_382_);
lean_dec(v_length_380_);
if (v_isShared_386_ == 0)
{
lean_ctor_set(v___x_385_, 1, v___f_388_);
lean_ctor_set(v___x_385_, 0, v___x_389_);
v___x_391_ = v___x_385_;
goto v_reusejp_390_;
}
else
{
lean_object* v_reuseFailAlloc_392_; 
v_reuseFailAlloc_392_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_392_, 0, v___x_389_);
lean_ctor_set(v_reuseFailAlloc_392_, 1, v___f_388_);
v___x_391_ = v_reuseFailAlloc_392_;
goto v_reusejp_390_;
}
v_reusejp_390_:
{
return v___x_391_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_smash(lean_object* v_00_u03b1_394_, lean_object* v_r_395_, lean_object* v_p_396_, lean_object* v_q_397_, lean_object* v_connect_398_){
_start:
{
lean_object* v___x_399_; 
v___x_399_ = lp_mathlib_RelSeries_smash___redArg(v_p_396_, v_q_397_);
return v___x_399_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_take___redArg___lam__0(lean_object* v_p_400_, lean_object* v_x_401_){
_start:
{
lean_object* v_toFun_402_; lean_object* v___x_403_; 
v_toFun_402_ = lean_ctor_get(v_p_400_, 1);
lean_inc(v_toFun_402_);
lean_dec_ref(v_p_400_);
v___x_403_ = lean_apply_1(v_toFun_402_, v_x_401_);
return v___x_403_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_take___redArg(lean_object* v_p_404_, lean_object* v_i_405_){
_start:
{
lean_object* v___f_406_; lean_object* v___x_407_; 
v___f_406_ = lean_alloc_closure((void*)(lp_mathlib_RelSeries_take___redArg___lam__0), 2, 1);
lean_closure_set(v___f_406_, 0, v_p_404_);
v___x_407_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_407_, 0, v_i_405_);
lean_ctor_set(v___x_407_, 1, v___f_406_);
return v___x_407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_take(lean_object* v_00_u03b1_408_, lean_object* v_r_409_, lean_object* v_p_410_, lean_object* v_i_411_){
_start:
{
lean_object* v___x_412_; 
v___x_412_ = lp_mathlib_RelSeries_take___redArg(v_p_410_, v_i_411_);
return v___x_412_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_drop___redArg___lam__0(lean_object* v_i_413_, lean_object* v_toFun_414_, lean_object* v_x_415_){
_start:
{
lean_object* v___x_416_; lean_object* v___x_417_; 
v___x_416_ = lean_nat_add(v_x_415_, v_i_413_);
v___x_417_ = lean_apply_1(v_toFun_414_, v___x_416_);
return v___x_417_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_drop___redArg___lam__0___boxed(lean_object* v_i_418_, lean_object* v_toFun_419_, lean_object* v_x_420_){
_start:
{
lean_object* v_res_421_; 
v_res_421_ = lp_mathlib_RelSeries_drop___redArg___lam__0(v_i_418_, v_toFun_419_, v_x_420_);
lean_dec(v_x_420_);
lean_dec(v_i_418_);
return v_res_421_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_drop___redArg(lean_object* v_p_422_, lean_object* v_i_423_){
_start:
{
lean_object* v_length_424_; lean_object* v_toFun_425_; lean_object* v___x_427_; uint8_t v_isShared_428_; uint8_t v_isSharedCheck_434_; 
v_length_424_ = lean_ctor_get(v_p_422_, 0);
v_toFun_425_ = lean_ctor_get(v_p_422_, 1);
v_isSharedCheck_434_ = !lean_is_exclusive(v_p_422_);
if (v_isSharedCheck_434_ == 0)
{
v___x_427_ = v_p_422_;
v_isShared_428_ = v_isSharedCheck_434_;
goto v_resetjp_426_;
}
else
{
lean_inc(v_toFun_425_);
lean_inc(v_length_424_);
lean_dec(v_p_422_);
v___x_427_ = lean_box(0);
v_isShared_428_ = v_isSharedCheck_434_;
goto v_resetjp_426_;
}
v_resetjp_426_:
{
lean_object* v___f_429_; lean_object* v___x_430_; lean_object* v___x_432_; 
lean_inc(v_i_423_);
v___f_429_ = lean_alloc_closure((void*)(lp_mathlib_RelSeries_drop___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_429_, 0, v_i_423_);
lean_closure_set(v___f_429_, 1, v_toFun_425_);
v___x_430_ = lean_nat_sub(v_length_424_, v_i_423_);
lean_dec(v_i_423_);
lean_dec(v_length_424_);
if (v_isShared_428_ == 0)
{
lean_ctor_set(v___x_427_, 1, v___f_429_);
lean_ctor_set(v___x_427_, 0, v___x_430_);
v___x_432_ = v___x_427_;
goto v_reusejp_431_;
}
else
{
lean_object* v_reuseFailAlloc_433_; 
v_reuseFailAlloc_433_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_433_, 0, v___x_430_);
lean_ctor_set(v_reuseFailAlloc_433_, 1, v___f_429_);
v___x_432_ = v_reuseFailAlloc_433_;
goto v_reusejp_431_;
}
v_reusejp_431_:
{
return v___x_432_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelSeries_drop(lean_object* v_00_u03b1_435_, lean_object* v_r_436_, lean_object* v_p_437_, lean_object* v_i_438_){
_start:
{
lean_object* v___x_439_; 
v___x_439_ = lp_mathlib_RelSeries_drop___redArg(v_p_437_, v_i_438_);
return v___x_439_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_mk___redArg(lean_object* v_length_440_, lean_object* v_toFun_441_){
_start:
{
lean_object* v___x_442_; 
v___x_442_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_442_, 0, v_length_440_);
lean_ctor_set(v___x_442_, 1, v_toFun_441_);
return v___x_442_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_mk(lean_object* v_00_u03b1_443_, lean_object* v_inst_444_, lean_object* v_length_445_, lean_object* v_toFun_446_, lean_object* v_strictMono_447_){
_start:
{
lean_object* v___x_448_; 
v___x_448_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_448_, 0, v_length_445_);
lean_ctor_set(v___x_448_, 1, v_toFun_446_);
return v___x_448_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_mk___boxed(lean_object* v_00_u03b1_449_, lean_object* v_inst_450_, lean_object* v_length_451_, lean_object* v_toFun_452_, lean_object* v_strictMono_453_){
_start:
{
lean_object* v_res_454_; 
v_res_454_ = lp_mathlib_LTSeries_mk(v_00_u03b1_449_, v_inst_450_, v_length_451_, v_toFun_452_, v_strictMono_453_);
lean_dec_ref(v_inst_450_);
return v_res_454_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_injStrictMono___lam__0(lean_object* v_f_455_){
_start:
{
lean_object* v_fst_456_; lean_object* v_snd_457_; lean_object* v___x_459_; uint8_t v_isShared_460_; uint8_t v_isSharedCheck_464_; 
v_fst_456_ = lean_ctor_get(v_f_455_, 0);
v_snd_457_ = lean_ctor_get(v_f_455_, 1);
v_isSharedCheck_464_ = !lean_is_exclusive(v_f_455_);
if (v_isSharedCheck_464_ == 0)
{
v___x_459_ = v_f_455_;
v_isShared_460_ = v_isSharedCheck_464_;
goto v_resetjp_458_;
}
else
{
lean_inc(v_snd_457_);
lean_inc(v_fst_456_);
lean_dec(v_f_455_);
v___x_459_ = lean_box(0);
v_isShared_460_ = v_isSharedCheck_464_;
goto v_resetjp_458_;
}
v_resetjp_458_:
{
lean_object* v___x_462_; 
if (v_isShared_460_ == 0)
{
v___x_462_ = v___x_459_;
goto v_reusejp_461_;
}
else
{
lean_object* v_reuseFailAlloc_463_; 
v_reuseFailAlloc_463_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_463_, 0, v_fst_456_);
lean_ctor_set(v_reuseFailAlloc_463_, 1, v_snd_457_);
v___x_462_ = v_reuseFailAlloc_463_;
goto v_reusejp_461_;
}
v_reusejp_461_:
{
return v___x_462_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_injStrictMono(lean_object* v_00_u03b1_466_, lean_object* v_inst_467_, lean_object* v_n_468_){
_start:
{
lean_object* v___f_469_; 
v___f_469_ = ((lean_object*)(lp_mathlib_LTSeries_injStrictMono___closed__0));
return v___f_469_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_injStrictMono___boxed(lean_object* v_00_u03b1_470_, lean_object* v_inst_471_, lean_object* v_n_472_){
_start:
{
lean_object* v_res_473_; 
v_res_473_ = lp_mathlib_LTSeries_injStrictMono(v_00_u03b1_470_, v_inst_471_, v_n_472_);
lean_dec(v_n_472_);
lean_dec_ref(v_inst_471_);
return v_res_473_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_map___redArg(lean_object* v_p_474_, lean_object* v_f_475_){
_start:
{
lean_object* v_length_476_; lean_object* v_toFun_477_; lean_object* v___x_479_; uint8_t v_isShared_480_; uint8_t v_isSharedCheck_485_; 
v_length_476_ = lean_ctor_get(v_p_474_, 0);
v_toFun_477_ = lean_ctor_get(v_p_474_, 1);
v_isSharedCheck_485_ = !lean_is_exclusive(v_p_474_);
if (v_isSharedCheck_485_ == 0)
{
v___x_479_ = v_p_474_;
v_isShared_480_ = v_isSharedCheck_485_;
goto v_resetjp_478_;
}
else
{
lean_inc(v_toFun_477_);
lean_inc(v_length_476_);
lean_dec(v_p_474_);
v___x_479_ = lean_box(0);
v_isShared_480_ = v_isSharedCheck_485_;
goto v_resetjp_478_;
}
v_resetjp_478_:
{
lean_object* v___x_481_; lean_object* v___x_483_; 
v___x_481_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_481_, 0, lean_box(0));
lean_closure_set(v___x_481_, 1, lean_box(0));
lean_closure_set(v___x_481_, 2, lean_box(0));
lean_closure_set(v___x_481_, 3, v_f_475_);
lean_closure_set(v___x_481_, 4, v_toFun_477_);
if (v_isShared_480_ == 0)
{
lean_ctor_set(v___x_479_, 1, v___x_481_);
v___x_483_ = v___x_479_;
goto v_reusejp_482_;
}
else
{
lean_object* v_reuseFailAlloc_484_; 
v_reuseFailAlloc_484_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_484_, 0, v_length_476_);
lean_ctor_set(v_reuseFailAlloc_484_, 1, v___x_481_);
v___x_483_ = v_reuseFailAlloc_484_;
goto v_reusejp_482_;
}
v_reusejp_482_:
{
return v___x_483_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_map(lean_object* v_00_u03b1_486_, lean_object* v_00_u03b2_487_, lean_object* v_inst_488_, lean_object* v_inst_489_, lean_object* v_p_490_, lean_object* v_f_491_, lean_object* v_hf_492_){
_start:
{
lean_object* v___x_493_; 
v___x_493_ = lp_mathlib_LTSeries_map___redArg(v_p_490_, v_f_491_);
return v___x_493_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_map___boxed(lean_object* v_00_u03b1_494_, lean_object* v_00_u03b2_495_, lean_object* v_inst_496_, lean_object* v_inst_497_, lean_object* v_p_498_, lean_object* v_f_499_, lean_object* v_hf_500_){
_start:
{
lean_object* v_res_501_; 
v_res_501_ = lp_mathlib_LTSeries_map(v_00_u03b1_494_, v_00_u03b2_495_, v_inst_496_, v_inst_497_, v_p_498_, v_f_499_, v_hf_500_);
lean_dec_ref(v_inst_497_);
lean_dec_ref(v_inst_496_);
return v_res_501_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_range___lam__0(lean_object* v_i_502_){
_start:
{
lean_inc(v_i_502_);
return v_i_502_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_range___lam__0___boxed(lean_object* v_i_503_){
_start:
{
lean_object* v_res_504_; 
v_res_504_ = lp_mathlib_LTSeries_range___lam__0(v_i_503_);
lean_dec(v_i_503_);
return v_res_504_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_range(lean_object* v_n_506_){
_start:
{
lean_object* v___f_507_; lean_object* v___x_508_; 
v___f_507_ = ((lean_object*)(lp_mathlib_LTSeries_range___closed__0));
v___x_508_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_508_, 0, v_n_506_);
lean_ctor_set(v___x_508_, 1, v___f_507_);
return v___x_508_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__0(lean_object* v_a_509_, lean_object* v_snd_510_, lean_object* v_inst_511_, lean_object* v_a_512_){
_start:
{
uint8_t v___x_513_; 
v___x_513_ = lean_nat_dec_lt(v_a_509_, v_a_512_);
if (v___x_513_ == 0)
{
uint8_t v___x_514_; 
lean_dec(v_a_512_);
lean_dec_ref(v_inst_511_);
lean_dec(v_snd_510_);
lean_dec(v_a_509_);
v___x_514_ = 1;
return v___x_514_;
}
else
{
lean_object* v___x_515_; lean_object* v___x_516_; lean_object* v___x_517_; uint8_t v___x_518_; 
lean_inc(v_snd_510_);
v___x_515_ = lean_apply_1(v_snd_510_, v_a_509_);
v___x_516_ = lean_apply_1(v_snd_510_, v_a_512_);
v___x_517_ = lean_apply_2(v_inst_511_, v___x_515_, v___x_516_);
v___x_518_ = lean_unbox(v___x_517_);
return v___x_518_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__0___boxed(lean_object* v_a_519_, lean_object* v_snd_520_, lean_object* v_inst_521_, lean_object* v_a_522_){
_start:
{
uint8_t v_res_523_; lean_object* v_r_524_; 
v_res_523_ = lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__0(v_a_519_, v_snd_520_, v_inst_521_, v_a_522_);
v_r_524_ = lean_box(v_res_523_);
return v_r_524_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__1(lean_object* v_snd_525_, lean_object* v_inst_526_, lean_object* v___x_527_, lean_object* v_a_528_){
_start:
{
lean_object* v___f_529_; uint8_t v___x_530_; 
v___f_529_ = lean_alloc_closure((void*)(lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_529_, 0, v_a_528_);
lean_closure_set(v___f_529_, 1, v_snd_525_);
lean_closure_set(v___f_529_, 2, v_inst_526_);
v___x_530_ = l_Nat_decidableForallFin___redArg(v___x_527_, v___f_529_);
return v___x_530_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__1___boxed(lean_object* v_snd_531_, lean_object* v_inst_532_, lean_object* v___x_533_, lean_object* v_a_534_){
_start:
{
uint8_t v_res_535_; lean_object* v_r_536_; 
v_res_535_ = lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__1(v_snd_531_, v_inst_532_, v___x_533_, v_a_534_);
v_r_536_ = lean_box(v_res_535_);
return v_r_536_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__2(lean_object* v_inst_537_, lean_object* v_a_538_){
_start:
{
lean_object* v_fst_539_; lean_object* v_snd_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___f_543_; uint8_t v___x_544_; 
v_fst_539_ = lean_ctor_get(v_a_538_, 0);
lean_inc(v_fst_539_);
v_snd_540_ = lean_ctor_get(v_a_538_, 1);
lean_inc(v_snd_540_);
lean_dec_ref(v_a_538_);
v___x_541_ = lean_unsigned_to_nat(1u);
v___x_542_ = lean_nat_add(v_fst_539_, v___x_541_);
lean_dec(v_fst_539_);
lean_inc(v___x_542_);
v___f_543_ = lean_alloc_closure((void*)(lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__1___boxed), 4, 3);
lean_closure_set(v___f_543_, 0, v_snd_540_);
lean_closure_set(v___f_543_, 1, v_inst_537_);
lean_closure_set(v___f_543_, 2, v___x_542_);
v___x_544_ = l_Nat_decidableForallFin___redArg(v___x_542_, v___f_543_);
return v___x_544_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__2___boxed(lean_object* v_inst_545_, lean_object* v_a_546_){
_start:
{
uint8_t v_res_547_; lean_object* v_r_548_; 
v_res_547_ = lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__2(v_inst_545_, v_a_546_);
v_r_548_ = lean_box(v_res_547_);
return v_r_548_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__3(lean_object* v_inst_549_, lean_object* v_x_550_){
_start:
{
lean_inc(v_inst_549_);
return v_inst_549_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__3___boxed(lean_object* v_inst_551_, lean_object* v_x_552_){
_start:
{
lean_object* v_res_553_; 
v_res_553_ = lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__3(v_inst_551_, v_x_552_);
lean_dec(v_x_552_);
lean_dec(v_inst_551_);
return v_res_553_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__4(lean_object* v___f_554_, lean_object* v_x_555_){
_start:
{
lean_object* v___x_556_; lean_object* v___x_557_; lean_object* v___x_558_; lean_object* v___x_559_; lean_object* v___x_560_; 
v___x_556_ = lean_unsigned_to_nat(1u);
v___x_557_ = lean_nat_add(v_x_555_, v___x_556_);
lean_inc(v___x_557_);
v___x_558_ = lean_alloc_closure((void*)(l_instDecidableEqFin___boxed), 3, 1);
lean_closure_set(v___x_558_, 0, v___x_557_);
v___x_559_ = l_List_finRange(v___x_557_);
v___x_560_ = lp_mathlib_Fintype_piFinset___redArg(v___x_558_, v___x_559_, v___f_554_);
return v___x_560_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__4___boxed(lean_object* v___f_561_, lean_object* v_x_562_){
_start:
{
lean_object* v_res_563_; 
v_res_563_ = lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__4(v___f_561_, v_x_562_);
lean_dec(v_x_562_);
return v_res_563_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__5(lean_object* v_val_564_, lean_object* v_property_565_){
_start:
{
lean_inc_ref(v_val_564_);
return v_val_564_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__5___boxed(lean_object* v_val_566_, lean_object* v_property_567_){
_start:
{
lean_object* v_res_568_; 
v_res_568_ = lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__5(v_val_566_, v_property_567_);
lean_dec_ref(v_val_566_);
return v_res_568_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg(lean_object* v_inst_570_, lean_object* v_inst_571_){
_start:
{
lean_object* v___f_572_; lean_object* v___f_573_; lean_object* v___f_574_; lean_object* v___f_575_; lean_object* v___x_576_; lean_object* v___f_577_; lean_object* v___x_578_; lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; 
v___f_572_ = lean_alloc_closure((void*)(lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__2___boxed), 2, 1);
lean_closure_set(v___f_572_, 0, v_inst_571_);
lean_inc(v_inst_570_);
v___f_573_ = lean_alloc_closure((void*)(lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__3___boxed), 2, 1);
lean_closure_set(v___f_573_, 0, v_inst_570_);
v___f_574_ = lean_alloc_closure((void*)(lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___lam__4___boxed), 2, 1);
lean_closure_set(v___f_574_, 0, v___f_573_);
v___f_575_ = ((lean_object*)(lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg___closed__0));
v___x_576_ = l_List_lengthTR___redArg(v_inst_570_);
lean_dec(v_inst_570_);
v___f_577_ = ((lean_object*)(lp_mathlib_LTSeries_injStrictMono___closed__0));
v___x_578_ = l_List_finRange(v___x_576_);
v___x_579_ = lp_mathlib_Finset_sigma___redArg(v___x_578_, v___f_574_);
v___x_580_ = lp_mathlib_Multiset_filter___redArg(v___f_572_, v___x_579_);
v___x_581_ = lp_mathlib_Multiset_pmap___redArg(v___f_575_, v___x_580_);
v___x_582_ = lp_mathlib_Finset_map___redArg(v___f_577_, v___x_581_);
return v___x_582_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT(lean_object* v_00_u03b1_583_, lean_object* v_inst_584_, lean_object* v_inst_585_, lean_object* v_inst_586_){
_start:
{
lean_object* v___x_587_; 
v___x_587_ = lp_mathlib_LTSeries_instFintypeOfDecidableLT___redArg(v_inst_585_, v_inst_586_);
return v___x_587_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LTSeries_instFintypeOfDecidableLT___boxed(lean_object* v_00_u03b1_588_, lean_object* v_inst_589_, lean_object* v_inst_590_, lean_object* v_inst_591_){
_start:
{
lean_object* v_res_592_; 
v_res_592_ = lp_mathlib_LTSeries_instFintypeOfDecidableLT(v_00_u03b1_588_, v_inst_589_, v_inst_590_, v_inst_591_);
lean_dec_ref(v_inst_589_);
return v_res_592_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_NatCast(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_Rel(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fin_VecNotation(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Pi(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Pigeonhole(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Sigma(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_OrderIsoNat(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_RelSeries(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_NatCast(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Rel(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fin_VecNotation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Pigeonhole(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Sigma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_OrderIsoNat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_RelSeries(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Monoid_NatCast(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Basic_Rel(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fin_VecNotation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Pigeonhole(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Sigma(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_OrderIsoNat(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_RelSeries(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Monoid_NatCast(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_Rel(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fin_VecNotation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Pigeonhole(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Sigma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_OrderIsoNat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_RelSeries(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_RelSeries(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_RelSeries(builtin);
}
#ifdef __cplusplus
}
#endif
