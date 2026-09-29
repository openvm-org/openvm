// Lean compiler output
// Module: Aesop.Rule.Forward
// Imports: public import Init public meta import Init public import Aesop.Forward.RuleInfo public import Aesop.Percent public import Aesop.RuleTac.RuleTerm
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
uint8_t lean_int_dec_lt(lean_object*, lean_object*);
uint8_t lean_int_dec_eq(lean_object*, lean_object*);
uint8_t lean_float_decLt(double, double);
double lean_float_sub(double, double);
double l_Float_ofScientific(lean_object*, uint8_t, lean_object*);
uint8_t lp_aesop_Aesop_RuleName_compare(lean_object*, lean_object*);
lean_object* l_Int_repr(lean_object*);
lean_object* lp_aesop_Aesop_Percent_toHumanString(double);
lean_object* lean_nat_to_int(lean_object*);
uint8_t lean_uint64_dec_eq(uint64_t, uint64_t);
uint8_t lp_aesop_Aesop_instBEqBuilderName_beq(uint8_t, uint8_t);
uint8_t lp_aesop_Aesop_instBEqPhaseName_beq(uint8_t, uint8_t);
uint8_t lp_aesop_Aesop_instBEqScopeName_beq(uint8_t, uint8_t);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
extern lean_object* lp_aesop_Aesop_instInhabitedRuleTerm_default;
extern lean_object* lp_aesop_Aesop_instInhabitedRuleName_default;
extern lean_object* lp_aesop_Aesop_instInhabitedForwardRuleInfo_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRulePriority_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRulePriority_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRulePriority_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRulePriority_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRulePriority_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRulePriority_normSafe_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRulePriority_normSafe_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRulePriority_unsafe_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRulePriority_unsafe_elim(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_instInhabitedForwardRulePriority_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedForwardRulePriority_default___closed__0;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedForwardRulePriority_default___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedForwardRulePriority_default___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedForwardRulePriority_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedForwardRulePriority;
static lean_once_cell_t lp_aesop_Aesop_instBEqForwardRulePriority_beq___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop_Aesop_instBEqForwardRulePriority_beq___closed__0;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqForwardRulePriority_beq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqForwardRulePriority_beq___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_instBEqForwardRulePriority___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instBEqForwardRulePriority_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instBEqForwardRulePriority___closed__0 = (const lean_object*)&lp_aesop_Aesop_instBEqForwardRulePriority___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instBEqForwardRulePriority = (const lean_object*)&lp_aesop_Aesop_instBEqForwardRulePriority___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRulePriority_penalty_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRulePriority_successProbability_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRulePriority_successProbability_x3f___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_ForwardRulePriority_compare(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRulePriority_compare___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_ForwardRulePriority_instOrd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_ForwardRulePriority_compare___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_ForwardRulePriority_instOrd___closed__0 = (const lean_object*)&lp_aesop_Aesop_ForwardRulePriority_instOrd___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_ForwardRulePriority_instOrd = (const lean_object*)&lp_aesop_Aesop_ForwardRulePriority_instOrd___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRulePriority_instToString___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRulePriority_instToString___lam__0___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_ForwardRulePriority_instToString___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_ForwardRulePriority_instToString___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_ForwardRulePriority_instToString___closed__0 = (const lean_object*)&lp_aesop_Aesop_ForwardRulePriority_instToString___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_ForwardRulePriority_instToString = (const lean_object*)&lp_aesop_Aesop_ForwardRulePriority_instToString___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedForwardRule_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedForwardRule_default___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedForwardRule_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedForwardRule;
LEAN_EXPORT uint8_t lp_aesop_Aesop_ForwardRule_instBEq___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRule_instBEq___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_ForwardRule_instBEq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_ForwardRule_instBEq___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_ForwardRule_instBEq___closed__0 = (const lean_object*)&lp_aesop_Aesop_ForwardRule_instBEq___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_ForwardRule_instBEq = (const lean_object*)&lp_aesop_Aesop_ForwardRule_instBEq___closed__0_value;
LEAN_EXPORT uint64_t lp_aesop_Aesop_ForwardRule_instHashable___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRule_instHashable___lam__0___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_ForwardRule_instHashable___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_ForwardRule_instHashable___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_ForwardRule_instHashable___closed__0 = (const lean_object*)&lp_aesop_Aesop_ForwardRule_instHashable___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_ForwardRule_instHashable = (const lean_object*)&lp_aesop_Aesop_ForwardRule_instHashable___closed__0_value;
LEAN_EXPORT uint8_t lp_aesop_Aesop_ForwardRule_instOrd___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRule_instOrd___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_ForwardRule_instOrd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_ForwardRule_instOrd___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_ForwardRule_instOrd___closed__0 = (const lean_object*)&lp_aesop_Aesop_ForwardRule_instOrd___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_ForwardRule_instOrd = (const lean_object*)&lp_aesop_Aesop_ForwardRule_instOrd___closed__0_value;
static const lean_string_object lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "global"};
static const lean_object* lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__0_value;
static const lean_string_object lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "local"};
static const lean_object* lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__1 = (const lean_object*)&lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__1_value;
static const lean_string_object lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__2 = (const lean_object*)&lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__2_value;
static const lean_string_object lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "apply"};
static const lean_object* lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__3 = (const lean_object*)&lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__3_value;
static const lean_string_object lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "cases"};
static const lean_object* lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__4 = (const lean_object*)&lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__4_value;
static const lean_string_object lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "constructors"};
static const lean_object* lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__5 = (const lean_object*)&lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__5_value;
static const lean_string_object lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "destruct"};
static const lean_object* lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__6 = (const lean_object*)&lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__6_value;
static const lean_string_object lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "forward"};
static const lean_object* lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__7 = (const lean_object*)&lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__7_value;
static const lean_string_object lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__8 = (const lean_object*)&lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__8_value;
static const lean_string_object lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__9 = (const lean_object*)&lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__9_value;
static const lean_string_object lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unfold"};
static const lean_object* lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__10 = (const lean_object*)&lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__10_value;
static const lean_string_object lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__11 = (const lean_object*)&lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__11_value;
static const lean_string_object lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "] "};
static const lean_object* lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__12 = (const lean_object*)&lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__12_value;
static const lean_string_object lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "norm"};
static const lean_object* lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__13 = (const lean_object*)&lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__13_value;
static const lean_string_object lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "safe"};
static const lean_object* lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__14 = (const lean_object*)&lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__14_value;
static const lean_string_object lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unsafe"};
static const lean_object* lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__15 = (const lean_object*)&lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__15_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRule_instToString___lam__0(lean_object*);
static const lean_closure_object lp_aesop_Aesop_ForwardRule_instToString___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_ForwardRule_instToString___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_ForwardRule_instToString___closed__0 = (const lean_object*)&lp_aesop_Aesop_ForwardRule_instToString___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_ForwardRule_instToString = (const lean_object*)&lp_aesop_Aesop_ForwardRule_instToString___closed__0_value;
LEAN_EXPORT uint8_t lp_aesop_Aesop_ForwardRule_destruct(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRule_destruct___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRulePriority_ctorIdx(lean_object* v_x_1_){
_start:
{
if (lean_obj_tag(v_x_1_) == 0)
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(0u);
return v___x_2_;
}
else
{
lean_object* v___x_3_; 
v___x_3_ = lean_unsigned_to_nat(1u);
return v___x_3_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRulePriority_ctorIdx___boxed(lean_object* v_x_4_){
_start:
{
lean_object* v_res_5_; 
v_res_5_ = lp_aesop_Aesop_ForwardRulePriority_ctorIdx(v_x_4_);
lean_dec_ref(v_x_4_);
return v_res_5_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRulePriority_ctorElim___redArg(lean_object* v_t_6_, lean_object* v_k_7_){
_start:
{
if (lean_obj_tag(v_t_6_) == 0)
{
lean_object* v_n_8_; lean_object* v___x_9_; 
v_n_8_ = lean_ctor_get(v_t_6_, 0);
lean_inc(v_n_8_);
lean_dec_ref_known(v_t_6_, 1);
v___x_9_ = lean_apply_1(v_k_7_, v_n_8_);
return v___x_9_;
}
else
{
double v_p_10_; lean_object* v___x_11_; lean_object* v___x_12_; 
v_p_10_ = lean_ctor_get_float(v_t_6_, 0);
lean_dec_ref_known(v_t_6_, 0);
v___x_11_ = lean_box_float(v_p_10_);
v___x_12_ = lean_apply_1(v_k_7_, v___x_11_);
return v___x_12_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRulePriority_ctorElim(lean_object* v_motive_13_, lean_object* v_ctorIdx_14_, lean_object* v_t_15_, lean_object* v_h_16_, lean_object* v_k_17_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = lp_aesop_Aesop_ForwardRulePriority_ctorElim___redArg(v_t_15_, v_k_17_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRulePriority_ctorElim___boxed(lean_object* v_motive_19_, lean_object* v_ctorIdx_20_, lean_object* v_t_21_, lean_object* v_h_22_, lean_object* v_k_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_aesop_Aesop_ForwardRulePriority_ctorElim(v_motive_19_, v_ctorIdx_20_, v_t_21_, v_h_22_, v_k_23_);
lean_dec(v_ctorIdx_20_);
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRulePriority_normSafe_elim___redArg(lean_object* v_t_25_, lean_object* v_normSafe_26_){
_start:
{
lean_object* v___x_27_; 
v___x_27_ = lp_aesop_Aesop_ForwardRulePriority_ctorElim___redArg(v_t_25_, v_normSafe_26_);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRulePriority_normSafe_elim(lean_object* v_motive_28_, lean_object* v_t_29_, lean_object* v_h_30_, lean_object* v_normSafe_31_){
_start:
{
lean_object* v___x_32_; 
v___x_32_ = lp_aesop_Aesop_ForwardRulePriority_ctorElim___redArg(v_t_29_, v_normSafe_31_);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRulePriority_unsafe_elim___redArg(lean_object* v_t_33_, lean_object* v_unsafe_34_){
_start:
{
lean_object* v___x_35_; 
v___x_35_ = lp_aesop_Aesop_ForwardRulePriority_ctorElim___redArg(v_t_33_, v_unsafe_34_);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRulePriority_unsafe_elim(lean_object* v_motive_36_, lean_object* v_t_37_, lean_object* v_h_38_, lean_object* v_unsafe_39_){
_start:
{
lean_object* v___x_40_; 
v___x_40_ = lp_aesop_Aesop_ForwardRulePriority_ctorElim___redArg(v_t_37_, v_unsafe_39_);
return v___x_40_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedForwardRulePriority_default___closed__0(void){
_start:
{
lean_object* v___x_41_; lean_object* v___x_42_; 
v___x_41_ = lean_unsigned_to_nat(0u);
v___x_42_ = lean_nat_to_int(v___x_41_);
return v___x_42_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedForwardRulePriority_default___closed__1(void){
_start:
{
lean_object* v___x_43_; lean_object* v___x_44_; 
v___x_43_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedForwardRulePriority_default___closed__0, &lp_aesop_Aesop_instInhabitedForwardRulePriority_default___closed__0_once, _init_lp_aesop_Aesop_instInhabitedForwardRulePriority_default___closed__0);
v___x_44_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_44_, 0, v___x_43_);
return v___x_44_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedForwardRulePriority_default(void){
_start:
{
lean_object* v___x_45_; 
v___x_45_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedForwardRulePriority_default___closed__1, &lp_aesop_Aesop_instInhabitedForwardRulePriority_default___closed__1_once, _init_lp_aesop_Aesop_instInhabitedForwardRulePriority_default___closed__1);
return v___x_45_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedForwardRulePriority(void){
_start:
{
lean_object* v___x_46_; 
v___x_46_ = lp_aesop_Aesop_instInhabitedForwardRulePriority_default;
return v___x_46_;
}
}
static double _init_lp_aesop_Aesop_instBEqForwardRulePriority_beq___closed__0(void){
_start:
{
lean_object* v___x_47_; uint8_t v___x_48_; lean_object* v___x_49_; double v___x_50_; 
v___x_47_ = lean_unsigned_to_nat(5u);
v___x_48_ = 1;
v___x_49_ = lean_unsigned_to_nat(1u);
v___x_50_ = l_Float_ofScientific(v___x_49_, v___x_48_, v___x_47_);
return v___x_50_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqForwardRulePriority_beq(lean_object* v_x_51_, lean_object* v_x_52_){
_start:
{
if (lean_obj_tag(v_x_51_) == 0)
{
if (lean_obj_tag(v_x_52_) == 0)
{
lean_object* v_n_53_; lean_object* v_n_54_; uint8_t v___x_55_; 
v_n_53_ = lean_ctor_get(v_x_51_, 0);
v_n_54_ = lean_ctor_get(v_x_52_, 0);
v___x_55_ = lean_int_dec_eq(v_n_53_, v_n_54_);
return v___x_55_;
}
else
{
uint8_t v___x_56_; 
v___x_56_ = 0;
return v___x_56_;
}
}
else
{
if (lean_obj_tag(v_x_52_) == 1)
{
double v_p_57_; double v_p_58_; uint8_t v___x_59_; 
v_p_57_ = lean_ctor_get_float(v_x_51_, 0);
v_p_58_ = lean_ctor_get_float(v_x_52_, 0);
v___x_59_ = lean_float_decLt(v_p_58_, v_p_57_);
if (v___x_59_ == 0)
{
double v___x_60_; double v___x_61_; uint8_t v___x_62_; 
v___x_60_ = lean_float_sub(v_p_58_, v_p_57_);
v___x_61_ = lean_float_once(&lp_aesop_Aesop_instBEqForwardRulePriority_beq___closed__0, &lp_aesop_Aesop_instBEqForwardRulePriority_beq___closed__0_once, _init_lp_aesop_Aesop_instBEqForwardRulePriority_beq___closed__0);
v___x_62_ = lean_float_decLt(v___x_60_, v___x_61_);
return v___x_62_;
}
else
{
double v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; double v___x_66_; uint8_t v___x_67_; 
v___x_63_ = lean_float_sub(v_p_57_, v_p_58_);
v___x_64_ = lean_unsigned_to_nat(1u);
v___x_65_ = lean_unsigned_to_nat(5u);
v___x_66_ = l_Float_ofScientific(v___x_64_, v___x_59_, v___x_65_);
v___x_67_ = lean_float_decLt(v___x_63_, v___x_66_);
return v___x_67_;
}
}
else
{
uint8_t v___x_68_; 
v___x_68_ = 0;
return v___x_68_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqForwardRulePriority_beq___boxed(lean_object* v_x_69_, lean_object* v_x_70_){
_start:
{
uint8_t v_res_71_; lean_object* v_r_72_; 
v_res_71_ = lp_aesop_Aesop_instBEqForwardRulePriority_beq(v_x_69_, v_x_70_);
lean_dec_ref(v_x_70_);
lean_dec_ref(v_x_69_);
v_r_72_ = lean_box(v_res_71_);
return v_r_72_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRulePriority_penalty_x3f(lean_object* v_x_75_){
_start:
{
if (lean_obj_tag(v_x_75_) == 0)
{
lean_object* v_n_76_; lean_object* v___x_78_; uint8_t v_isShared_79_; uint8_t v_isSharedCheck_83_; 
v_n_76_ = lean_ctor_get(v_x_75_, 0);
v_isSharedCheck_83_ = !lean_is_exclusive(v_x_75_);
if (v_isSharedCheck_83_ == 0)
{
v___x_78_ = v_x_75_;
v_isShared_79_ = v_isSharedCheck_83_;
goto v_resetjp_77_;
}
else
{
lean_inc(v_n_76_);
lean_dec(v_x_75_);
v___x_78_ = lean_box(0);
v_isShared_79_ = v_isSharedCheck_83_;
goto v_resetjp_77_;
}
v_resetjp_77_:
{
lean_object* v___x_81_; 
if (v_isShared_79_ == 0)
{
lean_ctor_set_tag(v___x_78_, 1);
v___x_81_ = v___x_78_;
goto v_reusejp_80_;
}
else
{
lean_object* v_reuseFailAlloc_82_; 
v_reuseFailAlloc_82_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_82_, 0, v_n_76_);
v___x_81_ = v_reuseFailAlloc_82_;
goto v_reusejp_80_;
}
v_reusejp_80_:
{
return v___x_81_;
}
}
}
else
{
lean_object* v___x_84_; 
lean_dec_ref_known(v_x_75_, 0);
v___x_84_ = lean_box(0);
return v___x_84_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRulePriority_successProbability_x3f(lean_object* v_x_85_){
_start:
{
if (lean_obj_tag(v_x_85_) == 0)
{
lean_object* v___x_86_; 
v___x_86_ = lean_box(0);
return v___x_86_;
}
else
{
double v_p_87_; lean_object* v___x_88_; lean_object* v___x_89_; 
v_p_87_ = lean_ctor_get_float(v_x_85_, 0);
v___x_88_ = lean_box_float(v_p_87_);
v___x_89_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_89_, 0, v___x_88_);
return v___x_89_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRulePriority_successProbability_x3f___boxed(lean_object* v_x_90_){
_start:
{
lean_object* v_res_91_; 
v_res_91_ = lp_aesop_Aesop_ForwardRulePriority_successProbability_x3f(v_x_90_);
lean_dec_ref(v_x_90_);
return v_res_91_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_ForwardRulePriority_compare(lean_object* v_x_92_, lean_object* v_x_93_){
_start:
{
if (lean_obj_tag(v_x_92_) == 0)
{
if (lean_obj_tag(v_x_93_) == 0)
{
lean_object* v_n_94_; lean_object* v_n_95_; uint8_t v___x_96_; 
v_n_94_ = lean_ctor_get(v_x_92_, 0);
v_n_95_ = lean_ctor_get(v_x_93_, 0);
v___x_96_ = lean_int_dec_lt(v_n_94_, v_n_95_);
if (v___x_96_ == 0)
{
uint8_t v___x_97_; 
v___x_97_ = lean_int_dec_eq(v_n_94_, v_n_95_);
if (v___x_97_ == 0)
{
uint8_t v___x_98_; 
v___x_98_ = 2;
return v___x_98_;
}
else
{
uint8_t v___x_99_; 
v___x_99_ = 1;
return v___x_99_;
}
}
else
{
uint8_t v___x_100_; 
v___x_100_ = 0;
return v___x_100_;
}
}
else
{
uint8_t v___x_101_; 
v___x_101_ = 0;
return v___x_101_;
}
}
else
{
if (lean_obj_tag(v_x_93_) == 0)
{
uint8_t v___x_102_; 
v___x_102_ = 2;
return v___x_102_;
}
else
{
double v_p_103_; double v_p_104_; uint8_t v___y_106_; uint8_t v___x_111_; 
v_p_103_ = lean_ctor_get_float(v_x_92_, 0);
v_p_104_ = lean_ctor_get_float(v_x_93_, 0);
v___x_111_ = lean_float_decLt(v_p_104_, v_p_103_);
if (v___x_111_ == 0)
{
double v___x_112_; double v___x_113_; uint8_t v___x_114_; 
v___x_112_ = lean_float_sub(v_p_104_, v_p_103_);
v___x_113_ = lean_float_once(&lp_aesop_Aesop_instBEqForwardRulePriority_beq___closed__0, &lp_aesop_Aesop_instBEqForwardRulePriority_beq___closed__0_once, _init_lp_aesop_Aesop_instBEqForwardRulePriority_beq___closed__0);
v___x_114_ = lean_float_decLt(v___x_112_, v___x_113_);
v___y_106_ = v___x_114_;
goto v___jp_105_;
}
else
{
double v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; double v___x_118_; uint8_t v___x_119_; 
v___x_115_ = lean_float_sub(v_p_103_, v_p_104_);
v___x_116_ = lean_unsigned_to_nat(1u);
v___x_117_ = lean_unsigned_to_nat(5u);
v___x_118_ = l_Float_ofScientific(v___x_116_, v___x_111_, v___x_117_);
v___x_119_ = lean_float_decLt(v___x_115_, v___x_118_);
v___y_106_ = v___x_119_;
goto v___jp_105_;
}
v___jp_105_:
{
if (v___y_106_ == 0)
{
uint8_t v___x_107_; 
v___x_107_ = lean_float_decLt(v_p_103_, v_p_104_);
if (v___x_107_ == 0)
{
uint8_t v___x_108_; 
v___x_108_ = 0;
return v___x_108_;
}
else
{
uint8_t v___x_109_; 
v___x_109_ = 2;
return v___x_109_;
}
}
else
{
uint8_t v___x_110_; 
v___x_110_ = 1;
return v___x_110_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRulePriority_compare___boxed(lean_object* v_x_120_, lean_object* v_x_121_){
_start:
{
uint8_t v_res_122_; lean_object* v_r_123_; 
v_res_122_ = lp_aesop_Aesop_ForwardRulePriority_compare(v_x_120_, v_x_121_);
lean_dec_ref(v_x_121_);
lean_dec_ref(v_x_120_);
v_r_123_ = lean_box(v_res_122_);
return v_r_123_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRulePriority_instToString___lam__0(lean_object* v_x_126_){
_start:
{
if (lean_obj_tag(v_x_126_) == 0)
{
lean_object* v_n_127_; lean_object* v___x_128_; 
v_n_127_ = lean_ctor_get(v_x_126_, 0);
v___x_128_ = l_Int_repr(v_n_127_);
return v___x_128_;
}
else
{
double v_p_129_; lean_object* v___x_130_; 
v_p_129_ = lean_ctor_get_float(v_x_126_, 0);
v___x_130_ = lp_aesop_Aesop_Percent_toHumanString(v_p_129_);
return v___x_130_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRulePriority_instToString___lam__0___boxed(lean_object* v_x_131_){
_start:
{
lean_object* v_res_132_; 
v_res_132_ = lp_aesop_Aesop_ForwardRulePriority_instToString___lam__0(v_x_131_);
lean_dec_ref(v_x_131_);
return v_res_132_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedForwardRule_default___closed__0(void){
_start:
{
lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; 
v___x_135_ = lp_aesop_Aesop_instInhabitedForwardRulePriority_default;
v___x_136_ = lp_aesop_Aesop_instInhabitedRuleTerm_default;
v___x_137_ = lp_aesop_Aesop_instInhabitedRuleName_default;
v___x_138_ = lp_aesop_Aesop_instInhabitedForwardRuleInfo_default;
v___x_139_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_139_, 0, v___x_138_);
lean_ctor_set(v___x_139_, 1, v___x_137_);
lean_ctor_set(v___x_139_, 2, v___x_136_);
lean_ctor_set(v___x_139_, 3, v___x_135_);
return v___x_139_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedForwardRule_default(void){
_start:
{
lean_object* v___x_140_; 
v___x_140_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedForwardRule_default___closed__0, &lp_aesop_Aesop_instInhabitedForwardRule_default___closed__0_once, _init_lp_aesop_Aesop_instInhabitedForwardRule_default___closed__0);
return v___x_140_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedForwardRule(void){
_start:
{
lean_object* v___x_141_; 
v___x_141_ = lp_aesop_Aesop_instInhabitedForwardRule_default;
return v___x_141_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_ForwardRule_instBEq___lam__0(lean_object* v_r_u2081_142_, lean_object* v_r_u2082_143_){
_start:
{
lean_object* v_name_144_; lean_object* v_name_145_; lean_object* v_name_146_; uint8_t v_builder_147_; uint8_t v_phase_148_; uint8_t v_scope_149_; uint64_t v_hash_150_; lean_object* v_name_151_; uint8_t v_builder_152_; uint8_t v_phase_153_; uint8_t v_scope_154_; uint64_t v_hash_155_; uint8_t v___x_156_; 
v_name_144_ = lean_ctor_get(v_r_u2081_142_, 1);
v_name_145_ = lean_ctor_get(v_r_u2082_143_, 1);
v_name_146_ = lean_ctor_get(v_name_144_, 0);
v_builder_147_ = lean_ctor_get_uint8(v_name_144_, sizeof(void*)*1 + 8);
v_phase_148_ = lean_ctor_get_uint8(v_name_144_, sizeof(void*)*1 + 9);
v_scope_149_ = lean_ctor_get_uint8(v_name_144_, sizeof(void*)*1 + 10);
v_hash_150_ = lean_ctor_get_uint64(v_name_144_, sizeof(void*)*1);
v_name_151_ = lean_ctor_get(v_name_145_, 0);
v_builder_152_ = lean_ctor_get_uint8(v_name_145_, sizeof(void*)*1 + 8);
v_phase_153_ = lean_ctor_get_uint8(v_name_145_, sizeof(void*)*1 + 9);
v_scope_154_ = lean_ctor_get_uint8(v_name_145_, sizeof(void*)*1 + 10);
v_hash_155_ = lean_ctor_get_uint64(v_name_145_, sizeof(void*)*1);
v___x_156_ = lean_uint64_dec_eq(v_hash_150_, v_hash_155_);
if (v___x_156_ == 0)
{
return v___x_156_;
}
else
{
uint8_t v___x_157_; 
v___x_157_ = lp_aesop_Aesop_instBEqBuilderName_beq(v_builder_147_, v_builder_152_);
if (v___x_157_ == 0)
{
return v___x_157_;
}
else
{
uint8_t v___x_158_; 
v___x_158_ = lp_aesop_Aesop_instBEqPhaseName_beq(v_phase_148_, v_phase_153_);
if (v___x_158_ == 0)
{
return v___x_158_;
}
else
{
uint8_t v___x_159_; 
v___x_159_ = lp_aesop_Aesop_instBEqScopeName_beq(v_scope_149_, v_scope_154_);
if (v___x_159_ == 0)
{
return v___x_159_;
}
else
{
uint8_t v___x_160_; 
v___x_160_ = lean_name_eq(v_name_146_, v_name_151_);
return v___x_160_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRule_instBEq___lam__0___boxed(lean_object* v_r_u2081_161_, lean_object* v_r_u2082_162_){
_start:
{
uint8_t v_res_163_; lean_object* v_r_164_; 
v_res_163_ = lp_aesop_Aesop_ForwardRule_instBEq___lam__0(v_r_u2081_161_, v_r_u2082_162_);
lean_dec_ref(v_r_u2082_162_);
lean_dec_ref(v_r_u2081_161_);
v_r_164_ = lean_box(v_res_163_);
return v_r_164_;
}
}
LEAN_EXPORT uint64_t lp_aesop_Aesop_ForwardRule_instHashable___lam__0(lean_object* v_r_167_){
_start:
{
lean_object* v_name_168_; uint64_t v_hash_169_; 
v_name_168_ = lean_ctor_get(v_r_167_, 1);
v_hash_169_ = lean_ctor_get_uint64(v_name_168_, sizeof(void*)*1);
return v_hash_169_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRule_instHashable___lam__0___boxed(lean_object* v_r_170_){
_start:
{
uint64_t v_res_171_; lean_object* v_r_172_; 
v_res_171_ = lp_aesop_Aesop_ForwardRule_instHashable___lam__0(v_r_170_);
lean_dec_ref(v_r_170_);
v_r_172_ = lean_box_uint64(v_res_171_);
return v_r_172_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_ForwardRule_instOrd___lam__0(lean_object* v_r_u2081_175_, lean_object* v_r_u2082_176_){
_start:
{
lean_object* v_name_177_; lean_object* v_prio_178_; lean_object* v_name_179_; lean_object* v_prio_180_; uint8_t v___x_181_; 
v_name_177_ = lean_ctor_get(v_r_u2081_175_, 1);
v_prio_178_ = lean_ctor_get(v_r_u2081_175_, 3);
v_name_179_ = lean_ctor_get(v_r_u2082_176_, 1);
v_prio_180_ = lean_ctor_get(v_r_u2082_176_, 3);
v___x_181_ = lp_aesop_Aesop_ForwardRulePriority_compare(v_prio_178_, v_prio_180_);
if (v___x_181_ == 1)
{
uint8_t v___x_182_; 
v___x_182_ = lp_aesop_Aesop_RuleName_compare(v_name_177_, v_name_179_);
return v___x_182_;
}
else
{
return v___x_181_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRule_instOrd___lam__0___boxed(lean_object* v_r_u2081_183_, lean_object* v_r_u2082_184_){
_start:
{
uint8_t v_res_185_; lean_object* v_r_186_; 
v_res_185_ = lp_aesop_Aesop_ForwardRule_instOrd___lam__0(v_r_u2081_183_, v_r_u2082_184_);
lean_dec_ref(v_r_u2082_184_);
lean_dec_ref(v_r_u2081_183_);
v_r_186_ = lean_box(v_res_185_);
return v_r_186_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRule_instToString___lam__0(lean_object* v_r_205_){
_start:
{
lean_object* v___y_207_; lean_object* v___y_208_; lean_object* v_name_209_; lean_object* v___y_210_; lean_object* v___y_211_; lean_object* v___y_219_; lean_object* v_name_220_; uint8_t v_scope_221_; lean_object* v___y_222_; lean_object* v___y_223_; lean_object* v___y_224_; lean_object* v___y_230_; lean_object* v_name_231_; uint8_t v_builder_232_; uint8_t v_scope_233_; lean_object* v___y_234_; lean_object* v_name_245_; lean_object* v_prio_246_; lean_object* v___x_247_; lean_object* v___y_249_; 
v_name_245_ = lean_ctor_get(v_r_205_, 1);
lean_inc_ref(v_name_245_);
v_prio_246_ = lean_ctor_get(v_r_205_, 3);
lean_inc_ref(v_prio_246_);
lean_dec_ref(v_r_205_);
v___x_247_ = ((lean_object*)(lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__11));
if (lean_obj_tag(v_prio_246_) == 0)
{
lean_object* v_n_260_; lean_object* v___x_261_; 
v_n_260_ = lean_ctor_get(v_prio_246_, 0);
lean_inc(v_n_260_);
lean_dec_ref_known(v_prio_246_, 1);
v___x_261_ = l_Int_repr(v_n_260_);
lean_dec(v_n_260_);
v___y_249_ = v___x_261_;
goto v___jp_248_;
}
else
{
double v_p_262_; lean_object* v___x_263_; 
v_p_262_ = lean_ctor_get_float(v_prio_246_, 0);
lean_dec_ref_known(v_prio_246_, 0);
v___x_263_ = lp_aesop_Aesop_Percent_toHumanString(v_p_262_);
v___y_249_ = v___x_263_;
goto v___jp_248_;
}
v___jp_206_:
{
lean_object* v___x_212_; lean_object* v___x_213_; uint8_t v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; 
v___x_212_ = lean_string_append(v___y_207_, v___y_211_);
v___x_213_ = lean_string_append(v___x_212_, v___y_210_);
v___x_214_ = 1;
v___x_215_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_209_, v___x_214_);
v___x_216_ = lean_string_append(v___x_213_, v___x_215_);
lean_dec_ref(v___x_215_);
v___x_217_ = lean_string_append(v___y_208_, v___x_216_);
lean_dec_ref(v___x_216_);
return v___x_217_;
}
v___jp_218_:
{
lean_object* v___x_225_; lean_object* v___x_226_; 
v___x_225_ = lean_string_append(v___y_223_, v___y_224_);
v___x_226_ = lean_string_append(v___x_225_, v___y_222_);
if (v_scope_221_ == 0)
{
lean_object* v___x_227_; 
v___x_227_ = ((lean_object*)(lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__0));
v___y_207_ = v___x_226_;
v___y_208_ = v___y_219_;
v_name_209_ = v_name_220_;
v___y_210_ = v___y_222_;
v___y_211_ = v___x_227_;
goto v___jp_206_;
}
else
{
lean_object* v___x_228_; 
v___x_228_ = ((lean_object*)(lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__1));
v___y_207_ = v___x_226_;
v___y_208_ = v___y_219_;
v_name_209_ = v_name_220_;
v___y_210_ = v___y_222_;
v___y_211_ = v___x_228_;
goto v___jp_206_;
}
}
v___jp_229_:
{
lean_object* v___x_235_; lean_object* v___x_236_; 
v___x_235_ = ((lean_object*)(lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__2));
lean_inc_ref(v___y_234_);
v___x_236_ = lean_string_append(v___y_234_, v___x_235_);
switch(v_builder_232_)
{
case 0:
{
lean_object* v___x_237_; 
v___x_237_ = ((lean_object*)(lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__3));
v___y_219_ = v___y_230_;
v_name_220_ = v_name_231_;
v_scope_221_ = v_scope_233_;
v___y_222_ = v___x_235_;
v___y_223_ = v___x_236_;
v___y_224_ = v___x_237_;
goto v___jp_218_;
}
case 1:
{
lean_object* v___x_238_; 
v___x_238_ = ((lean_object*)(lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__4));
v___y_219_ = v___y_230_;
v_name_220_ = v_name_231_;
v_scope_221_ = v_scope_233_;
v___y_222_ = v___x_235_;
v___y_223_ = v___x_236_;
v___y_224_ = v___x_238_;
goto v___jp_218_;
}
case 2:
{
lean_object* v___x_239_; 
v___x_239_ = ((lean_object*)(lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__5));
v___y_219_ = v___y_230_;
v_name_220_ = v_name_231_;
v_scope_221_ = v_scope_233_;
v___y_222_ = v___x_235_;
v___y_223_ = v___x_236_;
v___y_224_ = v___x_239_;
goto v___jp_218_;
}
case 3:
{
lean_object* v___x_240_; 
v___x_240_ = ((lean_object*)(lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__6));
v___y_219_ = v___y_230_;
v_name_220_ = v_name_231_;
v_scope_221_ = v_scope_233_;
v___y_222_ = v___x_235_;
v___y_223_ = v___x_236_;
v___y_224_ = v___x_240_;
goto v___jp_218_;
}
case 4:
{
lean_object* v___x_241_; 
v___x_241_ = ((lean_object*)(lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__7));
v___y_219_ = v___y_230_;
v_name_220_ = v_name_231_;
v_scope_221_ = v_scope_233_;
v___y_222_ = v___x_235_;
v___y_223_ = v___x_236_;
v___y_224_ = v___x_241_;
goto v___jp_218_;
}
case 5:
{
lean_object* v___x_242_; 
v___x_242_ = ((lean_object*)(lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__8));
v___y_219_ = v___y_230_;
v_name_220_ = v_name_231_;
v_scope_221_ = v_scope_233_;
v___y_222_ = v___x_235_;
v___y_223_ = v___x_236_;
v___y_224_ = v___x_242_;
goto v___jp_218_;
}
case 6:
{
lean_object* v___x_243_; 
v___x_243_ = ((lean_object*)(lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__9));
v___y_219_ = v___y_230_;
v_name_220_ = v_name_231_;
v_scope_221_ = v_scope_233_;
v___y_222_ = v___x_235_;
v___y_223_ = v___x_236_;
v___y_224_ = v___x_243_;
goto v___jp_218_;
}
default: 
{
lean_object* v___x_244_; 
v___x_244_ = ((lean_object*)(lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__10));
v___y_219_ = v___y_230_;
v_name_220_ = v_name_231_;
v_scope_221_ = v_scope_233_;
v___y_222_ = v___x_235_;
v___y_223_ = v___x_236_;
v___y_224_ = v___x_244_;
goto v___jp_218_;
}
}
}
v___jp_248_:
{
lean_object* v_name_250_; uint8_t v_builder_251_; uint8_t v_phase_252_; uint8_t v_scope_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; 
v_name_250_ = lean_ctor_get(v_name_245_, 0);
lean_inc(v_name_250_);
v_builder_251_ = lean_ctor_get_uint8(v_name_245_, sizeof(void*)*1 + 8);
v_phase_252_ = lean_ctor_get_uint8(v_name_245_, sizeof(void*)*1 + 9);
v_scope_253_ = lean_ctor_get_uint8(v_name_245_, sizeof(void*)*1 + 10);
lean_dec_ref(v_name_245_);
v___x_254_ = lean_string_append(v___x_247_, v___y_249_);
lean_dec_ref(v___y_249_);
v___x_255_ = ((lean_object*)(lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__12));
v___x_256_ = lean_string_append(v___x_254_, v___x_255_);
switch(v_phase_252_)
{
case 0:
{
lean_object* v___x_257_; 
v___x_257_ = ((lean_object*)(lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__13));
v___y_230_ = v___x_256_;
v_name_231_ = v_name_250_;
v_builder_232_ = v_builder_251_;
v_scope_233_ = v_scope_253_;
v___y_234_ = v___x_257_;
goto v___jp_229_;
}
case 1:
{
lean_object* v___x_258_; 
v___x_258_ = ((lean_object*)(lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__14));
v___y_230_ = v___x_256_;
v_name_231_ = v_name_250_;
v_builder_232_ = v_builder_251_;
v_scope_233_ = v_scope_253_;
v___y_234_ = v___x_258_;
goto v___jp_229_;
}
default: 
{
lean_object* v___x_259_; 
v___x_259_ = ((lean_object*)(lp_aesop_Aesop_ForwardRule_instToString___lam__0___closed__15));
v___y_230_ = v___x_256_;
v_name_231_ = v_name_250_;
v_builder_232_ = v_builder_251_;
v_scope_233_ = v_scope_253_;
v___y_234_ = v___x_259_;
goto v___jp_229_;
}
}
}
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_ForwardRule_destruct(lean_object* v_r_266_){
_start:
{
lean_object* v_name_267_; uint8_t v_builder_268_; 
v_name_267_ = lean_ctor_get(v_r_266_, 1);
v_builder_268_ = lean_ctor_get_uint8(v_name_267_, sizeof(void*)*1 + 8);
if (v_builder_268_ == 3)
{
uint8_t v___x_269_; 
v___x_269_ = 1;
return v___x_269_;
}
else
{
uint8_t v___x_270_; 
v___x_270_ = 0;
return v___x_270_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRule_destruct___boxed(lean_object* v_r_271_){
_start:
{
uint8_t v_res_272_; lean_object* v_r_273_; 
v_res_272_ = lp_aesop_Aesop_ForwardRule_destruct(v_r_271_);
lean_dec_ref(v_r_271_);
v_r_273_ = lean_box(v_res_272_);
return v_r_273_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Forward_RuleInfo(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Percent(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_RuleTac_RuleTerm(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Rule_Forward(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Forward_RuleInfo(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Percent(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleTac_RuleTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_instInhabitedForwardRulePriority_default = _init_lp_aesop_Aesop_instInhabitedForwardRulePriority_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedForwardRulePriority_default);
lp_aesop_Aesop_instInhabitedForwardRulePriority = _init_lp_aesop_Aesop_instInhabitedForwardRulePriority();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedForwardRulePriority);
lp_aesop_Aesop_instInhabitedForwardRule_default = _init_lp_aesop_Aesop_instInhabitedForwardRule_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedForwardRule_default);
lp_aesop_Aesop_instInhabitedForwardRule = _init_lp_aesop_Aesop_instInhabitedForwardRule();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedForwardRule);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Rule_Forward(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Forward_RuleInfo(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Percent(uint8_t builtin);
lean_object* initialize_aesop_Aesop_RuleTac_RuleTerm(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Rule_Forward(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Forward_RuleInfo(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Percent(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleTac_RuleTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Rule_Forward(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Rule_Forward(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Rule_Forward(builtin);
}
#ifdef __cplusplus
}
#endif
