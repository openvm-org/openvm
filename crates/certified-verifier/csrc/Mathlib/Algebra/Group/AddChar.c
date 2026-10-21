// Lean compiler output
// Module: Mathlib.Algebra.Group.AddChar
// Imports: public import Init public meta import Init public import Mathlib.Algebra.BigOperators.Pi public import Mathlib.Algebra.BigOperators.Ring.Finset public import Mathlib.Algebra.Group.Subgroup.Ker public import Mathlib.Algebra.Group.TransferInstance public import Mathlib.Algebra.Group.Units.Equiv
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
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_OneHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Additive_ofMul(lean_object*);
lean_object* lp_mathlib_Additive_toMul(lean_object*);
lean_object* lp_mathlib_negAddMonoidHom___redArg(lean_object*);
lean_object* lp_mathlib_DivInvMonoid_div_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiplicative_ofAdd(lean_object*);
lean_object* l_npowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_zpowRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_Ring_toNonAssocRing___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoidHom_mulLeft___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_DivInvMonoid_div_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_zpowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toMonoidHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toMonoidHomEquiv___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toMonoidHomEquiv___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_AddChar_toMonoidHomEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddChar_toMonoidHomEquiv___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddChar_toMonoidHomEquiv___closed__0 = (const lean_object*)&lp_mathlib_AddChar_toMonoidHomEquiv___closed__0_value;
static const lean_closure_object lp_mathlib_AddChar_toMonoidHomEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddChar_toMonoidHomEquiv___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddChar_toMonoidHomEquiv___closed__1 = (const lean_object*)&lp_mathlib_AddChar_toMonoidHomEquiv___closed__1_value;
static const lean_ctor_object lp_mathlib_AddChar_toMonoidHomEquiv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_AddChar_toMonoidHomEquiv___closed__0_value),((lean_object*)&lp_mathlib_AddChar_toMonoidHomEquiv___closed__1_value)}};
static const lean_object* lp_mathlib_AddChar_toMonoidHomEquiv___closed__2 = (const lean_object*)&lp_mathlib_AddChar_toMonoidHomEquiv___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toMonoidHomEquiv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toMonoidHomEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toAddMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toAddMonoidHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toAddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toAddMonoidHomEquiv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toAddMonoidHomEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instOne___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instOne___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instOne___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instOne___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instOne(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instOne___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instZero___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instZero___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instZero(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instInhabited___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instInhabited___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instInhabited(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instInhabited___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compAddChar___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compAddChar___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compAddChar(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compAddChar___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_compAddMonoidHom___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_compAddMonoidHom___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_compAddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_compAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instCommMonoid___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instCommMonoid___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instCommMonoid___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instCommMonoid___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instCommMonoid___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instCommMonoid___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instCommMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__0;
static lean_once_cell_t lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommMonoid___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommMonoid___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommMonoid___aux__6___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommMonoid___aux__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommMonoid___aux__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommMonoid___aux__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommMonoid___aux__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toMonoidHomMulEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toMonoidHomMulEquiv___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toMonoidHomMulEquiv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toMonoidHomMulEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toAddMonoidAddEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toAddMonoidAddEquiv___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toAddMonoidAddEquiv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toAddMonoidAddEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_doubleDualEmb___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_AddChar_doubleDualEmb___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddChar_doubleDualEmb___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddChar_doubleDualEmb___closed__0 = (const lean_object*)&lp_mathlib_AddChar_doubleDualEmb___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_AddChar_doubleDualEmb(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_doubleDualEmb___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instCommGroup___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instCommGroup___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instCommGroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instCommGroup(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_AddChar_instAddCommGroup___aux__1___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddChar_instAddCommGroup___aux__1___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommGroup___aux__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommGroup___aux__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommGroup___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommGroup___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommGroup___aux__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommGroup___aux__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommGroup___aux__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommGroup___aux__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommGroup___aux__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommGroup___aux__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommGroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommGroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_mulShift___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_mulShift___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_mulShift(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_mulShift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toMonoidHom___redArg(lean_object* v_00_u03c6_1_){
_start:
{
lean_inc(v_00_u03c6_1_);
return v_00_u03c6_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toMonoidHom___redArg___boxed(lean_object* v_00_u03c6_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_AddChar_toMonoidHom___redArg(v_00_u03c6_2_);
lean_dec(v_00_u03c6_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toMonoidHom(lean_object* v_A_4_, lean_object* v_M_5_, lean_object* v_inst_6_, lean_object* v_inst_7_, lean_object* v_00_u03c6_8_){
_start:
{
lean_inc(v_00_u03c6_8_);
return v_00_u03c6_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toMonoidHom___boxed(lean_object* v_A_9_, lean_object* v_M_10_, lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_00_u03c6_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_AddChar_toMonoidHom(v_A_9_, v_M_10_, v_inst_11_, v_inst_12_, v_00_u03c6_13_);
lean_dec(v_00_u03c6_13_);
lean_dec_ref(v_inst_12_);
lean_dec_ref(v_inst_11_);
return v_res_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toMonoidHomEquiv___lam__0(lean_object* v_00_u03c6_15_, lean_object* v___y_16_){
_start:
{
lean_object* v___x_17_; 
v___x_17_ = lean_apply_1(v_00_u03c6_15_, v___y_16_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toMonoidHomEquiv___lam__1(lean_object* v_f_18_, lean_object* v___y_19_){
_start:
{
lean_object* v___x_20_; 
v___x_20_ = lean_apply_1(v_f_18_, v___y_19_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toMonoidHomEquiv(lean_object* v_A_26_, lean_object* v_M_27_, lean_object* v_inst_28_, lean_object* v_inst_29_){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = ((lean_object*)(lp_mathlib_AddChar_toMonoidHomEquiv___closed__2));
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toMonoidHomEquiv___boxed(lean_object* v_A_31_, lean_object* v_M_32_, lean_object* v_inst_33_, lean_object* v_inst_34_){
_start:
{
lean_object* v_res_35_; 
v_res_35_ = lp_mathlib_AddChar_toMonoidHomEquiv(v_A_31_, v_M_32_, v_inst_33_, v_inst_34_);
lean_dec_ref(v_inst_34_);
lean_dec_ref(v_inst_33_);
return v_res_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toAddMonoidHom___redArg(lean_object* v_00_u03c6_36_){
_start:
{
lean_inc(v_00_u03c6_36_);
return v_00_u03c6_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toAddMonoidHom___redArg___boxed(lean_object* v_00_u03c6_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_mathlib_AddChar_toAddMonoidHom___redArg(v_00_u03c6_37_);
lean_dec(v_00_u03c6_37_);
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toAddMonoidHom(lean_object* v_A_39_, lean_object* v_M_40_, lean_object* v_inst_41_, lean_object* v_inst_42_, lean_object* v_00_u03c6_43_){
_start:
{
lean_inc(v_00_u03c6_43_);
return v_00_u03c6_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toAddMonoidHom___boxed(lean_object* v_A_44_, lean_object* v_M_45_, lean_object* v_inst_46_, lean_object* v_inst_47_, lean_object* v_00_u03c6_48_){
_start:
{
lean_object* v_res_49_; 
v_res_49_ = lp_mathlib_AddChar_toAddMonoidHom(v_A_44_, v_M_45_, v_inst_46_, v_inst_47_, v_00_u03c6_48_);
lean_dec(v_00_u03c6_48_);
lean_dec_ref(v_inst_47_);
lean_dec_ref(v_inst_46_);
return v_res_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toAddMonoidHomEquiv(lean_object* v_A_50_, lean_object* v_M_51_, lean_object* v_inst_52_, lean_object* v_inst_53_){
_start:
{
lean_object* v___x_54_; 
v___x_54_ = ((lean_object*)(lp_mathlib_AddChar_toMonoidHomEquiv___closed__2));
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toAddMonoidHomEquiv___boxed(lean_object* v_A_55_, lean_object* v_M_56_, lean_object* v_inst_57_, lean_object* v_inst_58_){
_start:
{
lean_object* v_res_59_; 
v_res_59_ = lp_mathlib_AddChar_toAddMonoidHomEquiv(v_A_55_, v_M_56_, v_inst_57_, v_inst_58_);
lean_dec_ref(v_inst_58_);
lean_dec_ref(v_inst_57_);
return v_res_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instOne___redArg___lam__0(lean_object* v_toOne_60_, lean_object* v_x_61_){
_start:
{
lean_inc(v_toOne_60_);
return v_toOne_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instOne___redArg___lam__0___boxed(lean_object* v_toOne_62_, lean_object* v_x_63_){
_start:
{
lean_object* v_res_64_; 
v_res_64_ = lp_mathlib_AddChar_instOne___redArg___lam__0(v_toOne_62_, v_x_63_);
lean_dec(v_x_63_);
lean_dec(v_toOne_62_);
return v_res_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instOne___redArg(lean_object* v_inst_65_, lean_object* v_inst_66_){
_start:
{
lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v_invFun_69_; lean_object* v___x_70_; lean_object* v_toOne_71_; lean_object* v___f_72_; lean_object* v___x_73_; 
v___x_67_ = lp_mathlib_AddChar_toMonoidHomEquiv(lean_box(0), lean_box(0), v_inst_65_, v_inst_66_);
v___x_68_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_66_);
v_invFun_69_ = lean_ctor_get(v___x_67_, 1);
lean_inc(v_invFun_69_);
lean_dec_ref(v___x_67_);
v___x_70_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_68_);
v_toOne_71_ = lean_ctor_get(v___x_70_, 0);
lean_inc(v_toOne_71_);
lean_dec_ref(v___x_70_);
v___f_72_ = lean_alloc_closure((void*)(lp_mathlib_AddChar_instOne___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_72_, 0, v_toOne_71_);
v___x_73_ = lean_apply_1(v_invFun_69_, v___f_72_);
return v___x_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instOne___redArg___boxed(lean_object* v_inst_74_, lean_object* v_inst_75_){
_start:
{
lean_object* v_res_76_; 
v_res_76_ = lp_mathlib_AddChar_instOne___redArg(v_inst_74_, v_inst_75_);
lean_dec_ref(v_inst_75_);
lean_dec_ref(v_inst_74_);
return v_res_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instOne(lean_object* v_A_77_, lean_object* v_M_78_, lean_object* v_inst_79_, lean_object* v_inst_80_){
_start:
{
lean_object* v___x_81_; 
v___x_81_ = lp_mathlib_AddChar_instOne___redArg(v_inst_79_, v_inst_80_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instOne___boxed(lean_object* v_A_82_, lean_object* v_M_83_, lean_object* v_inst_84_, lean_object* v_inst_85_){
_start:
{
lean_object* v_res_86_; 
v_res_86_ = lp_mathlib_AddChar_instOne(v_A_82_, v_M_83_, v_inst_84_, v_inst_85_);
lean_dec_ref(v_inst_85_);
lean_dec_ref(v_inst_84_);
return v_res_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instZero___redArg(lean_object* v_inst_87_, lean_object* v_inst_88_){
_start:
{
lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v_invFun_91_; lean_object* v___x_92_; lean_object* v_toOne_93_; lean_object* v___f_94_; lean_object* v___x_95_; 
v___x_89_ = lp_mathlib_AddChar_toMonoidHomEquiv(lean_box(0), lean_box(0), v_inst_87_, v_inst_88_);
v___x_90_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_88_);
v_invFun_91_ = lean_ctor_get(v___x_89_, 1);
lean_inc(v_invFun_91_);
lean_dec_ref(v___x_89_);
v___x_92_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_90_);
v_toOne_93_ = lean_ctor_get(v___x_92_, 0);
lean_inc(v_toOne_93_);
lean_dec_ref(v___x_92_);
v___f_94_ = lean_alloc_closure((void*)(lp_mathlib_AddChar_instOne___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_94_, 0, v_toOne_93_);
v___x_95_ = lean_apply_1(v_invFun_91_, v___f_94_);
return v___x_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instZero___redArg___boxed(lean_object* v_inst_96_, lean_object* v_inst_97_){
_start:
{
lean_object* v_res_98_; 
v_res_98_ = lp_mathlib_AddChar_instZero___redArg(v_inst_96_, v_inst_97_);
lean_dec_ref(v_inst_97_);
lean_dec_ref(v_inst_96_);
return v_res_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instZero(lean_object* v_A_99_, lean_object* v_M_100_, lean_object* v_inst_101_, lean_object* v_inst_102_){
_start:
{
lean_object* v___x_103_; 
v___x_103_ = lp_mathlib_AddChar_instZero___redArg(v_inst_101_, v_inst_102_);
return v___x_103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instZero___boxed(lean_object* v_A_104_, lean_object* v_M_105_, lean_object* v_inst_106_, lean_object* v_inst_107_){
_start:
{
lean_object* v_res_108_; 
v_res_108_ = lp_mathlib_AddChar_instZero(v_A_104_, v_M_105_, v_inst_106_, v_inst_107_);
lean_dec_ref(v_inst_107_);
lean_dec_ref(v_inst_106_);
return v_res_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instInhabited___redArg(lean_object* v_inst_109_, lean_object* v_inst_110_){
_start:
{
lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v_invFun_113_; lean_object* v___x_114_; lean_object* v_toOne_115_; lean_object* v___f_116_; lean_object* v___x_117_; 
v___x_111_ = lp_mathlib_AddChar_toMonoidHomEquiv(lean_box(0), lean_box(0), v_inst_109_, v_inst_110_);
v___x_112_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_110_);
v_invFun_113_ = lean_ctor_get(v___x_111_, 1);
lean_inc(v_invFun_113_);
lean_dec_ref(v___x_111_);
v___x_114_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_112_);
v_toOne_115_ = lean_ctor_get(v___x_114_, 0);
lean_inc(v_toOne_115_);
lean_dec_ref(v___x_114_);
v___f_116_ = lean_alloc_closure((void*)(lp_mathlib_AddChar_instOne___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_116_, 0, v_toOne_115_);
v___x_117_ = lean_apply_1(v_invFun_113_, v___f_116_);
return v___x_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instInhabited___redArg___boxed(lean_object* v_inst_118_, lean_object* v_inst_119_){
_start:
{
lean_object* v_res_120_; 
v_res_120_ = lp_mathlib_AddChar_instInhabited___redArg(v_inst_118_, v_inst_119_);
lean_dec_ref(v_inst_119_);
lean_dec_ref(v_inst_118_);
return v_res_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instInhabited(lean_object* v_A_121_, lean_object* v_M_122_, lean_object* v_inst_123_, lean_object* v_inst_124_){
_start:
{
lean_object* v___x_125_; 
v___x_125_ = lp_mathlib_AddChar_instInhabited___redArg(v_inst_123_, v_inst_124_);
return v___x_125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instInhabited___boxed(lean_object* v_A_126_, lean_object* v_M_127_, lean_object* v_inst_128_, lean_object* v_inst_129_){
_start:
{
lean_object* v_res_130_; 
v_res_130_ = lp_mathlib_AddChar_instInhabited(v_A_126_, v_M_127_, v_inst_128_, v_inst_129_);
lean_dec_ref(v_inst_129_);
lean_dec_ref(v_inst_128_);
return v_res_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compAddChar___redArg(lean_object* v_inst_131_, lean_object* v_inst_132_, lean_object* v_f_133_, lean_object* v_00_u03c6_134_){
_start:
{
lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v_toFun_137_; lean_object* v___f_138_; lean_object* v___x_139_; 
v___x_135_ = lp_mathlib_AddChar_toMonoidHomEquiv(lean_box(0), lean_box(0), v_inst_131_, v_inst_132_);
v___x_136_ = lp_mathlib_Equiv_symm___redArg(v___x_135_);
v_toFun_137_ = lean_ctor_get(v___x_136_, 0);
lean_inc(v_toFun_137_);
lean_dec_ref(v___x_136_);
v___f_138_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_138_, 0, v_00_u03c6_134_);
lean_closure_set(v___f_138_, 1, v_f_133_);
v___x_139_ = lean_apply_1(v_toFun_137_, v___f_138_);
return v___x_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compAddChar___redArg___boxed(lean_object* v_inst_140_, lean_object* v_inst_141_, lean_object* v_f_142_, lean_object* v_00_u03c6_143_){
_start:
{
lean_object* v_res_144_; 
v_res_144_ = lp_mathlib_MonoidHom_compAddChar___redArg(v_inst_140_, v_inst_141_, v_f_142_, v_00_u03c6_143_);
lean_dec_ref(v_inst_141_);
lean_dec_ref(v_inst_140_);
return v_res_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compAddChar(lean_object* v_A_145_, lean_object* v_M_146_, lean_object* v_inst_147_, lean_object* v_inst_148_, lean_object* v_N_149_, lean_object* v_inst_150_, lean_object* v_f_151_, lean_object* v_00_u03c6_152_){
_start:
{
lean_object* v___x_153_; 
v___x_153_ = lp_mathlib_MonoidHom_compAddChar___redArg(v_inst_147_, v_inst_150_, v_f_151_, v_00_u03c6_152_);
return v___x_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compAddChar___boxed(lean_object* v_A_154_, lean_object* v_M_155_, lean_object* v_inst_156_, lean_object* v_inst_157_, lean_object* v_N_158_, lean_object* v_inst_159_, lean_object* v_f_160_, lean_object* v_00_u03c6_161_){
_start:
{
lean_object* v_res_162_; 
v_res_162_ = lp_mathlib_MonoidHom_compAddChar(v_A_154_, v_M_155_, v_inst_156_, v_inst_157_, v_N_158_, v_inst_159_, v_f_160_, v_00_u03c6_161_);
lean_dec_ref(v_inst_159_);
lean_dec_ref(v_inst_157_);
lean_dec_ref(v_inst_156_);
return v_res_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_compAddMonoidHom___redArg(lean_object* v_inst_163_, lean_object* v_inst_164_, lean_object* v_00_u03c6_165_, lean_object* v_f_166_){
_start:
{
lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v_toFun_169_; lean_object* v___f_170_; lean_object* v___x_171_; 
v___x_167_ = lp_mathlib_AddChar_toAddMonoidHomEquiv(lean_box(0), lean_box(0), v_inst_163_, v_inst_164_);
v___x_168_ = lp_mathlib_Equiv_symm___redArg(v___x_167_);
v_toFun_169_ = lean_ctor_get(v___x_168_, 0);
lean_inc(v_toFun_169_);
lean_dec_ref(v___x_168_);
v___f_170_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_170_, 0, v_f_166_);
lean_closure_set(v___f_170_, 1, v_00_u03c6_165_);
v___x_171_ = lean_apply_1(v_toFun_169_, v___f_170_);
return v___x_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_compAddMonoidHom___redArg___boxed(lean_object* v_inst_172_, lean_object* v_inst_173_, lean_object* v_00_u03c6_174_, lean_object* v_f_175_){
_start:
{
lean_object* v_res_176_; 
v_res_176_ = lp_mathlib_AddChar_compAddMonoidHom___redArg(v_inst_172_, v_inst_173_, v_00_u03c6_174_, v_f_175_);
lean_dec_ref(v_inst_173_);
lean_dec_ref(v_inst_172_);
return v_res_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_compAddMonoidHom(lean_object* v_A_177_, lean_object* v_B_178_, lean_object* v_M_179_, lean_object* v_inst_180_, lean_object* v_inst_181_, lean_object* v_inst_182_, lean_object* v_00_u03c6_183_, lean_object* v_f_184_){
_start:
{
lean_object* v___x_185_; 
v___x_185_ = lp_mathlib_AddChar_compAddMonoidHom___redArg(v_inst_180_, v_inst_182_, v_00_u03c6_183_, v_f_184_);
return v___x_185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_compAddMonoidHom___boxed(lean_object* v_A_186_, lean_object* v_B_187_, lean_object* v_M_188_, lean_object* v_inst_189_, lean_object* v_inst_190_, lean_object* v_inst_191_, lean_object* v_00_u03c6_192_, lean_object* v_f_193_){
_start:
{
lean_object* v_res_194_; 
v_res_194_ = lp_mathlib_AddChar_compAddMonoidHom(v_A_186_, v_B_187_, v_M_188_, v_inst_189_, v_inst_190_, v_inst_191_, v_00_u03c6_192_, v_f_193_);
lean_dec_ref(v_inst_191_);
lean_dec_ref(v_inst_190_);
lean_dec_ref(v_inst_189_);
return v_res_194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instCommMonoid___redArg___lam__0(lean_object* v_toFun_195_, lean_object* v_a_196_, lean_object* v_toNPow_197_, lean_object* v_a_198_, lean_object* v___y_199_){
_start:
{
lean_object* v___x_200_; lean_object* v___x_201_; 
v___x_200_ = lean_apply_2(v_toFun_195_, v_a_196_, v___y_199_);
v___x_201_ = lean_apply_2(v_toNPow_197_, v_a_198_, v___x_200_);
return v___x_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instCommMonoid___redArg___lam__1(lean_object* v_inst_202_, lean_object* v_inst_203_, lean_object* v_a_204_, lean_object* v_a_205_, lean_object* v___y_206_){
_start:
{
lean_object* v___x_207_; lean_object* v_toFun_208_; lean_object* v_invFun_209_; lean_object* v_toNPow_210_; lean_object* v___f_211_; lean_object* v___x_212_; 
v___x_207_ = lp_mathlib_AddChar_toMonoidHomEquiv(lean_box(0), lean_box(0), v_inst_202_, v_inst_203_);
v_toFun_208_ = lean_ctor_get(v___x_207_, 0);
lean_inc(v_toFun_208_);
v_invFun_209_ = lean_ctor_get(v___x_207_, 1);
lean_inc(v_invFun_209_);
lean_dec_ref(v___x_207_);
v_toNPow_210_ = lean_ctor_get(v_inst_203_, 2);
lean_inc(v_toNPow_210_);
lean_dec_ref(v_inst_203_);
v___f_211_ = lean_alloc_closure((void*)(lp_mathlib_AddChar_instCommMonoid___redArg___lam__0), 5, 4);
lean_closure_set(v___f_211_, 0, v_toFun_208_);
lean_closure_set(v___f_211_, 1, v_a_205_);
lean_closure_set(v___f_211_, 2, v_toNPow_210_);
lean_closure_set(v___f_211_, 3, v_a_204_);
v___x_212_ = lean_apply_2(v_invFun_209_, v___f_211_, v___y_206_);
return v___x_212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instCommMonoid___redArg___lam__1___boxed(lean_object* v_inst_213_, lean_object* v_inst_214_, lean_object* v_a_215_, lean_object* v_a_216_, lean_object* v___y_217_){
_start:
{
lean_object* v_res_218_; 
v_res_218_ = lp_mathlib_AddChar_instCommMonoid___redArg___lam__1(v_inst_213_, v_inst_214_, v_a_215_, v_a_216_, v___y_217_);
lean_dec_ref(v_inst_213_);
return v_res_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instCommMonoid___redArg___lam__2(lean_object* v_toFun_219_, lean_object* v_a_220_, lean_object* v_a_221_, lean_object* v_toMul_222_, lean_object* v_m_223_){
_start:
{
lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; 
lean_inc(v_toFun_219_);
lean_inc(v_m_223_);
v___x_224_ = lean_apply_2(v_toFun_219_, v_a_220_, v_m_223_);
v___x_225_ = lean_apply_2(v_toFun_219_, v_a_221_, v_m_223_);
v___x_226_ = lean_apply_2(v_toMul_222_, v___x_224_, v___x_225_);
return v___x_226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instCommMonoid___redArg___lam__3(lean_object* v_inst_227_, lean_object* v_inst_228_, lean_object* v_a_229_, lean_object* v_a_230_, lean_object* v___y_231_){
_start:
{
lean_object* v___x_232_; lean_object* v_toFun_233_; lean_object* v_invFun_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v_toMul_237_; lean_object* v___f_238_; lean_object* v___x_239_; 
v___x_232_ = lp_mathlib_AddChar_toMonoidHomEquiv(lean_box(0), lean_box(0), v_inst_227_, v_inst_228_);
v_toFun_233_ = lean_ctor_get(v___x_232_, 0);
lean_inc(v_toFun_233_);
v_invFun_234_ = lean_ctor_get(v___x_232_, 1);
lean_inc(v_invFun_234_);
lean_dec_ref(v___x_232_);
v___x_235_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_228_);
v___x_236_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_235_);
v_toMul_237_ = lean_ctor_get(v___x_236_, 1);
lean_inc(v_toMul_237_);
lean_dec_ref(v___x_236_);
v___f_238_ = lean_alloc_closure((void*)(lp_mathlib_AddChar_instCommMonoid___redArg___lam__2), 5, 4);
lean_closure_set(v___f_238_, 0, v_toFun_233_);
lean_closure_set(v___f_238_, 1, v_a_229_);
lean_closure_set(v___f_238_, 2, v_a_230_);
lean_closure_set(v___f_238_, 3, v_toMul_237_);
v___x_239_ = lean_apply_2(v_invFun_234_, v___f_238_, v___y_231_);
return v___x_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instCommMonoid___redArg___lam__3___boxed(lean_object* v_inst_240_, lean_object* v_inst_241_, lean_object* v_a_242_, lean_object* v_a_243_, lean_object* v___y_244_){
_start:
{
lean_object* v_res_245_; 
v_res_245_ = lp_mathlib_AddChar_instCommMonoid___redArg___lam__3(v_inst_240_, v_inst_241_, v_a_242_, v_a_243_, v___y_244_);
lean_dec_ref(v_inst_241_);
lean_dec_ref(v_inst_240_);
return v_res_245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instCommMonoid___redArg(lean_object* v_inst_246_, lean_object* v_inst_247_){
_start:
{
lean_object* v___f_248_; lean_object* v___f_249_; lean_object* v___x_250_; lean_object* v___x_251_; 
lean_inc_ref_n(v_inst_247_, 2);
lean_inc_ref_n(v_inst_246_, 2);
v___f_248_ = lean_alloc_closure((void*)(lp_mathlib_AddChar_instCommMonoid___redArg___lam__1___boxed), 5, 2);
lean_closure_set(v___f_248_, 0, v_inst_246_);
lean_closure_set(v___f_248_, 1, v_inst_247_);
v___f_249_ = lean_alloc_closure((void*)(lp_mathlib_AddChar_instCommMonoid___redArg___lam__3___boxed), 5, 2);
lean_closure_set(v___f_249_, 0, v_inst_246_);
lean_closure_set(v___f_249_, 1, v_inst_247_);
v___x_250_ = lp_mathlib_AddChar_instOne___redArg(v_inst_246_, v_inst_247_);
lean_dec_ref(v_inst_247_);
lean_dec_ref(v_inst_246_);
v___x_251_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_251_, 0, v___x_250_);
lean_ctor_set(v___x_251_, 1, v___f_249_);
lean_ctor_set(v___x_251_, 2, v___f_248_);
return v___x_251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instCommMonoid(lean_object* v_A_252_, lean_object* v_M_253_, lean_object* v_inst_254_, lean_object* v_inst_255_){
_start:
{
lean_object* v___x_256_; 
v___x_256_ = lp_mathlib_AddChar_instCommMonoid___redArg(v_inst_254_, v_inst_255_);
return v___x_256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___lam__0(lean_object* v_self_257_, lean_object* v___y_258_, lean_object* v___y_259_){
_start:
{
lean_object* v_toFun_260_; lean_object* v___x_261_; 
v_toFun_260_ = lean_ctor_get(v_self_257_, 0);
lean_inc(v_toFun_260_);
lean_dec_ref(v_self_257_);
v___x_261_ = lean_apply_2(v_toFun_260_, v___y_258_, v___y_259_);
return v___x_261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___lam__1(lean_object* v_toFun_262_, lean_object* v___x_263_, lean_object* v___x_264_, lean_object* v_toMul_265_, lean_object* v_m_266_){
_start:
{
lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; 
lean_inc(v_toFun_262_);
lean_inc(v_m_266_);
v___x_267_ = lean_apply_2(v_toFun_262_, v___x_263_, v_m_266_);
v___x_268_ = lean_apply_2(v_toFun_262_, v___x_264_, v_m_266_);
v___x_269_ = lean_apply_2(v_toMul_265_, v___x_267_, v___x_268_);
return v___x_269_;
}
}
static lean_object* _init_lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__0(void){
_start:
{
lean_object* v___x_270_; 
v___x_270_ = lp_mathlib_Additive_ofMul(lean_box(0));
return v___x_270_;
}
}
static lean_object* _init_lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__1(void){
_start:
{
lean_object* v___x_271_; 
v___x_271_ = lp_mathlib_Additive_toMul(lean_box(0));
return v___x_271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg(lean_object* v_inst_272_, lean_object* v_inst_273_, lean_object* v_x_274_, lean_object* v_y_275_){
_start:
{
lean_object* v___x_276_; lean_object* v_toFun_277_; lean_object* v_invFun_278_; lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v_toMul_281_; lean_object* v___x_282_; lean_object* v_toFun_283_; lean_object* v___x_284_; lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v___f_287_; lean_object* v___x_288_; lean_object* v___x_289_; 
v___x_276_ = lp_mathlib_AddChar_toMonoidHomEquiv(lean_box(0), lean_box(0), v_inst_272_, v_inst_273_);
v_toFun_277_ = lean_ctor_get(v___x_276_, 0);
lean_inc(v_toFun_277_);
v_invFun_278_ = lean_ctor_get(v___x_276_, 1);
lean_inc(v_invFun_278_);
lean_dec_ref(v___x_276_);
v___x_279_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_273_);
v___x_280_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_279_);
v_toMul_281_ = lean_ctor_get(v___x_280_, 1);
lean_inc(v_toMul_281_);
lean_dec_ref(v___x_280_);
v___x_282_ = lean_obj_once(&lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__0, &lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__0_once, _init_lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__0);
v_toFun_283_ = lean_ctor_get(v___x_282_, 0);
v___x_284_ = lean_obj_once(&lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__1, &lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__1_once, _init_lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__1);
v___x_285_ = lean_alloc_closure((void*)(lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___lam__0), 3, 2);
lean_closure_set(v___x_285_, 0, v___x_284_);
lean_closure_set(v___x_285_, 1, v_x_274_);
v___x_286_ = lean_alloc_closure((void*)(lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___lam__0), 3, 2);
lean_closure_set(v___x_286_, 0, v___x_284_);
lean_closure_set(v___x_286_, 1, v_y_275_);
v___f_287_ = lean_alloc_closure((void*)(lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___lam__1), 5, 4);
lean_closure_set(v___f_287_, 0, v_toFun_277_);
lean_closure_set(v___f_287_, 1, v___x_285_);
lean_closure_set(v___f_287_, 2, v___x_286_);
lean_closure_set(v___f_287_, 3, v_toMul_281_);
v___x_288_ = lean_apply_1(v_invFun_278_, v___f_287_);
lean_inc(v_toFun_283_);
v___x_289_ = lean_apply_1(v_toFun_283_, v___x_288_);
return v___x_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___boxed(lean_object* v_inst_290_, lean_object* v_inst_291_, lean_object* v_x_292_, lean_object* v_y_293_){
_start:
{
lean_object* v_res_294_; 
v_res_294_ = lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg(v_inst_290_, v_inst_291_, v_x_292_, v_y_293_);
lean_dec_ref(v_inst_291_);
lean_dec_ref(v_inst_290_);
return v_res_294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommMonoid___aux__1(lean_object* v_A_295_, lean_object* v_M_296_, lean_object* v_inst_297_, lean_object* v_inst_298_, lean_object* v_x_299_, lean_object* v_y_300_){
_start:
{
lean_object* v___x_301_; lean_object* v_toFun_302_; lean_object* v_invFun_303_; lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v_toMul_306_; lean_object* v___x_307_; lean_object* v_toFun_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___f_312_; lean_object* v___x_313_; lean_object* v___x_314_; 
v___x_301_ = lp_mathlib_AddChar_toMonoidHomEquiv(lean_box(0), lean_box(0), v_inst_297_, v_inst_298_);
v_toFun_302_ = lean_ctor_get(v___x_301_, 0);
lean_inc(v_toFun_302_);
v_invFun_303_ = lean_ctor_get(v___x_301_, 1);
lean_inc(v_invFun_303_);
lean_dec_ref(v___x_301_);
v___x_304_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_298_);
v___x_305_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_304_);
v_toMul_306_ = lean_ctor_get(v___x_305_, 1);
lean_inc(v_toMul_306_);
lean_dec_ref(v___x_305_);
v___x_307_ = lean_obj_once(&lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__0, &lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__0_once, _init_lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__0);
v_toFun_308_ = lean_ctor_get(v___x_307_, 0);
v___x_309_ = lean_obj_once(&lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__1, &lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__1_once, _init_lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__1);
v___x_310_ = lean_alloc_closure((void*)(lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___lam__0), 3, 2);
lean_closure_set(v___x_310_, 0, v___x_309_);
lean_closure_set(v___x_310_, 1, v_x_299_);
v___x_311_ = lean_alloc_closure((void*)(lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___lam__0), 3, 2);
lean_closure_set(v___x_311_, 0, v___x_309_);
lean_closure_set(v___x_311_, 1, v_y_300_);
v___f_312_ = lean_alloc_closure((void*)(lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___lam__1), 5, 4);
lean_closure_set(v___f_312_, 0, v_toFun_302_);
lean_closure_set(v___f_312_, 1, v___x_310_);
lean_closure_set(v___f_312_, 2, v___x_311_);
lean_closure_set(v___f_312_, 3, v_toMul_306_);
v___x_313_ = lean_apply_1(v_invFun_303_, v___f_312_);
lean_inc(v_toFun_308_);
v___x_314_ = lean_apply_1(v_toFun_308_, v___x_313_);
return v___x_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommMonoid___aux__1___boxed(lean_object* v_A_315_, lean_object* v_M_316_, lean_object* v_inst_317_, lean_object* v_inst_318_, lean_object* v_x_319_, lean_object* v_y_320_){
_start:
{
lean_object* v_res_321_; 
v_res_321_ = lp_mathlib_AddChar_instAddCommMonoid___aux__1(v_A_315_, v_M_316_, v_inst_317_, v_inst_318_, v_x_319_, v_y_320_);
lean_dec_ref(v_inst_318_);
lean_dec_ref(v_inst_317_);
return v_res_321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommMonoid___aux__6___redArg___lam__0(lean_object* v_toFun_322_, lean_object* v___x_323_, lean_object* v_toNPow_324_, lean_object* v_n_325_, lean_object* v___y_326_){
_start:
{
lean_object* v___x_327_; lean_object* v___x_328_; 
v___x_327_ = lean_apply_2(v_toFun_322_, v___x_323_, v___y_326_);
v___x_328_ = lean_apply_2(v_toNPow_324_, v_n_325_, v___x_327_);
return v___x_328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommMonoid___aux__6___redArg(lean_object* v_inst_329_, lean_object* v_inst_330_, lean_object* v_n_331_, lean_object* v_a_332_){
_start:
{
lean_object* v___x_333_; lean_object* v_toFun_334_; lean_object* v___x_335_; lean_object* v_toFun_336_; lean_object* v_invFun_337_; lean_object* v_toNPow_338_; lean_object* v___x_339_; lean_object* v_toFun_340_; lean_object* v___x_341_; lean_object* v___f_342_; lean_object* v___x_343_; lean_object* v___x_344_; 
v___x_333_ = lean_obj_once(&lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__1, &lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__1_once, _init_lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__1);
v_toFun_334_ = lean_ctor_get(v___x_333_, 0);
v___x_335_ = lp_mathlib_AddChar_toMonoidHomEquiv(lean_box(0), lean_box(0), v_inst_329_, v_inst_330_);
v_toFun_336_ = lean_ctor_get(v___x_335_, 0);
lean_inc(v_toFun_336_);
v_invFun_337_ = lean_ctor_get(v___x_335_, 1);
lean_inc(v_invFun_337_);
lean_dec_ref(v___x_335_);
v_toNPow_338_ = lean_ctor_get(v_inst_330_, 2);
lean_inc(v_toNPow_338_);
lean_dec_ref(v_inst_330_);
v___x_339_ = lean_obj_once(&lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__0, &lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__0_once, _init_lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__0);
v_toFun_340_ = lean_ctor_get(v___x_339_, 0);
lean_inc(v_toFun_334_);
v___x_341_ = lean_apply_1(v_toFun_334_, v_a_332_);
v___f_342_ = lean_alloc_closure((void*)(lp_mathlib_AddChar_instAddCommMonoid___aux__6___redArg___lam__0), 5, 4);
lean_closure_set(v___f_342_, 0, v_toFun_336_);
lean_closure_set(v___f_342_, 1, v___x_341_);
lean_closure_set(v___f_342_, 2, v_toNPow_338_);
lean_closure_set(v___f_342_, 3, v_n_331_);
v___x_343_ = lean_apply_1(v_invFun_337_, v___f_342_);
lean_inc(v_toFun_340_);
v___x_344_ = lean_apply_1(v_toFun_340_, v___x_343_);
return v___x_344_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommMonoid___aux__6___redArg___boxed(lean_object* v_inst_345_, lean_object* v_inst_346_, lean_object* v_n_347_, lean_object* v_a_348_){
_start:
{
lean_object* v_res_349_; 
v_res_349_ = lp_mathlib_AddChar_instAddCommMonoid___aux__6___redArg(v_inst_345_, v_inst_346_, v_n_347_, v_a_348_);
lean_dec_ref(v_inst_345_);
return v_res_349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommMonoid___aux__6(lean_object* v_A_350_, lean_object* v_M_351_, lean_object* v_inst_352_, lean_object* v_inst_353_, lean_object* v_n_354_, lean_object* v_a_355_){
_start:
{
lean_object* v___x_356_; lean_object* v_toFun_357_; lean_object* v___x_358_; lean_object* v_toFun_359_; lean_object* v_invFun_360_; lean_object* v_toNPow_361_; lean_object* v___x_362_; lean_object* v_toFun_363_; lean_object* v___x_364_; lean_object* v___f_365_; lean_object* v___x_366_; lean_object* v___x_367_; 
v___x_356_ = lean_obj_once(&lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__1, &lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__1_once, _init_lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__1);
v_toFun_357_ = lean_ctor_get(v___x_356_, 0);
v___x_358_ = lp_mathlib_AddChar_toMonoidHomEquiv(lean_box(0), lean_box(0), v_inst_352_, v_inst_353_);
v_toFun_359_ = lean_ctor_get(v___x_358_, 0);
lean_inc(v_toFun_359_);
v_invFun_360_ = lean_ctor_get(v___x_358_, 1);
lean_inc(v_invFun_360_);
lean_dec_ref(v___x_358_);
v_toNPow_361_ = lean_ctor_get(v_inst_353_, 2);
lean_inc(v_toNPow_361_);
lean_dec_ref(v_inst_353_);
v___x_362_ = lean_obj_once(&lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__0, &lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__0_once, _init_lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__0);
v_toFun_363_ = lean_ctor_get(v___x_362_, 0);
lean_inc(v_toFun_357_);
v___x_364_ = lean_apply_1(v_toFun_357_, v_a_355_);
v___f_365_ = lean_alloc_closure((void*)(lp_mathlib_AddChar_instAddCommMonoid___aux__6___redArg___lam__0), 5, 4);
lean_closure_set(v___f_365_, 0, v_toFun_359_);
lean_closure_set(v___f_365_, 1, v___x_364_);
lean_closure_set(v___f_365_, 2, v_toNPow_361_);
lean_closure_set(v___f_365_, 3, v_n_354_);
v___x_366_ = lean_apply_1(v_invFun_360_, v___f_365_);
lean_inc(v_toFun_363_);
v___x_367_ = lean_apply_1(v_toFun_363_, v___x_366_);
return v___x_367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommMonoid___aux__6___boxed(lean_object* v_A_368_, lean_object* v_M_369_, lean_object* v_inst_370_, lean_object* v_inst_371_, lean_object* v_n_372_, lean_object* v_a_373_){
_start:
{
lean_object* v_res_374_; 
v_res_374_ = lp_mathlib_AddChar_instAddCommMonoid___aux__6(v_A_368_, v_M_369_, v_inst_370_, v_inst_371_, v_n_372_, v_a_373_);
lean_dec_ref(v_inst_370_);
return v_res_374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommMonoid___redArg(lean_object* v_inst_375_, lean_object* v_inst_376_){
_start:
{
lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v___x_380_; 
v___x_377_ = lp_mathlib_AddChar_instZero___redArg(v_inst_375_, v_inst_376_);
lean_inc_ref(v_inst_376_);
lean_inc_ref(v_inst_375_);
v___x_378_ = lean_alloc_closure((void*)(lp_mathlib_AddChar_instAddCommMonoid___aux__1___boxed), 6, 4);
lean_closure_set(v___x_378_, 0, lean_box(0));
lean_closure_set(v___x_378_, 1, lean_box(0));
lean_closure_set(v___x_378_, 2, v_inst_375_);
lean_closure_set(v___x_378_, 3, v_inst_376_);
v___x_379_ = lean_alloc_closure((void*)(lp_mathlib_AddChar_instAddCommMonoid___aux__6___boxed), 6, 4);
lean_closure_set(v___x_379_, 0, lean_box(0));
lean_closure_set(v___x_379_, 1, lean_box(0));
lean_closure_set(v___x_379_, 2, v_inst_375_);
lean_closure_set(v___x_379_, 3, v_inst_376_);
v___x_380_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_380_, 0, v___x_377_);
lean_ctor_set(v___x_380_, 1, v___x_378_);
lean_ctor_set(v___x_380_, 2, v___x_379_);
return v___x_380_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommMonoid(lean_object* v_A_381_, lean_object* v_M_382_, lean_object* v_inst_383_, lean_object* v_inst_384_){
_start:
{
lean_object* v___x_385_; 
v___x_385_ = lp_mathlib_AddChar_instAddCommMonoid___redArg(v_inst_383_, v_inst_384_);
return v___x_385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toMonoidHomMulEquiv___redArg(lean_object* v_inst_386_, lean_object* v_inst_387_){
_start:
{
lean_object* v___x_388_; 
v___x_388_ = lp_mathlib_AddChar_toMonoidHomEquiv(lean_box(0), lean_box(0), v_inst_386_, v_inst_387_);
return v___x_388_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toMonoidHomMulEquiv___redArg___boxed(lean_object* v_inst_389_, lean_object* v_inst_390_){
_start:
{
lean_object* v_res_391_; 
v_res_391_ = lp_mathlib_AddChar_toMonoidHomMulEquiv___redArg(v_inst_389_, v_inst_390_);
lean_dec_ref(v_inst_390_);
lean_dec_ref(v_inst_389_);
return v_res_391_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toMonoidHomMulEquiv(lean_object* v_A_392_, lean_object* v_M_393_, lean_object* v_inst_394_, lean_object* v_inst_395_){
_start:
{
lean_object* v___x_396_; 
v___x_396_ = lp_mathlib_AddChar_toMonoidHomEquiv(lean_box(0), lean_box(0), v_inst_394_, v_inst_395_);
return v___x_396_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toMonoidHomMulEquiv___boxed(lean_object* v_A_397_, lean_object* v_M_398_, lean_object* v_inst_399_, lean_object* v_inst_400_){
_start:
{
lean_object* v_res_401_; 
v_res_401_ = lp_mathlib_AddChar_toMonoidHomMulEquiv(v_A_397_, v_M_398_, v_inst_399_, v_inst_400_);
lean_dec_ref(v_inst_400_);
lean_dec_ref(v_inst_399_);
return v_res_401_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toAddMonoidAddEquiv___redArg(lean_object* v_inst_402_, lean_object* v_inst_403_){
_start:
{
lean_object* v___x_404_; 
v___x_404_ = lp_mathlib_AddChar_toAddMonoidHomEquiv(lean_box(0), lean_box(0), v_inst_402_, v_inst_403_);
return v___x_404_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toAddMonoidAddEquiv___redArg___boxed(lean_object* v_inst_405_, lean_object* v_inst_406_){
_start:
{
lean_object* v_res_407_; 
v_res_407_ = lp_mathlib_AddChar_toAddMonoidAddEquiv___redArg(v_inst_405_, v_inst_406_);
lean_dec_ref(v_inst_406_);
lean_dec_ref(v_inst_405_);
return v_res_407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toAddMonoidAddEquiv(lean_object* v_A_408_, lean_object* v_M_409_, lean_object* v_inst_410_, lean_object* v_inst_411_){
_start:
{
lean_object* v___x_412_; 
v___x_412_ = lp_mathlib_AddChar_toAddMonoidHomEquiv(lean_box(0), lean_box(0), v_inst_410_, v_inst_411_);
return v___x_412_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_toAddMonoidAddEquiv___boxed(lean_object* v_A_413_, lean_object* v_M_414_, lean_object* v_inst_415_, lean_object* v_inst_416_){
_start:
{
lean_object* v_res_417_; 
v_res_417_ = lp_mathlib_AddChar_toAddMonoidAddEquiv(v_A_413_, v_M_414_, v_inst_415_, v_inst_416_);
lean_dec_ref(v_inst_416_);
lean_dec_ref(v_inst_415_);
return v_res_417_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_doubleDualEmb___lam__0(lean_object* v_a_418_, lean_object* v___y_419_){
_start:
{
lean_object* v___x_420_; 
v___x_420_ = lean_apply_1(v___y_419_, v_a_418_);
return v___x_420_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_doubleDualEmb(lean_object* v_A_422_, lean_object* v_M_423_, lean_object* v_inst_424_, lean_object* v_inst_425_){
_start:
{
lean_object* v___f_426_; 
v___f_426_ = ((lean_object*)(lp_mathlib_AddChar_doubleDualEmb___closed__0));
return v___f_426_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_doubleDualEmb___boxed(lean_object* v_A_427_, lean_object* v_M_428_, lean_object* v_inst_429_, lean_object* v_inst_430_){
_start:
{
lean_object* v_res_431_; 
v_res_431_ = lp_mathlib_AddChar_doubleDualEmb(v_A_427_, v_M_428_, v_inst_429_, v_inst_430_);
lean_dec_ref(v_inst_430_);
lean_dec_ref(v_inst_429_);
return v_res_431_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instCommGroup___redArg___lam__0(lean_object* v_inst_432_, lean_object* v_toAddMonoid_433_, lean_object* v_inst_434_, lean_object* v_00_u03c8_435_, lean_object* v___y_436_){
_start:
{
lean_object* v___x_437_; lean_object* v___x_42__overap_438_; lean_object* v___x_439_; 
v___x_437_ = lp_mathlib_negAddMonoidHom___redArg(v_inst_432_);
v___x_42__overap_438_ = lp_mathlib_AddChar_compAddMonoidHom___redArg(v_toAddMonoid_433_, v_inst_434_, v_00_u03c8_435_, v___x_437_);
v___x_439_ = lean_apply_1(v___x_42__overap_438_, v___y_436_);
return v___x_439_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instCommGroup___redArg___lam__0___boxed(lean_object* v_inst_440_, lean_object* v_toAddMonoid_441_, lean_object* v_inst_442_, lean_object* v_00_u03c8_443_, lean_object* v___y_444_){
_start:
{
lean_object* v_res_445_; 
v_res_445_ = lp_mathlib_AddChar_instCommGroup___redArg___lam__0(v_inst_440_, v_toAddMonoid_441_, v_inst_442_, v_00_u03c8_443_, v___y_444_);
lean_dec_ref(v_inst_442_);
lean_dec_ref(v_toAddMonoid_441_);
lean_dec_ref(v_inst_440_);
return v_res_445_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instCommGroup___redArg(lean_object* v_inst_446_, lean_object* v_inst_447_){
_start:
{
lean_object* v_toAddMonoid_448_; lean_object* v___f_449_; lean_object* v___x_450_; lean_object* v___x_451_; lean_object* v___f_452_; lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; 
v_toAddMonoid_448_ = lean_ctor_get(v_inst_446_, 0);
lean_inc_ref_n(v_toAddMonoid_448_, 3);
lean_inc_ref_n(v_inst_447_, 2);
v___f_449_ = lean_alloc_closure((void*)(lp_mathlib_AddChar_instCommGroup___redArg___lam__0___boxed), 5, 3);
lean_closure_set(v___f_449_, 0, v_inst_446_);
lean_closure_set(v___f_449_, 1, v_toAddMonoid_448_);
lean_closure_set(v___f_449_, 2, v_inst_447_);
v___x_450_ = lp_mathlib_AddChar_instCommMonoid___redArg(v_toAddMonoid_448_, v_inst_447_);
v___x_451_ = lp_mathlib_AddChar_instOne___redArg(v_toAddMonoid_448_, v_inst_447_);
v___f_452_ = lean_alloc_closure((void*)(lp_mathlib_AddChar_instCommMonoid___redArg___lam__3___boxed), 5, 2);
lean_closure_set(v___f_452_, 0, v_toAddMonoid_448_);
lean_closure_set(v___f_452_, 1, v_inst_447_);
lean_inc_ref_n(v___f_449_, 2);
lean_inc_ref(v___x_450_);
v___x_453_ = lean_alloc_closure((void*)(lp_mathlib_DivInvMonoid_div_x27___boxed), 5, 3);
lean_closure_set(v___x_453_, 0, lean_box(0));
lean_closure_set(v___x_453_, 1, v___x_450_);
lean_closure_set(v___x_453_, 2, v___f_449_);
lean_inc_ref(v___f_452_);
lean_inc(v___x_451_);
v___x_454_ = lean_alloc_closure((void*)(l_npowRec___boxed), 5, 3);
lean_closure_set(v___x_454_, 0, lean_box(0));
lean_closure_set(v___x_454_, 1, v___x_451_);
lean_closure_set(v___x_454_, 2, v___f_452_);
v___x_455_ = lean_alloc_closure((void*)(lp_mathlib_zpowRec___boxed), 7, 5);
lean_closure_set(v___x_455_, 0, lean_box(0));
lean_closure_set(v___x_455_, 1, v___x_451_);
lean_closure_set(v___x_455_, 2, v___f_452_);
lean_closure_set(v___x_455_, 3, v___f_449_);
lean_closure_set(v___x_455_, 4, v___x_454_);
v___x_456_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_456_, 0, v___x_450_);
lean_ctor_set(v___x_456_, 1, v___f_449_);
lean_ctor_set(v___x_456_, 2, v___x_453_);
lean_ctor_set(v___x_456_, 3, v___x_455_);
return v___x_456_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instCommGroup(lean_object* v_A_457_, lean_object* v_M_458_, lean_object* v_inst_459_, lean_object* v_inst_460_){
_start:
{
lean_object* v___x_461_; 
v___x_461_ = lp_mathlib_AddChar_instCommGroup___redArg(v_inst_459_, v_inst_460_);
return v___x_461_;
}
}
static lean_object* _init_lp_mathlib_AddChar_instAddCommGroup___aux__1___redArg___closed__0(void){
_start:
{
lean_object* v___x_462_; 
v___x_462_ = lp_mathlib_Multiplicative_ofAdd(lean_box(0));
return v___x_462_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommGroup___aux__1___redArg(lean_object* v_inst_463_, lean_object* v_inst_464_, lean_object* v_x_465_){
_start:
{
lean_object* v_toAddMonoid_466_; lean_object* v___x_467_; lean_object* v_toFun_468_; lean_object* v___x_469_; lean_object* v_toFun_470_; lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; 
v_toAddMonoid_466_ = lean_ctor_get(v_inst_463_, 0);
v___x_467_ = lean_obj_once(&lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__1, &lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__1_once, _init_lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__1);
v_toFun_468_ = lean_ctor_get(v___x_467_, 0);
v___x_469_ = lean_obj_once(&lp_mathlib_AddChar_instAddCommGroup___aux__1___redArg___closed__0, &lp_mathlib_AddChar_instAddCommGroup___aux__1___redArg___closed__0_once, _init_lp_mathlib_AddChar_instAddCommGroup___aux__1___redArg___closed__0);
v_toFun_470_ = lean_ctor_get(v___x_469_, 0);
lean_inc(v_toFun_468_);
v___x_471_ = lean_apply_1(v_toFun_468_, v_x_465_);
v___x_472_ = lp_mathlib_negAddMonoidHom___redArg(v_inst_463_);
v___x_473_ = lp_mathlib_AddChar_compAddMonoidHom___redArg(v_toAddMonoid_466_, v_inst_464_, v___x_471_, v___x_472_);
lean_inc(v_toFun_470_);
v___x_474_ = lean_apply_1(v_toFun_470_, v___x_473_);
return v___x_474_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommGroup___aux__1___redArg___boxed(lean_object* v_inst_475_, lean_object* v_inst_476_, lean_object* v_x_477_){
_start:
{
lean_object* v_res_478_; 
v_res_478_ = lp_mathlib_AddChar_instAddCommGroup___aux__1___redArg(v_inst_475_, v_inst_476_, v_x_477_);
lean_dec_ref(v_inst_476_);
lean_dec_ref(v_inst_475_);
return v_res_478_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommGroup___aux__1(lean_object* v_A_479_, lean_object* v_M_480_, lean_object* v_inst_481_, lean_object* v_inst_482_, lean_object* v_x_483_){
_start:
{
lean_object* v_toAddMonoid_484_; lean_object* v___x_485_; lean_object* v_toFun_486_; lean_object* v___x_487_; lean_object* v_toFun_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; 
v_toAddMonoid_484_ = lean_ctor_get(v_inst_481_, 0);
v___x_485_ = lean_obj_once(&lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__1, &lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__1_once, _init_lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__1);
v_toFun_486_ = lean_ctor_get(v___x_485_, 0);
v___x_487_ = lean_obj_once(&lp_mathlib_AddChar_instAddCommGroup___aux__1___redArg___closed__0, &lp_mathlib_AddChar_instAddCommGroup___aux__1___redArg___closed__0_once, _init_lp_mathlib_AddChar_instAddCommGroup___aux__1___redArg___closed__0);
v_toFun_488_ = lean_ctor_get(v___x_487_, 0);
lean_inc(v_toFun_486_);
v___x_489_ = lean_apply_1(v_toFun_486_, v_x_483_);
v___x_490_ = lp_mathlib_negAddMonoidHom___redArg(v_inst_481_);
v___x_491_ = lp_mathlib_AddChar_compAddMonoidHom___redArg(v_toAddMonoid_484_, v_inst_482_, v___x_489_, v___x_490_);
lean_inc(v_toFun_488_);
v___x_492_ = lean_apply_1(v_toFun_488_, v___x_491_);
return v___x_492_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommGroup___aux__1___boxed(lean_object* v_A_493_, lean_object* v_M_494_, lean_object* v_inst_495_, lean_object* v_inst_496_, lean_object* v_x_497_){
_start:
{
lean_object* v_res_498_; 
v_res_498_ = lp_mathlib_AddChar_instAddCommGroup___aux__1(v_A_493_, v_M_494_, v_inst_495_, v_inst_496_, v_x_497_);
lean_dec_ref(v_inst_496_);
lean_dec_ref(v_inst_495_);
return v_res_498_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommGroup___aux__3___redArg(lean_object* v_inst_499_, lean_object* v_inst_500_, lean_object* v_x_501_, lean_object* v_y_502_){
_start:
{
lean_object* v_toAddMonoid_503_; lean_object* v___x_504_; lean_object* v___x_505_; lean_object* v_toFun_506_; lean_object* v___x_507_; lean_object* v___x_508_; lean_object* v___x_509_; lean_object* v___f_510_; lean_object* v___x_511_; lean_object* v___x_512_; 
v_toAddMonoid_503_ = lean_ctor_get(v_inst_499_, 0);
lean_inc_ref_n(v_toAddMonoid_503_, 2);
v___x_504_ = lean_obj_once(&lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__0, &lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__0_once, _init_lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__0);
lean_inc_ref(v_inst_500_);
v___x_505_ = lp_mathlib_AddChar_instCommMonoid___redArg(v_toAddMonoid_503_, v_inst_500_);
v_toFun_506_ = lean_ctor_get(v___x_504_, 0);
v___x_507_ = lean_obj_once(&lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__1, &lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__1_once, _init_lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__1);
v___x_508_ = lean_alloc_closure((void*)(lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___lam__0), 3, 2);
lean_closure_set(v___x_508_, 0, v___x_507_);
lean_closure_set(v___x_508_, 1, v_x_501_);
v___x_509_ = lean_alloc_closure((void*)(lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___lam__0), 3, 2);
lean_closure_set(v___x_509_, 0, v___x_507_);
lean_closure_set(v___x_509_, 1, v_y_502_);
v___f_510_ = lean_alloc_closure((void*)(lp_mathlib_AddChar_instCommGroup___redArg___lam__0___boxed), 5, 3);
lean_closure_set(v___f_510_, 0, v_inst_499_);
lean_closure_set(v___f_510_, 1, v_toAddMonoid_503_);
lean_closure_set(v___f_510_, 2, v_inst_500_);
v___x_511_ = lp_mathlib_DivInvMonoid_div_x27___redArg(v___x_505_, v___f_510_, v___x_508_, v___x_509_);
lean_dec_ref(v___x_505_);
lean_inc(v_toFun_506_);
v___x_512_ = lean_apply_1(v_toFun_506_, v___x_511_);
return v___x_512_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommGroup___aux__3(lean_object* v_A_513_, lean_object* v_M_514_, lean_object* v_inst_515_, lean_object* v_inst_516_, lean_object* v_x_517_, lean_object* v_y_518_){
_start:
{
lean_object* v_toAddMonoid_519_; lean_object* v___x_520_; lean_object* v___x_521_; lean_object* v_toFun_522_; lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; lean_object* v___f_526_; lean_object* v___x_527_; lean_object* v___x_528_; 
v_toAddMonoid_519_ = lean_ctor_get(v_inst_515_, 0);
lean_inc_ref_n(v_toAddMonoid_519_, 2);
v___x_520_ = lean_obj_once(&lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__0, &lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__0_once, _init_lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__0);
lean_inc_ref(v_inst_516_);
v___x_521_ = lp_mathlib_AddChar_instCommMonoid___redArg(v_toAddMonoid_519_, v_inst_516_);
v_toFun_522_ = lean_ctor_get(v___x_520_, 0);
v___x_523_ = lean_obj_once(&lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__1, &lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__1_once, _init_lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__1);
v___x_524_ = lean_alloc_closure((void*)(lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___lam__0), 3, 2);
lean_closure_set(v___x_524_, 0, v___x_523_);
lean_closure_set(v___x_524_, 1, v_x_517_);
v___x_525_ = lean_alloc_closure((void*)(lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___lam__0), 3, 2);
lean_closure_set(v___x_525_, 0, v___x_523_);
lean_closure_set(v___x_525_, 1, v_y_518_);
v___f_526_ = lean_alloc_closure((void*)(lp_mathlib_AddChar_instCommGroup___redArg___lam__0___boxed), 5, 3);
lean_closure_set(v___f_526_, 0, v_inst_515_);
lean_closure_set(v___f_526_, 1, v_toAddMonoid_519_);
lean_closure_set(v___f_526_, 2, v_inst_516_);
v___x_527_ = lp_mathlib_DivInvMonoid_div_x27___redArg(v___x_521_, v___f_526_, v___x_524_, v___x_525_);
lean_dec_ref(v___x_521_);
lean_inc(v_toFun_522_);
v___x_528_ = lean_apply_1(v_toFun_522_, v___x_527_);
return v___x_528_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommGroup___aux__5___redArg(lean_object* v_inst_529_, lean_object* v_inst_530_, lean_object* v_n_531_, lean_object* v_a_532_){
_start:
{
lean_object* v___x_533_; lean_object* v_toFun_534_; lean_object* v_toAddMonoid_535_; lean_object* v___x_536_; lean_object* v_toFun_537_; lean_object* v___x_538_; lean_object* v___f_539_; lean_object* v___x_540_; lean_object* v___f_541_; lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; 
v___x_533_ = lean_obj_once(&lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__1, &lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__1_once, _init_lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__1);
v_toFun_534_ = lean_ctor_get(v___x_533_, 0);
v_toAddMonoid_535_ = lean_ctor_get(v_inst_529_, 0);
lean_inc_ref_n(v_toAddMonoid_535_, 2);
v___x_536_ = lean_obj_once(&lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__0, &lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__0_once, _init_lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__0);
v_toFun_537_ = lean_ctor_get(v___x_536_, 0);
lean_inc(v_toFun_534_);
v___x_538_ = lean_apply_1(v_toFun_534_, v_a_532_);
lean_inc_ref(v_inst_530_);
v___f_539_ = lean_alloc_closure((void*)(lp_mathlib_AddChar_instCommGroup___redArg___lam__0___boxed), 5, 3);
lean_closure_set(v___f_539_, 0, v_inst_529_);
lean_closure_set(v___f_539_, 1, v_toAddMonoid_535_);
lean_closure_set(v___f_539_, 2, v_inst_530_);
v___x_540_ = lp_mathlib_AddChar_instOne___redArg(v_toAddMonoid_535_, v_inst_530_);
v___f_541_ = lean_alloc_closure((void*)(lp_mathlib_AddChar_instCommMonoid___redArg___lam__3___boxed), 5, 2);
lean_closure_set(v___f_541_, 0, v_toAddMonoid_535_);
lean_closure_set(v___f_541_, 1, v_inst_530_);
v___x_542_ = lean_alloc_closure((void*)(l_npowRec___boxed), 5, 3);
lean_closure_set(v___x_542_, 0, lean_box(0));
lean_closure_set(v___x_542_, 1, v___x_540_);
lean_closure_set(v___x_542_, 2, v___f_541_);
v___x_543_ = lp_mathlib_zpowRec___redArg(v___f_539_, v___x_542_, v_n_531_, v___x_538_);
lean_inc(v_toFun_537_);
v___x_544_ = lean_apply_1(v_toFun_537_, v___x_543_);
return v___x_544_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommGroup___aux__5___redArg___boxed(lean_object* v_inst_545_, lean_object* v_inst_546_, lean_object* v_n_547_, lean_object* v_a_548_){
_start:
{
lean_object* v_res_549_; 
v_res_549_ = lp_mathlib_AddChar_instAddCommGroup___aux__5___redArg(v_inst_545_, v_inst_546_, v_n_547_, v_a_548_);
lean_dec(v_n_547_);
return v_res_549_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommGroup___aux__5(lean_object* v_A_550_, lean_object* v_M_551_, lean_object* v_inst_552_, lean_object* v_inst_553_, lean_object* v_n_554_, lean_object* v_a_555_){
_start:
{
lean_object* v___x_556_; lean_object* v_toFun_557_; lean_object* v_toAddMonoid_558_; lean_object* v___x_559_; lean_object* v_toFun_560_; lean_object* v___x_561_; lean_object* v___f_562_; lean_object* v___x_563_; lean_object* v___f_564_; lean_object* v___x_565_; lean_object* v___x_566_; lean_object* v___x_567_; 
v___x_556_ = lean_obj_once(&lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__1, &lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__1_once, _init_lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__1);
v_toFun_557_ = lean_ctor_get(v___x_556_, 0);
v_toAddMonoid_558_ = lean_ctor_get(v_inst_552_, 0);
lean_inc_ref_n(v_toAddMonoid_558_, 2);
v___x_559_ = lean_obj_once(&lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__0, &lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__0_once, _init_lp_mathlib_AddChar_instAddCommMonoid___aux__1___redArg___closed__0);
v_toFun_560_ = lean_ctor_get(v___x_559_, 0);
lean_inc(v_toFun_557_);
v___x_561_ = lean_apply_1(v_toFun_557_, v_a_555_);
lean_inc_ref(v_inst_553_);
v___f_562_ = lean_alloc_closure((void*)(lp_mathlib_AddChar_instCommGroup___redArg___lam__0___boxed), 5, 3);
lean_closure_set(v___f_562_, 0, v_inst_552_);
lean_closure_set(v___f_562_, 1, v_toAddMonoid_558_);
lean_closure_set(v___f_562_, 2, v_inst_553_);
v___x_563_ = lp_mathlib_AddChar_instOne___redArg(v_toAddMonoid_558_, v_inst_553_);
v___f_564_ = lean_alloc_closure((void*)(lp_mathlib_AddChar_instCommMonoid___redArg___lam__3___boxed), 5, 2);
lean_closure_set(v___f_564_, 0, v_toAddMonoid_558_);
lean_closure_set(v___f_564_, 1, v_inst_553_);
v___x_565_ = lean_alloc_closure((void*)(l_npowRec___boxed), 5, 3);
lean_closure_set(v___x_565_, 0, lean_box(0));
lean_closure_set(v___x_565_, 1, v___x_563_);
lean_closure_set(v___x_565_, 2, v___f_564_);
v___x_566_ = lp_mathlib_zpowRec___redArg(v___f_562_, v___x_565_, v_n_554_, v___x_561_);
lean_inc(v_toFun_560_);
v___x_567_ = lean_apply_1(v_toFun_560_, v___x_566_);
return v___x_567_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommGroup___aux__5___boxed(lean_object* v_A_568_, lean_object* v_M_569_, lean_object* v_inst_570_, lean_object* v_inst_571_, lean_object* v_n_572_, lean_object* v_a_573_){
_start:
{
lean_object* v_res_574_; 
v_res_574_ = lp_mathlib_AddChar_instAddCommGroup___aux__5(v_A_568_, v_M_569_, v_inst_570_, v_inst_571_, v_n_572_, v_a_573_);
lean_dec(v_n_572_);
return v_res_574_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommGroup___redArg(lean_object* v_inst_575_, lean_object* v_inst_576_){
_start:
{
lean_object* v_toAddMonoid_577_; lean_object* v___x_578_; lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; 
v_toAddMonoid_577_ = lean_ctor_get(v_inst_575_, 0);
lean_inc_ref_n(v_inst_576_, 3);
lean_inc_ref(v_toAddMonoid_577_);
v___x_578_ = lp_mathlib_AddChar_instAddCommMonoid___redArg(v_toAddMonoid_577_, v_inst_576_);
lean_inc_ref_n(v_inst_575_, 2);
v___x_579_ = lean_alloc_closure((void*)(lp_mathlib_AddChar_instAddCommGroup___aux__1___boxed), 5, 4);
lean_closure_set(v___x_579_, 0, lean_box(0));
lean_closure_set(v___x_579_, 1, lean_box(0));
lean_closure_set(v___x_579_, 2, v_inst_575_);
lean_closure_set(v___x_579_, 3, v_inst_576_);
v___x_580_ = lean_alloc_closure((void*)(lp_mathlib_AddChar_instAddCommGroup___aux__3), 6, 4);
lean_closure_set(v___x_580_, 0, lean_box(0));
lean_closure_set(v___x_580_, 1, lean_box(0));
lean_closure_set(v___x_580_, 2, v_inst_575_);
lean_closure_set(v___x_580_, 3, v_inst_576_);
v___x_581_ = lean_alloc_closure((void*)(lp_mathlib_AddChar_instAddCommGroup___aux__5___boxed), 6, 4);
lean_closure_set(v___x_581_, 0, lean_box(0));
lean_closure_set(v___x_581_, 1, lean_box(0));
lean_closure_set(v___x_581_, 2, v_inst_575_);
lean_closure_set(v___x_581_, 3, v_inst_576_);
v___x_582_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_582_, 0, v___x_578_);
lean_ctor_set(v___x_582_, 1, v___x_579_);
lean_ctor_set(v___x_582_, 2, v___x_580_);
lean_ctor_set(v___x_582_, 3, v___x_581_);
return v___x_582_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_instAddCommGroup(lean_object* v_A_583_, lean_object* v_M_584_, lean_object* v_inst_585_, lean_object* v_inst_586_){
_start:
{
lean_object* v___x_587_; 
v___x_587_ = lp_mathlib_AddChar_instAddCommGroup___redArg(v_inst_585_, v_inst_586_);
return v___x_587_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_mulShift___redArg(lean_object* v_inst_588_, lean_object* v_inst_589_, lean_object* v_00_u03c8_590_, lean_object* v_r_591_){
_start:
{
lean_object* v___x_592_; lean_object* v_toAddMonoidWithOne_593_; lean_object* v_toAddMonoid_594_; lean_object* v___x_595_; lean_object* v_toNonUnitalNonAssocRing_596_; lean_object* v___x_597_; lean_object* v___x_598_; lean_object* v___x_599_; 
lean_inc_ref(v_inst_588_);
v___x_592_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_inst_588_);
v_toAddMonoidWithOne_593_ = lean_ctor_get(v___x_592_, 1);
lean_inc_ref(v_toAddMonoidWithOne_593_);
lean_dec_ref(v___x_592_);
v_toAddMonoid_594_ = lean_ctor_get(v_toAddMonoidWithOne_593_, 1);
lean_inc_ref(v_toAddMonoid_594_);
lean_dec_ref(v_toAddMonoidWithOne_593_);
v___x_595_ = lp_mathlib_Ring_toNonAssocRing___redArg(v_inst_588_);
lean_dec_ref(v_inst_588_);
v_toNonUnitalNonAssocRing_596_ = lean_ctor_get(v___x_595_, 0);
lean_inc_ref(v_toNonUnitalNonAssocRing_596_);
lean_dec_ref(v___x_595_);
v___x_597_ = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(v_toNonUnitalNonAssocRing_596_);
v___x_598_ = lp_mathlib_AddMonoidHom_mulLeft___redArg(v___x_597_, v_r_591_);
v___x_599_ = lp_mathlib_AddChar_compAddMonoidHom___redArg(v_toAddMonoid_594_, v_inst_589_, v_00_u03c8_590_, v___x_598_);
lean_dec_ref(v_toAddMonoid_594_);
return v___x_599_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_mulShift___redArg___boxed(lean_object* v_inst_600_, lean_object* v_inst_601_, lean_object* v_00_u03c8_602_, lean_object* v_r_603_){
_start:
{
lean_object* v_res_604_; 
v_res_604_ = lp_mathlib_AddChar_mulShift___redArg(v_inst_600_, v_inst_601_, v_00_u03c8_602_, v_r_603_);
lean_dec_ref(v_inst_601_);
return v_res_604_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_mulShift(lean_object* v_R_605_, lean_object* v_M_606_, lean_object* v_inst_607_, lean_object* v_inst_608_, lean_object* v_00_u03c8_609_, lean_object* v_r_610_){
_start:
{
lean_object* v___x_611_; 
v___x_611_ = lp_mathlib_AddChar_mulShift___redArg(v_inst_607_, v_inst_608_, v_00_u03c8_609_, v_r_610_);
return v___x_611_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddChar_mulShift___boxed(lean_object* v_R_612_, lean_object* v_M_613_, lean_object* v_inst_614_, lean_object* v_inst_615_, lean_object* v_00_u03c8_616_, lean_object* v_r_617_){
_start:
{
lean_object* v_res_618_; 
v_res_618_ = lp_mathlib_AddChar_mulShift(v_R_612_, v_M_613_, v_inst_614_, v_inst_615_, v_00_u03c8_616_, v_r_617_);
lean_dec_ref(v_inst_615_);
return v_res_618_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Pi(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Ring_Finset(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Ker(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_TransferInstance(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Units_Equiv(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_AddChar(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Ring_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Ker(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_TransferInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Units_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_AddChar(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Ring_Finset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Ker(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_TransferInstance(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Units_Equiv(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_AddChar(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Ring_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Ker(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_TransferInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Units_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_AddChar(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_AddChar(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_AddChar(builtin);
}
#ifdef __cplusplus
}
#endif
