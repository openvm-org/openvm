// Lean compiler output
// Module: Mathlib.Data.Rat.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Defs public import Mathlib.Data.Nat.Basic public import Mathlib.Data.Rat.Init public import Mathlib.Order.Basic public import Mathlib.Tactic.Common public import Mathlib.Tactic.Attr.Core
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
lean_object* l_Rat_ofInt(lean_object*);
lean_object* l_Rat_mul(lean_object*, lean_object*);
lean_object* l_Rat_sub(lean_object*, lean_object*);
lean_object* l_Rat_neg(lean_object*);
lean_object* l_Rat_instNatCast___lam__0(lean_object*);
lean_object* l_Rat_add(lean_object*, lean_object*);
lean_object* l_Rat_pow(lean_object*, lean_object*);
lean_object* l_Rat_mul___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_addCommGroup___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_addCommGroup___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Rat_addCommGroup___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Rat_addCommGroup___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Rat_addCommGroup___closed__0 = (const lean_object*)&lp_mathlib_Rat_addCommGroup___closed__0_value;
static const lean_closure_object lp_mathlib_Rat_addCommGroup___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Rat_addCommGroup___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Rat_addCommGroup___closed__1 = (const lean_object*)&lp_mathlib_Rat_addCommGroup___closed__1_value;
static const lean_closure_object lp_mathlib_Rat_addCommGroup___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Rat_add, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Rat_addCommGroup___closed__2 = (const lean_object*)&lp_mathlib_Rat_addCommGroup___closed__2_value;
static const lean_closure_object lp_mathlib_Rat_addCommGroup___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Rat_neg, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Rat_addCommGroup___closed__3 = (const lean_object*)&lp_mathlib_Rat_addCommGroup___closed__3_value;
static const lean_closure_object lp_mathlib_Rat_addCommGroup___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Rat_sub, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Rat_addCommGroup___closed__4 = (const lean_object*)&lp_mathlib_Rat_addCommGroup___closed__4_value;
static lean_once_cell_t lp_mathlib_Rat_addCommGroup___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Rat_addCommGroup___closed__5;
static lean_once_cell_t lp_mathlib_Rat_addCommGroup___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Rat_addCommGroup___closed__6;
static lean_once_cell_t lp_mathlib_Rat_addCommGroup___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Rat_addCommGroup___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Rat_addCommGroup;
LEAN_EXPORT lean_object* lp_mathlib_Rat_addGroup;
LEAN_EXPORT lean_object* lp_mathlib_Rat_addCommMonoid;
LEAN_EXPORT lean_object* lp_mathlib_Rat_addMonoid;
LEAN_EXPORT lean_object* lp_mathlib_Rat_addLeftCancelSemigroup;
LEAN_EXPORT lean_object* lp_mathlib_Rat_addRightCancelSemigroup;
LEAN_EXPORT lean_object* lp_mathlib_Rat_addCommSemigroup;
LEAN_EXPORT lean_object* lp_mathlib_Rat_addSemigroup;
LEAN_EXPORT lean_object* lp_mathlib_Rat_commMonoid___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_commMonoid___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Rat_commMonoid___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Rat_commMonoid___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Rat_commMonoid___closed__0 = (const lean_object*)&lp_mathlib_Rat_commMonoid___closed__0_value;
static const lean_closure_object lp_mathlib_Rat_commMonoid___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Rat_mul___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Rat_commMonoid___closed__1 = (const lean_object*)&lp_mathlib_Rat_commMonoid___closed__1_value;
static lean_once_cell_t lp_mathlib_Rat_commMonoid___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Rat_commMonoid___closed__2;
static lean_once_cell_t lp_mathlib_Rat_commMonoid___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Rat_commMonoid___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Rat_commMonoid;
LEAN_EXPORT lean_object* lp_mathlib_Rat_monoid;
LEAN_EXPORT lean_object* lp_mathlib_Rat_commSemigroup;
LEAN_EXPORT lean_object* lp_mathlib_Rat_semigroup;
LEAN_EXPORT lean_object* lp_mathlib_Rat_divCasesOn___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_divCasesOn(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_addCommGroup___lam__0(lean_object* v_x1_1_, lean_object* v_x2_2_){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; 
v___x_3_ = l_Rat_instNatCast___lam__0(v_x1_1_);
v___x_4_ = l_Rat_mul(v___x_3_, v_x2_2_);
lean_dec_ref(v___x_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_addCommGroup___lam__1(lean_object* v_x1_5_, lean_object* v_x2_6_){
_start:
{
lean_object* v___x_7_; lean_object* v___x_8_; 
v___x_7_ = l_Rat_ofInt(v_x1_5_);
v___x_8_ = l_Rat_mul(v___x_7_, v_x2_6_);
lean_dec_ref(v___x_7_);
return v___x_8_;
}
}
static lean_object* _init_lp_mathlib_Rat_addCommGroup___closed__5(void){
_start:
{
lean_object* v___x_14_; lean_object* v___x_15_; 
v___x_14_ = lean_unsigned_to_nat(0u);
v___x_15_ = l_Rat_instNatCast___lam__0(v___x_14_);
return v___x_15_;
}
}
static lean_object* _init_lp_mathlib_Rat_addCommGroup___closed__6(void){
_start:
{
lean_object* v___f_16_; lean_object* v___x_17_; lean_object* v___x_18_; lean_object* v___x_19_; 
v___f_16_ = ((lean_object*)(lp_mathlib_Rat_addCommGroup___closed__0));
v___x_17_ = ((lean_object*)(lp_mathlib_Rat_addCommGroup___closed__2));
v___x_18_ = lean_obj_once(&lp_mathlib_Rat_addCommGroup___closed__5, &lp_mathlib_Rat_addCommGroup___closed__5_once, _init_lp_mathlib_Rat_addCommGroup___closed__5);
v___x_19_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_19_, 0, v___x_18_);
lean_ctor_set(v___x_19_, 1, v___x_17_);
lean_ctor_set(v___x_19_, 2, v___f_16_);
return v___x_19_;
}
}
static lean_object* _init_lp_mathlib_Rat_addCommGroup___closed__7(void){
_start:
{
lean_object* v___f_20_; lean_object* v___x_21_; lean_object* v___x_22_; lean_object* v___x_23_; lean_object* v___x_24_; 
v___f_20_ = ((lean_object*)(lp_mathlib_Rat_addCommGroup___closed__1));
v___x_21_ = ((lean_object*)(lp_mathlib_Rat_addCommGroup___closed__4));
v___x_22_ = ((lean_object*)(lp_mathlib_Rat_addCommGroup___closed__3));
v___x_23_ = lean_obj_once(&lp_mathlib_Rat_addCommGroup___closed__6, &lp_mathlib_Rat_addCommGroup___closed__6_once, _init_lp_mathlib_Rat_addCommGroup___closed__6);
v___x_24_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_24_, 0, v___x_23_);
lean_ctor_set(v___x_24_, 1, v___x_22_);
lean_ctor_set(v___x_24_, 2, v___x_21_);
lean_ctor_set(v___x_24_, 3, v___f_20_);
return v___x_24_;
}
}
static lean_object* _init_lp_mathlib_Rat_addCommGroup(void){
_start:
{
lean_object* v___x_25_; 
v___x_25_ = lean_obj_once(&lp_mathlib_Rat_addCommGroup___closed__7, &lp_mathlib_Rat_addCommGroup___closed__7_once, _init_lp_mathlib_Rat_addCommGroup___closed__7);
return v___x_25_;
}
}
static lean_object* _init_lp_mathlib_Rat_addGroup(void){
_start:
{
lean_object* v___x_26_; 
v___x_26_ = lp_mathlib_Rat_addCommGroup;
return v___x_26_;
}
}
static lean_object* _init_lp_mathlib_Rat_addCommMonoid(void){
_start:
{
lean_object* v___x_27_; lean_object* v_toAddMonoid_28_; 
v___x_27_ = lp_mathlib_Rat_addCommGroup;
v_toAddMonoid_28_ = lean_ctor_get(v___x_27_, 0);
lean_inc_ref(v_toAddMonoid_28_);
return v_toAddMonoid_28_;
}
}
static lean_object* _init_lp_mathlib_Rat_addMonoid(void){
_start:
{
lean_object* v___x_29_; lean_object* v_toAddMonoid_30_; 
v___x_29_ = lp_mathlib_Rat_addCommGroup;
v_toAddMonoid_30_ = lean_ctor_get(v___x_29_, 0);
lean_inc_ref(v_toAddMonoid_30_);
return v_toAddMonoid_30_;
}
}
static lean_object* _init_lp_mathlib_Rat_addLeftCancelSemigroup(void){
_start:
{
lean_object* v___x_31_; lean_object* v_toAddMonoid_32_; lean_object* v_toAdd_33_; 
v___x_31_ = lp_mathlib_Rat_addCommGroup;
v_toAddMonoid_32_ = lean_ctor_get(v___x_31_, 0);
v_toAdd_33_ = lean_ctor_get(v_toAddMonoid_32_, 1);
lean_inc(v_toAdd_33_);
return v_toAdd_33_;
}
}
static lean_object* _init_lp_mathlib_Rat_addRightCancelSemigroup(void){
_start:
{
lean_object* v___x_34_; lean_object* v_toAddMonoid_35_; lean_object* v_toAdd_36_; 
v___x_34_ = lp_mathlib_Rat_addCommGroup;
v_toAddMonoid_35_ = lean_ctor_get(v___x_34_, 0);
v_toAdd_36_ = lean_ctor_get(v_toAddMonoid_35_, 1);
lean_inc(v_toAdd_36_);
return v_toAdd_36_;
}
}
static lean_object* _init_lp_mathlib_Rat_addCommSemigroup(void){
_start:
{
lean_object* v___x_37_; lean_object* v_toAdd_38_; 
v___x_37_ = lp_mathlib_Rat_addCommMonoid;
v_toAdd_38_ = lean_ctor_get(v___x_37_, 1);
lean_inc(v_toAdd_38_);
return v_toAdd_38_;
}
}
static lean_object* _init_lp_mathlib_Rat_addSemigroup(void){
_start:
{
lean_object* v___x_39_; lean_object* v_toAdd_40_; 
v___x_39_ = lp_mathlib_Rat_addMonoid;
v_toAdd_40_ = lean_ctor_get(v___x_39_, 1);
lean_inc(v_toAdd_40_);
return v_toAdd_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_commMonoid___lam__0(lean_object* v_n_41_, lean_object* v_q_42_){
_start:
{
lean_object* v___x_43_; 
v___x_43_ = l_Rat_pow(v_q_42_, v_n_41_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_commMonoid___lam__0___boxed(lean_object* v_n_44_, lean_object* v_q_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_mathlib_Rat_commMonoid___lam__0(v_n_44_, v_q_45_);
lean_dec(v_n_44_);
return v_res_46_;
}
}
static lean_object* _init_lp_mathlib_Rat_commMonoid___closed__2(void){
_start:
{
lean_object* v___x_49_; lean_object* v___x_50_; 
v___x_49_ = lean_unsigned_to_nat(1u);
v___x_50_ = l_Rat_instNatCast___lam__0(v___x_49_);
return v___x_50_;
}
}
static lean_object* _init_lp_mathlib_Rat_commMonoid___closed__3(void){
_start:
{
lean_object* v___f_51_; lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; 
v___f_51_ = ((lean_object*)(lp_mathlib_Rat_commMonoid___closed__0));
v___x_52_ = ((lean_object*)(lp_mathlib_Rat_commMonoid___closed__1));
v___x_53_ = lean_obj_once(&lp_mathlib_Rat_commMonoid___closed__2, &lp_mathlib_Rat_commMonoid___closed__2_once, _init_lp_mathlib_Rat_commMonoid___closed__2);
v___x_54_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_54_, 0, v___x_53_);
lean_ctor_set(v___x_54_, 1, v___x_52_);
lean_ctor_set(v___x_54_, 2, v___f_51_);
return v___x_54_;
}
}
static lean_object* _init_lp_mathlib_Rat_commMonoid(void){
_start:
{
lean_object* v___x_55_; 
v___x_55_ = lean_obj_once(&lp_mathlib_Rat_commMonoid___closed__3, &lp_mathlib_Rat_commMonoid___closed__3_once, _init_lp_mathlib_Rat_commMonoid___closed__3);
return v___x_55_;
}
}
static lean_object* _init_lp_mathlib_Rat_monoid(void){
_start:
{
lean_object* v___x_56_; 
v___x_56_ = lp_mathlib_Rat_commMonoid;
return v___x_56_;
}
}
static lean_object* _init_lp_mathlib_Rat_commSemigroup(void){
_start:
{
lean_object* v___x_57_; lean_object* v_toMul_58_; 
v___x_57_ = lp_mathlib_Rat_commMonoid;
v_toMul_58_ = lean_ctor_get(v___x_57_, 1);
lean_inc(v_toMul_58_);
return v_toMul_58_;
}
}
static lean_object* _init_lp_mathlib_Rat_semigroup(void){
_start:
{
lean_object* v___x_59_; lean_object* v_toMul_60_; 
v___x_59_ = lp_mathlib_Rat_commMonoid;
v_toMul_60_ = lean_ctor_get(v___x_59_, 1);
lean_inc(v_toMul_60_);
return v_toMul_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_divCasesOn___redArg(lean_object* v_a_61_, lean_object* v_div_62_){
_start:
{
lean_object* v_n_63_; lean_object* v_d_64_; lean_object* v___x_65_; 
v_n_63_ = lean_ctor_get(v_a_61_, 0);
lean_inc(v_n_63_);
v_d_64_ = lean_ctor_get(v_a_61_, 1);
lean_inc(v_d_64_);
lean_dec_ref(v_a_61_);
v___x_65_ = lean_apply_4(v_div_62_, v_n_63_, v_d_64_, lean_box(0), lean_box(0));
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_divCasesOn(lean_object* v_C_66_, lean_object* v_a_67_, lean_object* v_div_68_){
_start:
{
lean_object* v___x_69_; 
v___x_69_ = lp_mathlib_Rat_divCasesOn___redArg(v_a_67_, v_div_68_);
return v___x_69_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Rat_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Common(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Attr_Core(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Rat_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Rat_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Attr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Rat_addCommGroup = _init_lp_mathlib_Rat_addCommGroup();
lean_mark_persistent(lp_mathlib_Rat_addCommGroup);
lp_mathlib_Rat_addGroup = _init_lp_mathlib_Rat_addGroup();
lean_mark_persistent(lp_mathlib_Rat_addGroup);
lp_mathlib_Rat_addCommMonoid = _init_lp_mathlib_Rat_addCommMonoid();
lean_mark_persistent(lp_mathlib_Rat_addCommMonoid);
lp_mathlib_Rat_addMonoid = _init_lp_mathlib_Rat_addMonoid();
lean_mark_persistent(lp_mathlib_Rat_addMonoid);
lp_mathlib_Rat_addLeftCancelSemigroup = _init_lp_mathlib_Rat_addLeftCancelSemigroup();
lean_mark_persistent(lp_mathlib_Rat_addLeftCancelSemigroup);
lp_mathlib_Rat_addRightCancelSemigroup = _init_lp_mathlib_Rat_addRightCancelSemigroup();
lean_mark_persistent(lp_mathlib_Rat_addRightCancelSemigroup);
lp_mathlib_Rat_addCommSemigroup = _init_lp_mathlib_Rat_addCommSemigroup();
lean_mark_persistent(lp_mathlib_Rat_addCommSemigroup);
lp_mathlib_Rat_addSemigroup = _init_lp_mathlib_Rat_addSemigroup();
lean_mark_persistent(lp_mathlib_Rat_addSemigroup);
lp_mathlib_Rat_commMonoid = _init_lp_mathlib_Rat_commMonoid();
lean_mark_persistent(lp_mathlib_Rat_commMonoid);
lp_mathlib_Rat_monoid = _init_lp_mathlib_Rat_monoid();
lean_mark_persistent(lp_mathlib_Rat_monoid);
lp_mathlib_Rat_commSemigroup = _init_lp_mathlib_Rat_commSemigroup();
lean_mark_persistent(lp_mathlib_Rat_commSemigroup);
lp_mathlib_Rat_semigroup = _init_lp_mathlib_Rat_semigroup();
lean_mark_persistent(lp_mathlib_Rat_semigroup);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Rat_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Rat_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Common(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Attr_Core(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Rat_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Rat_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Attr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Rat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Rat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Rat_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
