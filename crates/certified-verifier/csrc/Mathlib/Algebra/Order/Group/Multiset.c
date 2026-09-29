// Lean compiler output
// Module: Mathlib.Algebra.Order.Group.Multiset
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Hom.Defs public import Mathlib.Algebra.Group.Nat.Defs public import Mathlib.Algebra.Order.Monoid.Unbundled.ExistsOfLE public import Mathlib.Algebra.Order.Sub.Defs public import Mathlib.Data.Multiset.Dedup
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
lean_object* l_List_replicateTR___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_add(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_countP(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_card___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_map(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_nsmulRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Multiset_instAddCancelCommMonoid___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Multiset_add, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Multiset_instAddCancelCommMonoid___closed__0 = (const lean_object*)&lp_mathlib_Multiset_instAddCancelCommMonoid___closed__0_value;
static const lean_closure_object lp_mathlib_Multiset_instAddCancelCommMonoid___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_nsmulRec___boxed, .m_arity = 5, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Multiset_instAddCancelCommMonoid___closed__0_value)} };
static const lean_object* lp_mathlib_Multiset_instAddCancelCommMonoid___closed__1 = (const lean_object*)&lp_mathlib_Multiset_instAddCancelCommMonoid___closed__1_value;
static const lean_ctor_object lp_mathlib_Multiset_instAddCancelCommMonoid___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Multiset_instAddCancelCommMonoid___closed__0_value),((lean_object*)&lp_mathlib_Multiset_instAddCancelCommMonoid___closed__1_value)}};
static const lean_object* lp_mathlib_Multiset_instAddCancelCommMonoid___closed__2 = (const lean_object*)&lp_mathlib_Multiset_instAddCancelCommMonoid___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instAddCancelCommMonoid(lean_object*);
static const lean_closure_object lp_mathlib_Multiset_cardHom___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Multiset_card___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Multiset_cardHom___closed__0 = (const lean_object*)&lp_mathlib_Multiset_cardHom___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Multiset_cardHom(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_replicateAddMonoidHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_replicateAddMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_replicateAddMonoidHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_mapAddMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_mapAddMonoidHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_countPAddMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_countPAddMonoidHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_countAddMonoidHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_countAddMonoidHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instAddCancelCommMonoid(lean_object* v_00_u03b1_9_){
_start:
{
lean_object* v___x_10_; 
v___x_10_ = ((lean_object*)(lp_mathlib_Multiset_instAddCancelCommMonoid___closed__2));
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_cardHom(lean_object* v_00_u03b1_12_){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = ((lean_object*)(lp_mathlib_Multiset_cardHom___closed__0));
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_replicateAddMonoidHom___redArg___lam__0(lean_object* v_a_14_, lean_object* v_n_15_){
_start:
{
lean_object* v___x_16_; 
v___x_16_ = l_List_replicateTR___redArg(v_n_15_, v_a_14_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_replicateAddMonoidHom___redArg(lean_object* v_a_17_){
_start:
{
lean_object* v___f_18_; 
v___f_18_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_replicateAddMonoidHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_18_, 0, v_a_17_);
return v___f_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_replicateAddMonoidHom(lean_object* v_00_u03b1_19_, lean_object* v_a_20_){
_start:
{
lean_object* v___f_21_; 
v___f_21_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_replicateAddMonoidHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_21_, 0, v_a_20_);
return v___f_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_mapAddMonoidHom___redArg(lean_object* v_f_22_){
_start:
{
lean_object* v___x_23_; 
v___x_23_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_map), 4, 3);
lean_closure_set(v___x_23_, 0, lean_box(0));
lean_closure_set(v___x_23_, 1, lean_box(0));
lean_closure_set(v___x_23_, 2, v_f_22_);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_mapAddMonoidHom(lean_object* v_00_u03b1_24_, lean_object* v_00_u03b2_25_, lean_object* v_f_26_){
_start:
{
lean_object* v___x_27_; 
v___x_27_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_map), 4, 3);
lean_closure_set(v___x_27_, 0, lean_box(0));
lean_closure_set(v___x_27_, 1, lean_box(0));
lean_closure_set(v___x_27_, 2, v_f_26_);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_countPAddMonoidHom___redArg(lean_object* v_inst_28_){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_countP), 4, 3);
lean_closure_set(v___x_29_, 0, lean_box(0));
lean_closure_set(v___x_29_, 1, lean_box(0));
lean_closure_set(v___x_29_, 2, v_inst_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_countPAddMonoidHom(lean_object* v_00_u03b1_30_, lean_object* v_p_31_, lean_object* v_inst_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_countP), 4, 3);
lean_closure_set(v___x_33_, 0, lean_box(0));
lean_closure_set(v___x_33_, 1, lean_box(0));
lean_closure_set(v___x_33_, 2, v_inst_32_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_countAddMonoidHom___redArg(lean_object* v_inst_34_, lean_object* v_a_35_){
_start:
{
lean_object* v___x_36_; lean_object* v___x_37_; 
v___x_36_ = lean_apply_1(v_inst_34_, v_a_35_);
v___x_37_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_countP), 4, 3);
lean_closure_set(v___x_37_, 0, lean_box(0));
lean_closure_set(v___x_37_, 1, lean_box(0));
lean_closure_set(v___x_37_, 2, v___x_36_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_countAddMonoidHom(lean_object* v_00_u03b1_38_, lean_object* v_inst_39_, lean_object* v_a_40_){
_start:
{
lean_object* v___x_41_; 
v___x_41_ = lp_mathlib_Multiset_countAddMonoidHom___redArg(v_inst_39_, v_a_40_);
return v___x_41_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_ExistsOfLE(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Sub_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Dedup(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Multiset(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_ExistsOfLE(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Sub_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Dedup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Order_Group_Multiset(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_ExistsOfLE(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Sub_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Dedup(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Multiset(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_ExistsOfLE(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Sub_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Dedup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Multiset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Order_Group_Multiset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Order_Group_Multiset(builtin);
}
#ifdef __cplusplus
}
#endif
