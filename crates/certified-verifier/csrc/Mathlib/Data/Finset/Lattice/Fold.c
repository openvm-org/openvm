// Lean compiler output
// Module: Mathlib.Data.Finset.Lattice.Fold
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.Fold public import Mathlib.Data.Finset.Sum public import Mathlib.Data.Multiset.Lattice public import Mathlib.Data.Set.BooleanAlgebra public import Mathlib.Order.Hom.BoundedLattice public import Mathlib.Order.Nat
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
lean_object* lp_mathlib_Finset_fold___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_WithBot_some(lean_object*, lean_object*);
lean_object* lp_mathlib_WithBot_semilatticeSup___redArg(lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_WithTop_some(lean_object*, lean_object*);
lean_object* lp_mathlib_WithTop_semilatticeInf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_sup___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_sup___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_sup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_inf___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_inf___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_inf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Finset_sup_x27___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithBot_some, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Finset_sup_x27___redArg___closed__0 = (const lean_object*)&lp_mathlib_Finset_sup_x27___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Finset_sup_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_sup_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Finset_inf_x27___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithTop_some, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Finset_inf_x27___redArg___closed__0 = (const lean_object*)&lp_mathlib_Finset_inf_x27___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Finset_inf_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_inf_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_sup___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_x1_2_, lean_object* v_x2_3_){
_start:
{
lean_object* v_sup_4_; lean_object* v___x_5_; 
v_sup_4_ = lean_ctor_get(v_inst_1_, 1);
lean_inc(v_sup_4_);
lean_dec_ref(v_inst_1_);
v___x_5_ = lean_apply_2(v_sup_4_, v_x1_2_, v_x2_3_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_sup___redArg(lean_object* v_inst_6_, lean_object* v_inst_7_, lean_object* v_s_8_, lean_object* v_f_9_){
_start:
{
lean_object* v___f_10_; lean_object* v___x_11_; 
v___f_10_ = lean_alloc_closure((void*)(lp_mathlib_Finset_sup___redArg___lam__0), 3, 1);
lean_closure_set(v___f_10_, 0, v_inst_6_);
v___x_11_ = lp_mathlib_Finset_fold___redArg(v___f_10_, v_inst_7_, v_f_9_, v_s_8_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_sup(lean_object* v_00_u03b1_12_, lean_object* v_00_u03b2_13_, lean_object* v_inst_14_, lean_object* v_inst_15_, lean_object* v_s_16_, lean_object* v_f_17_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = lp_mathlib_Finset_sup___redArg(v_inst_14_, v_inst_15_, v_s_16_, v_f_17_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_inf___redArg___lam__0(lean_object* v_inst_19_, lean_object* v_x1_20_, lean_object* v_x2_21_){
_start:
{
lean_object* v_inf_22_; lean_object* v___x_23_; 
v_inf_22_ = lean_ctor_get(v_inst_19_, 1);
lean_inc(v_inf_22_);
lean_dec_ref(v_inst_19_);
v___x_23_ = lean_apply_2(v_inf_22_, v_x1_20_, v_x2_21_);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_inf___redArg(lean_object* v_inst_24_, lean_object* v_inst_25_, lean_object* v_s_26_, lean_object* v_f_27_){
_start:
{
lean_object* v___f_28_; lean_object* v___x_29_; 
v___f_28_ = lean_alloc_closure((void*)(lp_mathlib_Finset_inf___redArg___lam__0), 3, 1);
lean_closure_set(v___f_28_, 0, v_inst_24_);
v___x_29_ = lp_mathlib_Finset_fold___redArg(v___f_28_, v_inst_25_, v_f_27_, v_s_26_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_inf(lean_object* v_00_u03b1_30_, lean_object* v_00_u03b2_31_, lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_s_34_, lean_object* v_f_35_){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lp_mathlib_Finset_inf___redArg(v_inst_32_, v_inst_33_, v_s_34_, v_f_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_sup_x27___redArg(lean_object* v_inst_38_, lean_object* v_s_39_, lean_object* v_f_40_){
_start:
{
lean_object* v___x_41_; lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___x_44_; lean_object* v___x_45_; lean_object* v_val_46_; 
v___x_41_ = lp_mathlib_WithBot_semilatticeSup___redArg(v_inst_38_);
v___x_42_ = lean_box(0);
v___x_43_ = ((lean_object*)(lp_mathlib_Finset_sup_x27___redArg___closed__0));
v___x_44_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_44_, 0, lean_box(0));
lean_closure_set(v___x_44_, 1, lean_box(0));
lean_closure_set(v___x_44_, 2, lean_box(0));
lean_closure_set(v___x_44_, 3, v___x_43_);
lean_closure_set(v___x_44_, 4, v_f_40_);
v___x_45_ = lp_mathlib_Finset_sup___redArg(v___x_41_, v___x_42_, v_s_39_, v___x_44_);
v_val_46_ = lean_ctor_get(v___x_45_, 0);
lean_inc(v_val_46_);
lean_dec(v___x_45_);
return v_val_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_sup_x27(lean_object* v_00_u03b1_47_, lean_object* v_00_u03b2_48_, lean_object* v_inst_49_, lean_object* v_s_50_, lean_object* v_H_51_, lean_object* v_f_52_){
_start:
{
lean_object* v___x_53_; 
v___x_53_ = lp_mathlib_Finset_sup_x27___redArg(v_inst_49_, v_s_50_, v_f_52_);
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_inf_x27___redArg(lean_object* v_inst_55_, lean_object* v_s_56_, lean_object* v_f_57_){
_start:
{
lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; lean_object* v_val_63_; 
v___x_58_ = lp_mathlib_WithTop_semilatticeInf___redArg(v_inst_55_);
v___x_59_ = lean_box(0);
v___x_60_ = ((lean_object*)(lp_mathlib_Finset_inf_x27___redArg___closed__0));
v___x_61_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_61_, 0, lean_box(0));
lean_closure_set(v___x_61_, 1, lean_box(0));
lean_closure_set(v___x_61_, 2, lean_box(0));
lean_closure_set(v___x_61_, 3, v___x_60_);
lean_closure_set(v___x_61_, 4, v_f_57_);
v___x_62_ = lp_mathlib_Finset_inf___redArg(v___x_58_, v___x_59_, v_s_56_, v___x_61_);
v_val_63_ = lean_ctor_get(v___x_62_, 0);
lean_inc(v_val_63_);
lean_dec(v___x_62_);
return v_val_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_inf_x27(lean_object* v_00_u03b1_64_, lean_object* v_00_u03b2_65_, lean_object* v_inst_66_, lean_object* v_s_67_, lean_object* v_H_68_, lean_object* v_f_69_){
_start:
{
lean_object* v___x_70_; 
v___x_70_ = lp_mathlib_Finset_inf_x27___redArg(v_inst_66_, v_s_67_, v_f_69_);
return v___x_70_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Fold(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Sum(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Lattice(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_BooleanAlgebra(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Hom_BoundedLattice(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Nat(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Lattice_Fold(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Fold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Sum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_BooleanAlgebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Hom_BoundedLattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Finset_Lattice_Fold(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Finset_Fold(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Sum(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Lattice(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_BooleanAlgebra(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Hom_BoundedLattice(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Nat(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_Lattice_Fold(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Fold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Sum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_BooleanAlgebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Hom_BoundedLattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Lattice_Fold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Finset_Lattice_Fold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Finset_Lattice_Fold(builtin);
}
#ifdef __cplusplus
}
#endif
