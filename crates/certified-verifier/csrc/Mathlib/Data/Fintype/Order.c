// Lean compiler output
// Module: Mathlib.Data.Fintype.Order
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.Lattice.Fold public import Mathlib.Data.Finset.Order public import Mathlib.Data.Set.Finite.Basic public import Mathlib.Order.Atoms import Mathlib.Basic.Finite.Prod import Mathlib.Order.ConditionallyCompleteLattice.Finset
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
lean_object* l_id___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_sup_x27___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_inf_x27___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lattice_toSemilatticeInf___redArg(lean_object*);
static const lean_closure_object lp_mathlib_Fintype_toOrderBot___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_id___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Fintype_toOrderBot___redArg___closed__0 = (const lean_object*)&lp_mathlib_Fintype_toOrderBot___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toOrderBot___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toOrderBot(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toOrderTop___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toOrderTop(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toBoundedOrder___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toBoundedOrder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toOrderBot___redArg(lean_object* v_inst_2_, lean_object* v_inst_3_){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; 
v___x_4_ = ((lean_object*)(lp_mathlib_Fintype_toOrderBot___redArg___closed__0));
v___x_5_ = lp_mathlib_Finset_inf_x27___redArg(v_inst_3_, v_inst_2_, v___x_4_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toOrderBot(lean_object* v_00_u03b1_6_, lean_object* v_inst_7_, lean_object* v_inst_8_, lean_object* v_inst_9_){
_start:
{
lean_object* v___x_10_; lean_object* v___x_11_; 
v___x_10_ = ((lean_object*)(lp_mathlib_Fintype_toOrderBot___redArg___closed__0));
v___x_11_ = lp_mathlib_Finset_inf_x27___redArg(v_inst_9_, v_inst_7_, v___x_10_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toOrderTop___redArg(lean_object* v_inst_12_, lean_object* v_inst_13_){
_start:
{
lean_object* v___x_14_; lean_object* v___x_15_; 
v___x_14_ = ((lean_object*)(lp_mathlib_Fintype_toOrderBot___redArg___closed__0));
v___x_15_ = lp_mathlib_Finset_sup_x27___redArg(v_inst_13_, v_inst_12_, v___x_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toOrderTop(lean_object* v_00_u03b1_16_, lean_object* v_inst_17_, lean_object* v_inst_18_, lean_object* v_inst_19_){
_start:
{
lean_object* v___x_20_; lean_object* v___x_21_; 
v___x_20_ = ((lean_object*)(lp_mathlib_Fintype_toOrderBot___redArg___closed__0));
v___x_21_ = lp_mathlib_Finset_sup_x27___redArg(v_inst_19_, v_inst_17_, v___x_20_);
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toBoundedOrder___redArg(lean_object* v_inst_22_, lean_object* v_inst_23_){
_start:
{
lean_object* v___x_24_; lean_object* v_toSemilatticeSup_25_; lean_object* v___x_27_; uint8_t v_isShared_28_; uint8_t v_isSharedCheck_35_; 
lean_inc_ref(v_inst_23_);
v___x_24_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_inst_23_);
v_toSemilatticeSup_25_ = lean_ctor_get(v_inst_23_, 0);
v_isSharedCheck_35_ = !lean_is_exclusive(v_inst_23_);
if (v_isSharedCheck_35_ == 0)
{
lean_object* v_unused_36_; 
v_unused_36_ = lean_ctor_get(v_inst_23_, 1);
lean_dec(v_unused_36_);
v___x_27_ = v_inst_23_;
v_isShared_28_ = v_isSharedCheck_35_;
goto v_resetjp_26_;
}
else
{
lean_inc(v_toSemilatticeSup_25_);
lean_dec(v_inst_23_);
v___x_27_ = lean_box(0);
v_isShared_28_ = v_isSharedCheck_35_;
goto v_resetjp_26_;
}
v_resetjp_26_:
{
lean_object* v___x_29_; lean_object* v___x_30_; lean_object* v___x_31_; lean_object* v___x_33_; 
v___x_29_ = ((lean_object*)(lp_mathlib_Fintype_toOrderBot___redArg___closed__0));
lean_inc(v_inst_22_);
v___x_30_ = lp_mathlib_Finset_inf_x27___redArg(v___x_24_, v_inst_22_, v___x_29_);
v___x_31_ = lp_mathlib_Finset_sup_x27___redArg(v_toSemilatticeSup_25_, v_inst_22_, v___x_29_);
if (v_isShared_28_ == 0)
{
lean_ctor_set(v___x_27_, 1, v___x_30_);
lean_ctor_set(v___x_27_, 0, v___x_31_);
v___x_33_ = v___x_27_;
goto v_reusejp_32_;
}
else
{
lean_object* v_reuseFailAlloc_34_; 
v_reuseFailAlloc_34_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_34_, 0, v___x_31_);
lean_ctor_set(v_reuseFailAlloc_34_, 1, v___x_30_);
v___x_33_ = v_reuseFailAlloc_34_;
goto v_reusejp_32_;
}
v_reusejp_32_:
{
return v___x_33_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toBoundedOrder(lean_object* v_00_u03b1_37_, lean_object* v_inst_38_, lean_object* v_inst_39_, lean_object* v_inst_40_){
_start:
{
lean_object* v___x_41_; lean_object* v_toSemilatticeSup_42_; lean_object* v___x_44_; uint8_t v_isShared_45_; uint8_t v_isSharedCheck_52_; 
lean_inc_ref(v_inst_40_);
v___x_41_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_inst_40_);
v_toSemilatticeSup_42_ = lean_ctor_get(v_inst_40_, 0);
v_isSharedCheck_52_ = !lean_is_exclusive(v_inst_40_);
if (v_isSharedCheck_52_ == 0)
{
lean_object* v_unused_53_; 
v_unused_53_ = lean_ctor_get(v_inst_40_, 1);
lean_dec(v_unused_53_);
v___x_44_ = v_inst_40_;
v_isShared_45_ = v_isSharedCheck_52_;
goto v_resetjp_43_;
}
else
{
lean_inc(v_toSemilatticeSup_42_);
lean_dec(v_inst_40_);
v___x_44_ = lean_box(0);
v_isShared_45_ = v_isSharedCheck_52_;
goto v_resetjp_43_;
}
v_resetjp_43_:
{
lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; lean_object* v___x_50_; 
v___x_46_ = ((lean_object*)(lp_mathlib_Fintype_toOrderBot___redArg___closed__0));
lean_inc(v_inst_38_);
v___x_47_ = lp_mathlib_Finset_inf_x27___redArg(v___x_41_, v_inst_38_, v___x_46_);
v___x_48_ = lp_mathlib_Finset_sup_x27___redArg(v_toSemilatticeSup_42_, v_inst_38_, v___x_46_);
if (v_isShared_45_ == 0)
{
lean_ctor_set(v___x_44_, 1, v___x_47_);
lean_ctor_set(v___x_44_, 0, v___x_48_);
v___x_50_ = v___x_44_;
goto v_reusejp_49_;
}
else
{
lean_object* v_reuseFailAlloc_51_; 
v_reuseFailAlloc_51_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_51_, 0, v___x_48_);
lean_ctor_set(v_reuseFailAlloc_51_, 1, v___x_47_);
v___x_50_ = v_reuseFailAlloc_51_;
goto v_reusejp_49_;
}
v_reusejp_49_:
{
return v___x_50_;
}
}
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Lattice_Fold(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Order(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Atoms(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_Finite_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Finset(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Order(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Lattice_Fold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Atoms(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Finite_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Fintype_Order(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Finset_Lattice_Fold(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Order(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Finite_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Atoms(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Basic_Finite_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Finset(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Fintype_Order(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Lattice_Fold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Finite_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Atoms(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_Finite_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Fintype_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Fintype_Order(builtin);
}
#ifdef __cplusplus
}
#endif
