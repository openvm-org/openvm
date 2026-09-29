// Lean compiler output
// Module: Mathlib.Data.Finset.Lattice.Basic
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.Defs public import Mathlib.Data.Multiset.FinsetOps
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
lean_object* lp_mathlib_Finset_instPartialOrder(lean_object*);
lean_object* lp_mathlib_Multiset_ndunion___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_ndinter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instUnion___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instUnion___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instUnion(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instInter___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instInter___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instInter(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instLattice___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instLattice___redArg___lam__1(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Finset_instLattice___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_instLattice___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Finset_instLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instDistribLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instDistribLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instUnion___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_s_2_, lean_object* v_t_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lp_mathlib_Multiset_ndunion___redArg(v_inst_1_, v_s_2_, v_t_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instUnion___redArg(lean_object* v_inst_5_){
_start:
{
lean_object* v___f_6_; 
v___f_6_ = lean_alloc_closure((void*)(lp_mathlib_Finset_instUnion___redArg___lam__0), 3, 1);
lean_closure_set(v___f_6_, 0, v_inst_5_);
return v___f_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instUnion(lean_object* v_00_u03b1_7_, lean_object* v_inst_8_){
_start:
{
lean_object* v___f_9_; 
v___f_9_ = lean_alloc_closure((void*)(lp_mathlib_Finset_instUnion___redArg___lam__0), 3, 1);
lean_closure_set(v___f_9_, 0, v_inst_8_);
return v___f_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instInter___redArg___lam__0(lean_object* v_inst_10_, lean_object* v_s_11_, lean_object* v_t_12_){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = lp_mathlib_Multiset_ndinter___redArg(v_inst_10_, v_s_11_, v_t_12_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instInter___redArg(lean_object* v_inst_14_){
_start:
{
lean_object* v___f_15_; 
v___f_15_ = lean_alloc_closure((void*)(lp_mathlib_Finset_instInter___redArg___lam__0), 3, 1);
lean_closure_set(v___f_15_, 0, v_inst_14_);
return v___f_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instInter(lean_object* v_00_u03b1_16_, lean_object* v_inst_17_){
_start:
{
lean_object* v___f_18_; 
v___f_18_ = lean_alloc_closure((void*)(lp_mathlib_Finset_instInter___redArg___lam__0), 3, 1);
lean_closure_set(v___f_18_, 0, v_inst_17_);
return v___f_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instLattice___redArg___lam__0(lean_object* v_inst_19_, lean_object* v_x1_20_, lean_object* v_x2_21_){
_start:
{
lean_object* v___x_22_; 
v___x_22_ = lp_mathlib_Multiset_ndunion___redArg(v_inst_19_, v_x1_20_, v_x2_21_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instLattice___redArg___lam__1(lean_object* v_inst_23_, lean_object* v_x1_24_, lean_object* v_x2_25_){
_start:
{
lean_object* v___x_26_; 
v___x_26_ = lp_mathlib_Multiset_ndinter___redArg(v_inst_23_, v_x1_24_, v_x2_25_);
return v___x_26_;
}
}
static lean_object* _init_lp_mathlib_Finset_instLattice___redArg___closed__0(void){
_start:
{
lean_object* v___x_27_; 
v___x_27_ = lp_mathlib_Finset_instPartialOrder(lean_box(0));
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instLattice___redArg(lean_object* v_inst_28_){
_start:
{
lean_object* v___f_29_; lean_object* v___f_30_; lean_object* v___x_31_; lean_object* v___x_32_; lean_object* v___x_33_; 
lean_inc_ref(v_inst_28_);
v___f_29_ = lean_alloc_closure((void*)(lp_mathlib_Finset_instLattice___redArg___lam__0), 3, 1);
lean_closure_set(v___f_29_, 0, v_inst_28_);
v___f_30_ = lean_alloc_closure((void*)(lp_mathlib_Finset_instLattice___redArg___lam__1), 3, 1);
lean_closure_set(v___f_30_, 0, v_inst_28_);
v___x_31_ = lean_obj_once(&lp_mathlib_Finset_instLattice___redArg___closed__0, &lp_mathlib_Finset_instLattice___redArg___closed__0_once, _init_lp_mathlib_Finset_instLattice___redArg___closed__0);
v___x_32_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_32_, 0, v___x_31_);
lean_ctor_set(v___x_32_, 1, v___f_29_);
v___x_33_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_33_, 0, v___x_32_);
lean_ctor_set(v___x_33_, 1, v___f_30_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instLattice(lean_object* v_00_u03b1_34_, lean_object* v_inst_35_){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lp_mathlib_Finset_instLattice___redArg(v_inst_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instDistribLattice___redArg(lean_object* v_inst_37_){
_start:
{
lean_object* v___x_38_; 
v___x_38_ = lp_mathlib_Finset_instLattice___redArg(v_inst_37_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instDistribLattice(lean_object* v_00_u03b1_39_, lean_object* v_inst_40_){
_start:
{
lean_object* v___x_41_; 
v___x_41_ = lp_mathlib_Finset_instLattice___redArg(v_inst_40_);
return v___x_41_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_FinsetOps(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Lattice_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_FinsetOps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Finset_Lattice_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Finset_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_FinsetOps(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_Lattice_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_FinsetOps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Lattice_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Finset_Lattice_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Finset_Lattice_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
