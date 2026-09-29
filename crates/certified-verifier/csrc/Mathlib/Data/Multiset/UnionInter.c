// Lean compiler output
// Module: Mathlib.Data.Multiset.UnionInter
// Imports: public import Init public meta import Init public import Mathlib.Data.List.Perm.Lattice public import Mathlib.Data.Multiset.Filter public import Mathlib.Order.MinMax public import Mathlib.Logic.Pairwise
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
lean_object* lp_mathlib_Multiset_instPartialOrder(lean_object*);
lean_object* lp_mathlib_Multiset_sub___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* l_instBEqOfDecidableEq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_List_bagInter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_union___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_union(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instUnion___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instUnion(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_inter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_inter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instInter___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instInter(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instLattice___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instLattice___redArg___lam__1(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Multiset_instLattice___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Multiset_instLattice___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instDistribLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instDistribLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_union___redArg(lean_object* v_inst_1_, lean_object* v_s_2_, lean_object* v_t_3_){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; 
lean_inc(v_t_3_);
v___x_4_ = lp_mathlib_Multiset_sub___redArg(v_inst_1_, v_s_2_, v_t_3_);
v___x_5_ = l_List_appendTR___redArg(v___x_4_, v_t_3_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_union(lean_object* v_00_u03b1_6_, lean_object* v_inst_7_, lean_object* v_s_8_, lean_object* v_t_9_){
_start:
{
lean_object* v___x_10_; 
v___x_10_ = lp_mathlib_Multiset_union___redArg(v_inst_7_, v_s_8_, v_t_9_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instUnion___redArg(lean_object* v_inst_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_union), 4, 2);
lean_closure_set(v___x_12_, 0, lean_box(0));
lean_closure_set(v___x_12_, 1, v_inst_11_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instUnion(lean_object* v_00_u03b1_13_, lean_object* v_inst_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_union), 4, 2);
lean_closure_set(v___x_15_, 0, lean_box(0));
lean_closure_set(v___x_15_, 1, v_inst_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_inter___redArg(lean_object* v_inst_16_, lean_object* v_s_17_, lean_object* v_t_18_){
_start:
{
lean_object* v___f_19_; lean_object* v___x_20_; 
v___f_19_ = lean_alloc_closure((void*)(l_instBEqOfDecidableEq___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_19_, 0, v_inst_16_);
v___x_20_ = lp_batteries_List_bagInter___redArg(v___f_19_, v_s_17_, v_t_18_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_inter(lean_object* v_00_u03b1_21_, lean_object* v_inst_22_, lean_object* v_s_23_, lean_object* v_t_24_){
_start:
{
lean_object* v___x_25_; 
v___x_25_ = lp_mathlib_Multiset_inter___redArg(v_inst_22_, v_s_23_, v_t_24_);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instInter___redArg(lean_object* v_inst_26_){
_start:
{
lean_object* v___x_27_; 
v___x_27_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_inter), 4, 2);
lean_closure_set(v___x_27_, 0, lean_box(0));
lean_closure_set(v___x_27_, 1, v_inst_26_);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instInter(lean_object* v_00_u03b1_28_, lean_object* v_inst_29_){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_inter), 4, 2);
lean_closure_set(v___x_30_, 0, lean_box(0));
lean_closure_set(v___x_30_, 1, v_inst_29_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instLattice___redArg___lam__0(lean_object* v_inst_31_, lean_object* v_x1_32_, lean_object* v_x2_33_){
_start:
{
lean_object* v___x_34_; 
v___x_34_ = lp_mathlib_Multiset_union___redArg(v_inst_31_, v_x1_32_, v_x2_33_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instLattice___redArg___lam__1(lean_object* v_inst_35_, lean_object* v_x1_36_, lean_object* v_x2_37_){
_start:
{
lean_object* v___x_38_; 
v___x_38_ = lp_mathlib_Multiset_inter___redArg(v_inst_35_, v_x1_36_, v_x2_37_);
return v___x_38_;
}
}
static lean_object* _init_lp_mathlib_Multiset_instLattice___redArg___closed__0(void){
_start:
{
lean_object* v___x_39_; 
v___x_39_ = lp_mathlib_Multiset_instPartialOrder(lean_box(0));
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instLattice___redArg(lean_object* v_inst_40_){
_start:
{
lean_object* v___f_41_; lean_object* v___f_42_; lean_object* v___x_43_; lean_object* v___x_44_; lean_object* v___x_45_; 
lean_inc_ref(v_inst_40_);
v___f_41_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_instLattice___redArg___lam__0), 3, 1);
lean_closure_set(v___f_41_, 0, v_inst_40_);
v___f_42_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_instLattice___redArg___lam__1), 3, 1);
lean_closure_set(v___f_42_, 0, v_inst_40_);
v___x_43_ = lean_obj_once(&lp_mathlib_Multiset_instLattice___redArg___closed__0, &lp_mathlib_Multiset_instLattice___redArg___closed__0_once, _init_lp_mathlib_Multiset_instLattice___redArg___closed__0);
v___x_44_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_44_, 0, v___x_43_);
lean_ctor_set(v___x_44_, 1, v___f_41_);
v___x_45_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_45_, 0, v___x_44_);
lean_ctor_set(v___x_45_, 1, v___f_42_);
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instLattice(lean_object* v_00_u03b1_46_, lean_object* v_inst_47_){
_start:
{
lean_object* v___x_48_; 
v___x_48_ = lp_mathlib_Multiset_instLattice___redArg(v_inst_47_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instDistribLattice___redArg(lean_object* v_inst_49_){
_start:
{
lean_object* v___x_50_; 
v___x_50_ = lp_mathlib_Multiset_instLattice___redArg(v_inst_49_);
return v___x_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instDistribLattice(lean_object* v_00_u03b1_51_, lean_object* v_inst_52_){
_start:
{
lean_object* v___x_53_; 
v___x_53_ = lp_mathlib_Multiset_instLattice___redArg(v_inst_52_);
return v___x_53_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Perm_Lattice(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Filter(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_MinMax(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Pairwise(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_UnionInter(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Perm_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Filter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_MinMax(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Pairwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Multiset_UnionInter(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_List_Perm_Lattice(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Filter(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_MinMax(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Pairwise(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Multiset_UnionInter(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Perm_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Filter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_MinMax(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Pairwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_UnionInter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Multiset_UnionInter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Multiset_UnionInter(builtin);
}
#ifdef __cplusplus
}
#endif
