// Lean compiler output
// Module: Mathlib.Algebra.Order.Ring.Unbundled.Rat
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Order.Group.Unbundled.Abs public import Mathlib.Algebra.Order.Group.Unbundled.Basic public import Mathlib.Algebra.Order.Group.Unbundled.Int public import Mathlib.Data.Rat.Defs public import Mathlib.Algebra.Ring.Int.Defs
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
lean_object* l_Rat_instDecidableLe___boxed(lean_object*, lean_object*);
lean_object* l_Rat_instDecidableLt___boxed(lean_object*, lean_object*);
lean_object* l_instDecidableEqRat___boxed(lean_object*, lean_object*);
uint8_t l_Rat_blt(lean_object*, lean_object*);
uint8_t l_instDecidableEqRat_decEq(lean_object*, lean_object*);
lean_object* l_Rat_instMax___lam__0(lean_object*, lean_object*);
lean_object* l_Rat_instMin___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_LinearOrder_toLattice___redArg(lean_object*);
lean_object* lp_mathlib_Lattice_toSemilatticeInf___redArg(lean_object*);
lean_object* l_Rat_ofScientific(lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRatCast_toOfScientific___redArg___lam__0(lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRatCast_toOfScientific___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRatCast_toOfScientific___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRatCast_toOfScientific(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Rat_linearOrder___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_linearOrder___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Rat_linearOrder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Rat_linearOrder___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Rat_linearOrder___closed__0 = (const lean_object*)&lp_mathlib_Rat_linearOrder___closed__0_value;
static const lean_closure_object lp_mathlib_Rat_linearOrder___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Rat_instMin___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Rat_linearOrder___closed__1 = (const lean_object*)&lp_mathlib_Rat_linearOrder___closed__1_value;
static const lean_closure_object lp_mathlib_Rat_linearOrder___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Rat_instMax___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Rat_linearOrder___closed__2 = (const lean_object*)&lp_mathlib_Rat_linearOrder___closed__2_value;
static const lean_ctor_object lp_mathlib_Rat_linearOrder___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Rat_linearOrder___closed__3 = (const lean_object*)&lp_mathlib_Rat_linearOrder___closed__3_value;
static const lean_closure_object lp_mathlib_Rat_linearOrder___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Rat_instDecidableLe___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Rat_linearOrder___closed__4 = (const lean_object*)&lp_mathlib_Rat_linearOrder___closed__4_value;
static const lean_closure_object lp_mathlib_Rat_linearOrder___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Rat_instDecidableLt___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Rat_linearOrder___closed__5 = (const lean_object*)&lp_mathlib_Rat_linearOrder___closed__5_value;
static lean_once_cell_t lp_mathlib_Rat_linearOrder___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Rat_linearOrder___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_Rat_linearOrder;
static lean_once_cell_t lp_mathlib_Rat_instDistribLattice___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Rat_instDistribLattice___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Rat_instDistribLattice;
LEAN_EXPORT lean_object* lp_mathlib_Rat_instLattice;
static lean_once_cell_t lp_mathlib_Rat_instSemilatticeInf___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Rat_instSemilatticeInf___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Rat_instSemilatticeInf;
LEAN_EXPORT lean_object* lp_mathlib_Rat_instSemilatticeSup;
LEAN_EXPORT const lean_object* lp_mathlib_Rat_instInf = (const lean_object*)&lp_mathlib_Rat_linearOrder___closed__1_value;
LEAN_EXPORT const lean_object* lp_mathlib_Rat_instSup = (const lean_object*)&lp_mathlib_Rat_linearOrder___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Rat_instPartialOrder;
LEAN_EXPORT lean_object* lp_mathlib_Rat_instPreorder;
LEAN_EXPORT lean_object* lp_mathlib_NNRatCast_toOfScientific___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_m_2_, uint8_t v_b_3_, lean_object* v_d_4_){
_start:
{
lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_5_ = l_Rat_ofScientific(v_m_2_, v_b_3_, v_d_4_);
v___x_6_ = lean_apply_1(v_inst_1_, v___x_5_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRatCast_toOfScientific___redArg___lam__0___boxed(lean_object* v_inst_7_, lean_object* v_m_8_, lean_object* v_b_9_, lean_object* v_d_10_){
_start:
{
uint8_t v_b_boxed_11_; lean_object* v_res_12_; 
v_b_boxed_11_ = lean_unbox(v_b_9_);
v_res_12_ = lp_mathlib_NNRatCast_toOfScientific___redArg___lam__0(v_inst_7_, v_m_8_, v_b_boxed_11_, v_d_10_);
lean_dec(v_d_10_);
return v_res_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRatCast_toOfScientific___redArg(lean_object* v_inst_13_){
_start:
{
lean_object* v___f_14_; 
v___f_14_ = lean_alloc_closure((void*)(lp_mathlib_NNRatCast_toOfScientific___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_14_, 0, v_inst_13_);
return v___f_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRatCast_toOfScientific(lean_object* v_K_15_, lean_object* v_inst_16_){
_start:
{
lean_object* v___f_17_; 
v___f_17_ = lean_alloc_closure((void*)(lp_mathlib_NNRatCast_toOfScientific___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_17_, 0, v_inst_16_);
return v___f_17_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Rat_linearOrder___lam__0(lean_object* v_a_18_, lean_object* v_b_19_){
_start:
{
uint8_t v___x_20_; 
lean_inc_ref(v_b_19_);
lean_inc_ref(v_a_18_);
v___x_20_ = l_Rat_blt(v_a_18_, v_b_19_);
if (v___x_20_ == 0)
{
uint8_t v___x_21_; 
v___x_21_ = l_instDecidableEqRat_decEq(v_a_18_, v_b_19_);
lean_dec_ref(v_b_19_);
lean_dec_ref(v_a_18_);
if (v___x_21_ == 0)
{
uint8_t v___x_22_; 
v___x_22_ = 2;
return v___x_22_;
}
else
{
uint8_t v___x_23_; 
v___x_23_ = 1;
return v___x_23_;
}
}
else
{
uint8_t v___x_24_; 
lean_dec_ref(v_b_19_);
lean_dec_ref(v_a_18_);
v___x_24_ = 0;
return v___x_24_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_linearOrder___lam__0___boxed(lean_object* v_a_25_, lean_object* v_b_26_){
_start:
{
uint8_t v_res_27_; lean_object* v_r_28_; 
v_res_27_ = lp_mathlib_Rat_linearOrder___lam__0(v_a_25_, v_b_26_);
v_r_28_ = lean_box(v_res_27_);
return v_r_28_;
}
}
static lean_object* _init_lp_mathlib_Rat_linearOrder___closed__6(void){
_start:
{
lean_object* v___x_37_; lean_object* v___x_38_; lean_object* v___x_39_; lean_object* v___f_40_; lean_object* v___f_41_; lean_object* v___f_42_; lean_object* v___x_43_; lean_object* v___x_44_; 
v___x_37_ = ((lean_object*)(lp_mathlib_Rat_linearOrder___closed__5));
v___x_38_ = lean_alloc_closure((void*)(l_instDecidableEqRat___boxed), 2, 0);
v___x_39_ = ((lean_object*)(lp_mathlib_Rat_linearOrder___closed__4));
v___f_40_ = ((lean_object*)(lp_mathlib_Rat_linearOrder___closed__0));
v___f_41_ = ((lean_object*)(lp_mathlib_Rat_linearOrder___closed__2));
v___f_42_ = ((lean_object*)(lp_mathlib_Rat_linearOrder___closed__1));
v___x_43_ = ((lean_object*)(lp_mathlib_Rat_linearOrder___closed__3));
v___x_44_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_44_, 0, v___x_43_);
lean_ctor_set(v___x_44_, 1, v___f_42_);
lean_ctor_set(v___x_44_, 2, v___f_41_);
lean_ctor_set(v___x_44_, 3, v___f_40_);
lean_ctor_set(v___x_44_, 4, v___x_39_);
lean_ctor_set(v___x_44_, 5, v___x_38_);
lean_ctor_set(v___x_44_, 6, v___x_37_);
return v___x_44_;
}
}
static lean_object* _init_lp_mathlib_Rat_linearOrder(void){
_start:
{
lean_object* v___x_45_; 
v___x_45_ = lean_obj_once(&lp_mathlib_Rat_linearOrder___closed__6, &lp_mathlib_Rat_linearOrder___closed__6_once, _init_lp_mathlib_Rat_linearOrder___closed__6);
return v___x_45_;
}
}
static lean_object* _init_lp_mathlib_Rat_instDistribLattice___closed__0(void){
_start:
{
lean_object* v___x_46_; lean_object* v___x_47_; 
v___x_46_ = lp_mathlib_Rat_linearOrder;
v___x_47_ = lp_mathlib_LinearOrder_toLattice___redArg(v___x_46_);
return v___x_47_;
}
}
static lean_object* _init_lp_mathlib_Rat_instDistribLattice(void){
_start:
{
lean_object* v___x_48_; 
v___x_48_ = lean_obj_once(&lp_mathlib_Rat_instDistribLattice___closed__0, &lp_mathlib_Rat_instDistribLattice___closed__0_once, _init_lp_mathlib_Rat_instDistribLattice___closed__0);
return v___x_48_;
}
}
static lean_object* _init_lp_mathlib_Rat_instLattice(void){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lp_mathlib_Rat_instDistribLattice;
return v___x_49_;
}
}
static lean_object* _init_lp_mathlib_Rat_instSemilatticeInf___closed__0(void){
_start:
{
lean_object* v___x_50_; lean_object* v___x_51_; 
v___x_50_ = lp_mathlib_Rat_instDistribLattice;
v___x_51_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_50_);
return v___x_51_;
}
}
static lean_object* _init_lp_mathlib_Rat_instSemilatticeInf(void){
_start:
{
lean_object* v___x_52_; 
v___x_52_ = lean_obj_once(&lp_mathlib_Rat_instSemilatticeInf___closed__0, &lp_mathlib_Rat_instSemilatticeInf___closed__0_once, _init_lp_mathlib_Rat_instSemilatticeInf___closed__0);
return v___x_52_;
}
}
static lean_object* _init_lp_mathlib_Rat_instSemilatticeSup(void){
_start:
{
lean_object* v___x_53_; lean_object* v_toSemilatticeSup_54_; 
v___x_53_ = lp_mathlib_Rat_instDistribLattice;
v_toSemilatticeSup_54_ = lean_ctor_get(v___x_53_, 0);
lean_inc_ref(v_toSemilatticeSup_54_);
return v_toSemilatticeSup_54_;
}
}
static lean_object* _init_lp_mathlib_Rat_instPartialOrder(void){
_start:
{
lean_object* v___x_57_; lean_object* v_toPartialOrder_58_; 
v___x_57_ = lp_mathlib_Rat_instSemilatticeInf;
v_toPartialOrder_58_ = lean_ctor_get(v___x_57_, 0);
lean_inc_ref(v_toPartialOrder_58_);
return v_toPartialOrder_58_;
}
}
static lean_object* _init_lp_mathlib_Rat_instPreorder(void){
_start:
{
lean_object* v___x_59_; 
v___x_59_ = lp_mathlib_Rat_instPartialOrder;
return v___x_59_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Abs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Int(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Rat_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Unbundled_Rat(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Abs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Int(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Rat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Rat_linearOrder = _init_lp_mathlib_Rat_linearOrder();
lean_mark_persistent(lp_mathlib_Rat_linearOrder);
lp_mathlib_Rat_instDistribLattice = _init_lp_mathlib_Rat_instDistribLattice();
lean_mark_persistent(lp_mathlib_Rat_instDistribLattice);
lp_mathlib_Rat_instLattice = _init_lp_mathlib_Rat_instLattice();
lean_mark_persistent(lp_mathlib_Rat_instLattice);
lp_mathlib_Rat_instSemilatticeInf = _init_lp_mathlib_Rat_instSemilatticeInf();
lean_mark_persistent(lp_mathlib_Rat_instSemilatticeInf);
lp_mathlib_Rat_instSemilatticeSup = _init_lp_mathlib_Rat_instSemilatticeSup();
lean_mark_persistent(lp_mathlib_Rat_instSemilatticeSup);
lp_mathlib_Rat_instPartialOrder = _init_lp_mathlib_Rat_instPartialOrder();
lean_mark_persistent(lp_mathlib_Rat_instPartialOrder);
lp_mathlib_Rat_instPreorder = _init_lp_mathlib_Rat_instPreorder();
lean_mark_persistent(lp_mathlib_Rat_instPreorder);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Order_Ring_Unbundled_Rat(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Abs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Int(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Rat_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Unbundled_Rat(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Abs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Int(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Rat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Unbundled_Rat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Order_Ring_Unbundled_Rat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Order_Ring_Unbundled_Rat(builtin);
}
#ifdef __cplusplus
}
#endif
