// Lean compiler output
// Module: Mathlib.Order.Filter.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Pi.Basic public import Mathlib.Data.Set.Lattice.Bounded public import Mathlib.Order.Filter.Defs public import Mathlib.Tactic.ToFun
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
lean_object* lp_mathlib_Filter_instInf___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Filter_instHNot___lam__0(lean_object*);
lean_object* lp_mathlib_Filter_sInf(lean_object*, lean_object*);
lean_object* lp_mathlib_Filter_instSupSet___lam__0(lean_object*);
lean_object* lp_mathlib_Filter_instPartialOrder(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_inhabitedMem(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_instTransSetMem(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_instTransSetMem__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_generate(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_mkOfClosure(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_giGenerate___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Filter_giGenerate___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Filter_giGenerate___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Filter_giGenerate___closed__0 = (const lean_object*)&lp_mathlib_Filter_giGenerate___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Filter_giGenerate(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_instCompleteLatticeFilter___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Filter_instCompleteLatticeFilter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Filter_instCompleteLatticeFilter___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Filter_instCompleteLatticeFilter___closed__0 = (const lean_object*)&lp_mathlib_Filter_instCompleteLatticeFilter___closed__0_value;
static lean_once_cell_t lp_mathlib_Filter_instCompleteLatticeFilter___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Filter_instCompleteLatticeFilter___closed__1;
static const lean_closure_object lp_mathlib_Filter_instCompleteLatticeFilter___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Filter_instSupSet___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Filter_instCompleteLatticeFilter___closed__2 = (const lean_object*)&lp_mathlib_Filter_instCompleteLatticeFilter___closed__2_value;
static const lean_closure_object lp_mathlib_Filter_instCompleteLatticeFilter___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Filter_sInf, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Filter_instCompleteLatticeFilter___closed__3 = (const lean_object*)&lp_mathlib_Filter_instCompleteLatticeFilter___closed__3_value;
static lean_once_cell_t lp_mathlib_Filter_instCompleteLatticeFilter___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Filter_instCompleteLatticeFilter___closed__4;
static lean_once_cell_t lp_mathlib_Filter_instCompleteLatticeFilter___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Filter_instCompleteLatticeFilter___closed__5;
static const lean_ctor_object lp_mathlib_Filter_instCompleteLatticeFilter___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Filter_instCompleteLatticeFilter___closed__6 = (const lean_object*)&lp_mathlib_Filter_instCompleteLatticeFilter___closed__6_value;
static lean_once_cell_t lp_mathlib_Filter_instCompleteLatticeFilter___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Filter_instCompleteLatticeFilter___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Filter_instCompleteLatticeFilter(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_instInhabited(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_unique(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Filter_instCoframe___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Filter_instCoframe___closed__0;
static const lean_closure_object lp_mathlib_Filter_instCoframe___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Filter_instInf___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Filter_instCoframe___closed__1 = (const lean_object*)&lp_mathlib_Filter_instCoframe___closed__1_value;
static const lean_closure_object lp_mathlib_Filter_instCoframe___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Filter_instHNot___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Filter_instCoframe___closed__2 = (const lean_object*)&lp_mathlib_Filter_instCoframe___closed__2_value;
static lean_once_cell_t lp_mathlib_Filter_instCoframe___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Filter_instCoframe___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Filter_instCoframe(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_instTransForallEventuallyEq(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_instTransForallEventuallyLE(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_instTransForallEventuallyLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_instTransForallEventuallyEqEventuallyLE(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_instTransForallEventuallyEqEventuallyLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_instTransForallEventuallyLEEventuallyEq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_instTransForallEventuallyLEEventuallyEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_EventuallySubset_instTransSet(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_inhabitedMem(lean_object* v_00_u03b1_1_, lean_object* v_f_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_box(0);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instTransSetMem(lean_object* v_00_u03b1_4_){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lean_box(0);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instTransSetMem__1(lean_object* v_00_u03b1_6_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lean_box(0);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_generate(lean_object* v_00_u03b1_8_, lean_object* v_g_9_){
_start:
{
lean_object* v___x_10_; 
v___x_10_ = lean_box(0);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_mkOfClosure(lean_object* v_00_u03b1_11_, lean_object* v_s_12_, lean_object* v_hs_13_){
_start:
{
lean_object* v___x_14_; 
v___x_14_ = lean_box(0);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_giGenerate___lam__0(lean_object* v_s_15_, lean_object* v_hs_16_){
_start:
{
lean_object* v___x_17_; 
v___x_17_ = lean_box(0);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_giGenerate(lean_object* v_00_u03b1_19_){
_start:
{
lean_object* v___f_20_; 
v___f_20_ = ((lean_object*)(lp_mathlib_Filter_giGenerate___closed__0));
return v___f_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instCompleteLatticeFilter___lam__0(lean_object* v_a_21_, lean_object* v_b_22_){
_start:
{
lean_object* v___x_23_; 
v___x_23_ = lean_box(0);
return v___x_23_;
}
}
static lean_object* _init_lp_mathlib_Filter_instCompleteLatticeFilter___closed__1(void){
_start:
{
lean_object* v___x_25_; 
v___x_25_ = lp_mathlib_Filter_instPartialOrder(lean_box(0));
return v___x_25_;
}
}
static lean_object* _init_lp_mathlib_Filter_instCompleteLatticeFilter___closed__4(void){
_start:
{
lean_object* v___f_28_; lean_object* v___x_29_; lean_object* v___x_30_; 
v___f_28_ = ((lean_object*)(lp_mathlib_Filter_instCompleteLatticeFilter___closed__0));
v___x_29_ = lean_obj_once(&lp_mathlib_Filter_instCompleteLatticeFilter___closed__1, &lp_mathlib_Filter_instCompleteLatticeFilter___closed__1_once, _init_lp_mathlib_Filter_instCompleteLatticeFilter___closed__1);
v___x_30_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_30_, 0, v___x_29_);
lean_ctor_set(v___x_30_, 1, v___f_28_);
return v___x_30_;
}
}
static lean_object* _init_lp_mathlib_Filter_instCompleteLatticeFilter___closed__5(void){
_start:
{
lean_object* v___f_31_; lean_object* v___x_32_; lean_object* v___x_33_; 
v___f_31_ = ((lean_object*)(lp_mathlib_Filter_instCompleteLatticeFilter___closed__0));
v___x_32_ = lean_obj_once(&lp_mathlib_Filter_instCompleteLatticeFilter___closed__4, &lp_mathlib_Filter_instCompleteLatticeFilter___closed__4_once, _init_lp_mathlib_Filter_instCompleteLatticeFilter___closed__4);
v___x_33_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_33_, 0, v___x_32_);
lean_ctor_set(v___x_33_, 1, v___f_31_);
return v___x_33_;
}
}
static lean_object* _init_lp_mathlib_Filter_instCompleteLatticeFilter___closed__7(void){
_start:
{
lean_object* v___x_36_; lean_object* v___x_37_; lean_object* v___f_38_; lean_object* v___x_39_; lean_object* v___x_40_; 
v___x_36_ = ((lean_object*)(lp_mathlib_Filter_instCompleteLatticeFilter___closed__6));
v___x_37_ = ((lean_object*)(lp_mathlib_Filter_instCompleteLatticeFilter___closed__3));
v___f_38_ = ((lean_object*)(lp_mathlib_Filter_instCompleteLatticeFilter___closed__2));
v___x_39_ = lean_obj_once(&lp_mathlib_Filter_instCompleteLatticeFilter___closed__5, &lp_mathlib_Filter_instCompleteLatticeFilter___closed__5_once, _init_lp_mathlib_Filter_instCompleteLatticeFilter___closed__5);
v___x_40_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_40_, 0, v___x_39_);
lean_ctor_set(v___x_40_, 1, v___f_38_);
lean_ctor_set(v___x_40_, 2, v___x_37_);
lean_ctor_set(v___x_40_, 3, v___x_36_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instCompleteLatticeFilter(lean_object* v_00_u03b1_41_){
_start:
{
lean_object* v___x_42_; 
v___x_42_ = lean_obj_once(&lp_mathlib_Filter_instCompleteLatticeFilter___closed__7, &lp_mathlib_Filter_instCompleteLatticeFilter___closed__7_once, _init_lp_mathlib_Filter_instCompleteLatticeFilter___closed__7);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instInhabited(lean_object* v_00_u03b1_43_){
_start:
{
lean_object* v___x_44_; 
v___x_44_ = lean_box(0);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_unique(lean_object* v_00_u03b1_45_, lean_object* v_inst_46_){
_start:
{
lean_object* v___x_47_; 
v___x_47_ = lean_box(0);
return v___x_47_;
}
}
static lean_object* _init_lp_mathlib_Filter_instCoframe___closed__0(void){
_start:
{
lean_object* v___x_48_; 
v___x_48_ = lp_mathlib_Filter_instCompleteLatticeFilter(lean_box(0));
return v___x_48_;
}
}
static lean_object* _init_lp_mathlib_Filter_instCoframe___closed__3(void){
_start:
{
lean_object* v___f_51_; lean_object* v___f_52_; lean_object* v___x_53_; lean_object* v___x_54_; 
v___f_51_ = ((lean_object*)(lp_mathlib_Filter_instCoframe___closed__2));
v___f_52_ = ((lean_object*)(lp_mathlib_Filter_instCoframe___closed__1));
v___x_53_ = lean_obj_once(&lp_mathlib_Filter_instCoframe___closed__0, &lp_mathlib_Filter_instCoframe___closed__0_once, _init_lp_mathlib_Filter_instCoframe___closed__0);
v___x_54_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_54_, 0, v___x_53_);
lean_ctor_set(v___x_54_, 1, v___f_52_);
lean_ctor_set(v___x_54_, 2, v___f_51_);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instCoframe(lean_object* v_00_u03b1_55_){
_start:
{
lean_object* v___x_56_; 
v___x_56_ = lean_obj_once(&lp_mathlib_Filter_instCoframe___closed__3, &lp_mathlib_Filter_instCoframe___closed__3_once, _init_lp_mathlib_Filter_instCoframe___closed__3);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instTransForallEventuallyEq(lean_object* v_00_u03b1_57_, lean_object* v_00_u03b2_58_, lean_object* v_l_59_){
_start:
{
lean_object* v___x_60_; 
v___x_60_ = lean_box(0);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instTransForallEventuallyLE(lean_object* v_00_u03b1_61_, lean_object* v_00_u03b2_62_, lean_object* v_inst_63_, lean_object* v_l_64_){
_start:
{
lean_object* v___x_65_; 
v___x_65_ = lean_box(0);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instTransForallEventuallyLE___boxed(lean_object* v_00_u03b1_66_, lean_object* v_00_u03b2_67_, lean_object* v_inst_68_, lean_object* v_l_69_){
_start:
{
lean_object* v_res_70_; 
v_res_70_ = lp_mathlib_Filter_instTransForallEventuallyLE(v_00_u03b1_66_, v_00_u03b2_67_, v_inst_68_, v_l_69_);
lean_dec_ref(v_inst_68_);
return v_res_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instTransForallEventuallyEqEventuallyLE(lean_object* v_00_u03b1_71_, lean_object* v_00_u03b2_72_, lean_object* v_inst_73_, lean_object* v_l_74_){
_start:
{
lean_object* v___x_75_; 
v___x_75_ = lean_box(0);
return v___x_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instTransForallEventuallyEqEventuallyLE___boxed(lean_object* v_00_u03b1_76_, lean_object* v_00_u03b2_77_, lean_object* v_inst_78_, lean_object* v_l_79_){
_start:
{
lean_object* v_res_80_; 
v_res_80_ = lp_mathlib_Filter_instTransForallEventuallyEqEventuallyLE(v_00_u03b1_76_, v_00_u03b2_77_, v_inst_78_, v_l_79_);
lean_dec_ref(v_inst_78_);
return v_res_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instTransForallEventuallyLEEventuallyEq(lean_object* v_00_u03b1_81_, lean_object* v_00_u03b2_82_, lean_object* v_inst_83_, lean_object* v_l_84_){
_start:
{
lean_object* v___x_85_; 
v___x_85_ = lean_box(0);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instTransForallEventuallyLEEventuallyEq___boxed(lean_object* v_00_u03b1_86_, lean_object* v_00_u03b2_87_, lean_object* v_inst_88_, lean_object* v_l_89_){
_start:
{
lean_object* v_res_90_; 
v_res_90_ = lp_mathlib_Filter_instTransForallEventuallyLEEventuallyEq(v_00_u03b1_86_, v_00_u03b2_87_, v_inst_88_, v_l_89_);
lean_dec_ref(v_inst_88_);
return v_res_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_EventuallySubset_instTransSet(lean_object* v_00_u03b1_91_, lean_object* v_l_92_){
_start:
{
lean_object* v___x_93_; 
v___x_93_ = lean_box(0);
return v___x_93_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Lattice_Bounded(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Filter_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ToFun(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Filter_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Lattice_Bounded(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Filter_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ToFun(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Filter_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Lattice_Bounded(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Filter_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ToFun(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Filter_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Lattice_Bounded(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Filter_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ToFun(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Filter_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Filter_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Filter_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
