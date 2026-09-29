// Lean compiler output
// Module: Mathlib.Algebra.Ring.Aut
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.End public import Mathlib.Algebra.Ring.Equiv
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
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_npowBinRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_RingEquiv_toAddEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_npowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_RingEquiv_toMulEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_RingEquiv_symm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DivInvMonoid_div_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_zpowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingAut_instGroup___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_RingAut_instGroup___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RingAut_instGroup___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RingAut_instGroup___redArg___closed__0 = (const lean_object*)&lp_mathlib_RingAut_instGroup___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_RingAut_instGroup___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_RingAut_instGroup___redArg___closed__1;
static lean_once_cell_t lp_mathlib_RingAut_instGroup___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_RingAut_instGroup___redArg___closed__2;
static lean_once_cell_t lp_mathlib_RingAut_instGroup___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_RingAut_instGroup___redArg___closed__3;
static lean_once_cell_t lp_mathlib_RingAut_instGroup___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_RingAut_instGroup___redArg___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_RingAut_instGroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingAut_instGroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingAut_instInhabited(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingAut_instInhabited___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingAut_toAddAut___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingAut_toAddAut(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingAut_toMulAut___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingAut_toMulAut(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingAut_toPerm___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingAut_toPerm___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_RingAut_toPerm___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RingAut_toPerm___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RingAut_toPerm___closed__0 = (const lean_object*)&lp_mathlib_RingAut_toPerm___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_RingAut_toPerm(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingAut_toPerm___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingAut_instGroup___redArg___lam__0(lean_object* v_g_1_, lean_object* v_h_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lp_mathlib_Equiv_trans___redArg(v_h_2_, v_g_1_);
return v___x_3_;
}
}
static lean_object* _init_lp_mathlib_RingAut_instGroup___redArg___closed__1(void){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_5_;
}
}
static lean_object* _init_lp_mathlib_RingAut_instGroup___redArg___closed__2(void){
_start:
{
lean_object* v___x_6_; lean_object* v___f_7_; lean_object* v___x_8_; 
v___x_6_ = lean_obj_once(&lp_mathlib_RingAut_instGroup___redArg___closed__1, &lp_mathlib_RingAut_instGroup___redArg___closed__1_once, _init_lp_mathlib_RingAut_instGroup___redArg___closed__1);
v___f_7_ = ((lean_object*)(lp_mathlib_RingAut_instGroup___redArg___closed__0));
v___x_8_ = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(v___x_8_, 0, lean_box(0));
lean_closure_set(v___x_8_, 1, v___f_7_);
lean_closure_set(v___x_8_, 2, v___x_6_);
return v___x_8_;
}
}
static lean_object* _init_lp_mathlib_RingAut_instGroup___redArg___closed__3(void){
_start:
{
lean_object* v___x_9_; lean_object* v___f_10_; lean_object* v___x_11_; lean_object* v___x_12_; 
v___x_9_ = lean_obj_once(&lp_mathlib_RingAut_instGroup___redArg___closed__2, &lp_mathlib_RingAut_instGroup___redArg___closed__2_once, _init_lp_mathlib_RingAut_instGroup___redArg___closed__2);
v___f_10_ = ((lean_object*)(lp_mathlib_RingAut_instGroup___redArg___closed__0));
v___x_11_ = lean_obj_once(&lp_mathlib_RingAut_instGroup___redArg___closed__1, &lp_mathlib_RingAut_instGroup___redArg___closed__1_once, _init_lp_mathlib_RingAut_instGroup___redArg___closed__1);
v___x_12_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_12_, 0, v___x_11_);
lean_ctor_set(v___x_12_, 1, v___f_10_);
lean_ctor_set(v___x_12_, 2, v___x_9_);
return v___x_12_;
}
}
static lean_object* _init_lp_mathlib_RingAut_instGroup___redArg___closed__4(void){
_start:
{
lean_object* v___f_13_; lean_object* v___x_14_; lean_object* v___x_15_; 
v___f_13_ = ((lean_object*)(lp_mathlib_RingAut_instGroup___redArg___closed__0));
v___x_14_ = lean_obj_once(&lp_mathlib_RingAut_instGroup___redArg___closed__1, &lp_mathlib_RingAut_instGroup___redArg___closed__1_once, _init_lp_mathlib_RingAut_instGroup___redArg___closed__1);
v___x_15_ = lean_alloc_closure((void*)(l_npowRec___boxed), 5, 3);
lean_closure_set(v___x_15_, 0, lean_box(0));
lean_closure_set(v___x_15_, 1, v___x_14_);
lean_closure_set(v___x_15_, 2, v___f_13_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingAut_instGroup___redArg(lean_object* v_inst_16_, lean_object* v_inst_17_){
_start:
{
lean_object* v___f_18_; lean_object* v___x_19_; lean_object* v___x_20_; lean_object* v___x_21_; lean_object* v___x_22_; lean_object* v___x_23_; lean_object* v___x_24_; lean_object* v___x_25_; 
v___f_18_ = ((lean_object*)(lp_mathlib_RingAut_instGroup___redArg___closed__0));
v___x_19_ = lean_obj_once(&lp_mathlib_RingAut_instGroup___redArg___closed__1, &lp_mathlib_RingAut_instGroup___redArg___closed__1_once, _init_lp_mathlib_RingAut_instGroup___redArg___closed__1);
v___x_20_ = lean_obj_once(&lp_mathlib_RingAut_instGroup___redArg___closed__3, &lp_mathlib_RingAut_instGroup___redArg___closed__3_once, _init_lp_mathlib_RingAut_instGroup___redArg___closed__3);
lean_inc(v_inst_17_);
lean_inc(v_inst_16_);
v___x_21_ = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_symm___boxed), 7, 6);
lean_closure_set(v___x_21_, 0, lean_box(0));
lean_closure_set(v___x_21_, 1, lean_box(0));
lean_closure_set(v___x_21_, 2, v_inst_16_);
lean_closure_set(v___x_21_, 3, v_inst_16_);
lean_closure_set(v___x_21_, 4, v_inst_17_);
lean_closure_set(v___x_21_, 5, v_inst_17_);
lean_inc_ref_n(v___x_21_, 2);
v___x_22_ = lean_alloc_closure((void*)(lp_mathlib_DivInvMonoid_div_x27___boxed), 5, 3);
lean_closure_set(v___x_22_, 0, lean_box(0));
lean_closure_set(v___x_22_, 1, v___x_20_);
lean_closure_set(v___x_22_, 2, v___x_21_);
v___x_23_ = lean_obj_once(&lp_mathlib_RingAut_instGroup___redArg___closed__4, &lp_mathlib_RingAut_instGroup___redArg___closed__4_once, _init_lp_mathlib_RingAut_instGroup___redArg___closed__4);
v___x_24_ = lean_alloc_closure((void*)(lp_mathlib_zpowRec___boxed), 7, 5);
lean_closure_set(v___x_24_, 0, lean_box(0));
lean_closure_set(v___x_24_, 1, v___x_19_);
lean_closure_set(v___x_24_, 2, v___f_18_);
lean_closure_set(v___x_24_, 3, v___x_21_);
lean_closure_set(v___x_24_, 4, v___x_23_);
v___x_25_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_25_, 0, v___x_20_);
lean_ctor_set(v___x_25_, 1, v___x_21_);
lean_ctor_set(v___x_25_, 2, v___x_22_);
lean_ctor_set(v___x_25_, 3, v___x_24_);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingAut_instGroup(lean_object* v_R_26_, lean_object* v_inst_27_, lean_object* v_inst_28_){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lp_mathlib_RingAut_instGroup___redArg(v_inst_27_, v_inst_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingAut_instInhabited(lean_object* v_R_30_, lean_object* v_inst_31_, lean_object* v_inst_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = lean_obj_once(&lp_mathlib_RingAut_instGroup___redArg___closed__1, &lp_mathlib_RingAut_instGroup___redArg___closed__1_once, _init_lp_mathlib_RingAut_instGroup___redArg___closed__1);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingAut_instInhabited___boxed(lean_object* v_R_34_, lean_object* v_inst_35_, lean_object* v_inst_36_){
_start:
{
lean_object* v_res_37_; 
v_res_37_ = lp_mathlib_RingAut_instInhabited(v_R_34_, v_inst_35_, v_inst_36_);
lean_dec(v_inst_36_);
lean_dec(v_inst_35_);
return v_res_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingAut_toAddAut___redArg(lean_object* v_inst_38_, lean_object* v_inst_39_){
_start:
{
lean_object* v___x_40_; 
lean_inc(v_inst_39_);
lean_inc(v_inst_38_);
v___x_40_ = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_toAddEquiv___boxed), 7, 6);
lean_closure_set(v___x_40_, 0, lean_box(0));
lean_closure_set(v___x_40_, 1, lean_box(0));
lean_closure_set(v___x_40_, 2, v_inst_38_);
lean_closure_set(v___x_40_, 3, v_inst_38_);
lean_closure_set(v___x_40_, 4, v_inst_39_);
lean_closure_set(v___x_40_, 5, v_inst_39_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingAut_toAddAut(lean_object* v_R_41_, lean_object* v_inst_42_, lean_object* v_inst_43_){
_start:
{
lean_object* v___x_44_; 
lean_inc(v_inst_43_);
lean_inc(v_inst_42_);
v___x_44_ = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_toAddEquiv___boxed), 7, 6);
lean_closure_set(v___x_44_, 0, lean_box(0));
lean_closure_set(v___x_44_, 1, lean_box(0));
lean_closure_set(v___x_44_, 2, v_inst_42_);
lean_closure_set(v___x_44_, 3, v_inst_42_);
lean_closure_set(v___x_44_, 4, v_inst_43_);
lean_closure_set(v___x_44_, 5, v_inst_43_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingAut_toMulAut___redArg(lean_object* v_inst_45_, lean_object* v_inst_46_){
_start:
{
lean_object* v___x_47_; 
lean_inc(v_inst_46_);
lean_inc(v_inst_45_);
v___x_47_ = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_toMulEquiv___boxed), 7, 6);
lean_closure_set(v___x_47_, 0, lean_box(0));
lean_closure_set(v___x_47_, 1, lean_box(0));
lean_closure_set(v___x_47_, 2, v_inst_45_);
lean_closure_set(v___x_47_, 3, v_inst_45_);
lean_closure_set(v___x_47_, 4, v_inst_46_);
lean_closure_set(v___x_47_, 5, v_inst_46_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingAut_toMulAut(lean_object* v_R_48_, lean_object* v_inst_49_, lean_object* v_inst_50_){
_start:
{
lean_object* v___x_51_; 
lean_inc(v_inst_50_);
lean_inc(v_inst_49_);
v___x_51_ = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_toMulEquiv___boxed), 7, 6);
lean_closure_set(v___x_51_, 0, lean_box(0));
lean_closure_set(v___x_51_, 1, lean_box(0));
lean_closure_set(v___x_51_, 2, v_inst_49_);
lean_closure_set(v___x_51_, 3, v_inst_49_);
lean_closure_set(v___x_51_, 4, v_inst_50_);
lean_closure_set(v___x_51_, 5, v_inst_50_);
return v___x_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingAut_toPerm___lam__0(lean_object* v_self_52_){
_start:
{
lean_inc_ref(v_self_52_);
return v_self_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingAut_toPerm___lam__0___boxed(lean_object* v_self_53_){
_start:
{
lean_object* v_res_54_; 
v_res_54_ = lp_mathlib_RingAut_toPerm___lam__0(v_self_53_);
lean_dec_ref(v_self_53_);
return v_res_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingAut_toPerm(lean_object* v_R_56_, lean_object* v_inst_57_, lean_object* v_inst_58_){
_start:
{
lean_object* v___f_59_; 
v___f_59_ = ((lean_object*)(lp_mathlib_RingAut_toPerm___closed__0));
return v___f_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingAut_toPerm___boxed(lean_object* v_R_60_, lean_object* v_inst_61_, lean_object* v_inst_62_){
_start:
{
lean_object* v_res_63_; 
v_res_63_ = lp_mathlib_RingAut_toPerm(v_R_60_, v_inst_61_, v_inst_62_);
lean_dec(v_inst_62_);
lean_dec(v_inst_61_);
return v_res_63_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_End(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Equiv(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Aut(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Ring_Aut(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_End(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Equiv(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Aut(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Aut(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Ring_Aut(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Ring_Aut(builtin);
}
#ifdef __cplusplus
}
#endif
