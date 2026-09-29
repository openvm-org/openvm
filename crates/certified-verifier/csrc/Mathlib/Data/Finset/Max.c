// Lean compiler output
// Module: Mathlib.Data.Finset.Max
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.Card public import Mathlib.Data.Finset.Lattice.Fold
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
lean_object* lp_mathlib_LinearOrder_toLattice___redArg(lean_object*);
lean_object* lp_mathlib_Finset_sup_x27___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lattice_toSemilatticeInf___redArg(lean_object*);
lean_object* lp_mathlib_Finset_inf_x27___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_WithTop_some(lean_object*, lean_object*);
lean_object* lp_mathlib_WithBot_semilatticeSup___redArg(lean_object*);
lean_object* lp_mathlib_WithBot_some(lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_sup___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_WithTop_semilatticeInf___redArg(lean_object*);
lean_object* lp_mathlib_Finset_inf___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Finset_max___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithBot_some, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Finset_max___redArg___closed__0 = (const lean_object*)&lp_mathlib_Finset_max___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Finset_max___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_max___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_max(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_max___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Finset_min___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithTop_some, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Finset_min___redArg___closed__0 = (const lean_object*)&lp_mathlib_Finset_min___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Finset_min___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_min___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_min(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_min___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Finset_max_x27___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_id___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Finset_max_x27___redArg___closed__0 = (const lean_object*)&lp_mathlib_Finset_max_x27___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Finset_max_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_max_x27___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_max_x27(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_max_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_min_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_min_x27___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_min_x27(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_min_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_max___redArg(lean_object* v_inst_2_, lean_object* v_s_3_){
_start:
{
lean_object* v___x_4_; lean_object* v_toSemilatticeSup_5_; lean_object* v___x_6_; lean_object* v___x_7_; lean_object* v___x_8_; lean_object* v___x_9_; 
v___x_4_ = lp_mathlib_LinearOrder_toLattice___redArg(v_inst_2_);
v_toSemilatticeSup_5_ = lean_ctor_get(v___x_4_, 0);
lean_inc_ref(v_toSemilatticeSup_5_);
lean_dec_ref(v___x_4_);
v___x_6_ = lp_mathlib_WithBot_semilatticeSup___redArg(v_toSemilatticeSup_5_);
v___x_7_ = lean_box(0);
v___x_8_ = ((lean_object*)(lp_mathlib_Finset_max___redArg___closed__0));
v___x_9_ = lp_mathlib_Finset_sup___redArg(v___x_6_, v___x_7_, v_s_3_, v___x_8_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_max___redArg___boxed(lean_object* v_inst_10_, lean_object* v_s_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_mathlib_Finset_max___redArg(v_inst_10_, v_s_11_);
lean_dec_ref(v_inst_10_);
return v_res_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_max(lean_object* v_00_u03b1_13_, lean_object* v_inst_14_, lean_object* v_s_15_){
_start:
{
lean_object* v___x_16_; 
v___x_16_ = lp_mathlib_Finset_max___redArg(v_inst_14_, v_s_15_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_max___boxed(lean_object* v_00_u03b1_17_, lean_object* v_inst_18_, lean_object* v_s_19_){
_start:
{
lean_object* v_res_20_; 
v_res_20_ = lp_mathlib_Finset_max(v_00_u03b1_17_, v_inst_18_, v_s_19_);
lean_dec_ref(v_inst_18_);
return v_res_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_min___redArg(lean_object* v_inst_22_, lean_object* v_s_23_){
_start:
{
lean_object* v___x_24_; lean_object* v___x_25_; lean_object* v___x_26_; lean_object* v___x_27_; lean_object* v___x_28_; lean_object* v___x_29_; 
v___x_24_ = lp_mathlib_LinearOrder_toLattice___redArg(v_inst_22_);
v___x_25_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_24_);
v___x_26_ = lp_mathlib_WithTop_semilatticeInf___redArg(v___x_25_);
v___x_27_ = lean_box(0);
v___x_28_ = ((lean_object*)(lp_mathlib_Finset_min___redArg___closed__0));
v___x_29_ = lp_mathlib_Finset_inf___redArg(v___x_26_, v___x_27_, v_s_23_, v___x_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_min___redArg___boxed(lean_object* v_inst_30_, lean_object* v_s_31_){
_start:
{
lean_object* v_res_32_; 
v_res_32_ = lp_mathlib_Finset_min___redArg(v_inst_30_, v_s_31_);
lean_dec_ref(v_inst_30_);
return v_res_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_min(lean_object* v_00_u03b1_33_, lean_object* v_inst_34_, lean_object* v_s_35_){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lp_mathlib_Finset_min___redArg(v_inst_34_, v_s_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_min___boxed(lean_object* v_00_u03b1_37_, lean_object* v_inst_38_, lean_object* v_s_39_){
_start:
{
lean_object* v_res_40_; 
v_res_40_ = lp_mathlib_Finset_min(v_00_u03b1_37_, v_inst_38_, v_s_39_);
lean_dec_ref(v_inst_38_);
return v_res_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_max_x27___redArg(lean_object* v_inst_42_, lean_object* v_s_43_){
_start:
{
lean_object* v___x_44_; lean_object* v_toSemilatticeSup_45_; lean_object* v___x_46_; lean_object* v___x_47_; 
v___x_44_ = lp_mathlib_LinearOrder_toLattice___redArg(v_inst_42_);
v_toSemilatticeSup_45_ = lean_ctor_get(v___x_44_, 0);
lean_inc_ref(v_toSemilatticeSup_45_);
lean_dec_ref(v___x_44_);
v___x_46_ = ((lean_object*)(lp_mathlib_Finset_max_x27___redArg___closed__0));
v___x_47_ = lp_mathlib_Finset_sup_x27___redArg(v_toSemilatticeSup_45_, v_s_43_, v___x_46_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_max_x27___redArg___boxed(lean_object* v_inst_48_, lean_object* v_s_49_){
_start:
{
lean_object* v_res_50_; 
v_res_50_ = lp_mathlib_Finset_max_x27___redArg(v_inst_48_, v_s_49_);
lean_dec_ref(v_inst_48_);
return v_res_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_max_x27(lean_object* v_00_u03b1_51_, lean_object* v_inst_52_, lean_object* v_s_53_, lean_object* v_H_54_){
_start:
{
lean_object* v___x_55_; 
v___x_55_ = lp_mathlib_Finset_max_x27___redArg(v_inst_52_, v_s_53_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_max_x27___boxed(lean_object* v_00_u03b1_56_, lean_object* v_inst_57_, lean_object* v_s_58_, lean_object* v_H_59_){
_start:
{
lean_object* v_res_60_; 
v_res_60_ = lp_mathlib_Finset_max_x27(v_00_u03b1_56_, v_inst_57_, v_s_58_, v_H_59_);
lean_dec_ref(v_inst_57_);
return v_res_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_min_x27___redArg(lean_object* v_inst_61_, lean_object* v_s_62_){
_start:
{
lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; 
v___x_63_ = lp_mathlib_LinearOrder_toLattice___redArg(v_inst_61_);
v___x_64_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_63_);
v___x_65_ = ((lean_object*)(lp_mathlib_Finset_max_x27___redArg___closed__0));
v___x_66_ = lp_mathlib_Finset_inf_x27___redArg(v___x_64_, v_s_62_, v___x_65_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_min_x27___redArg___boxed(lean_object* v_inst_67_, lean_object* v_s_68_){
_start:
{
lean_object* v_res_69_; 
v_res_69_ = lp_mathlib_Finset_min_x27___redArg(v_inst_67_, v_s_68_);
lean_dec_ref(v_inst_67_);
return v_res_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_min_x27(lean_object* v_00_u03b1_70_, lean_object* v_inst_71_, lean_object* v_s_72_, lean_object* v_H_73_){
_start:
{
lean_object* v___x_74_; 
v___x_74_ = lp_mathlib_Finset_min_x27___redArg(v_inst_71_, v_s_72_);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_min_x27___boxed(lean_object* v_00_u03b1_75_, lean_object* v_inst_76_, lean_object* v_s_77_, lean_object* v_H_78_){
_start:
{
lean_object* v_res_79_; 
v_res_79_ = lp_mathlib_Finset_min_x27(v_00_u03b1_75_, v_inst_76_, v_s_77_, v_H_78_);
lean_dec_ref(v_inst_76_);
return v_res_79_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Card(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Lattice_Fold(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Max(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Lattice_Fold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Finset_Max(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Finset_Card(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Lattice_Fold(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_Max(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Lattice_Fold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Max(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Finset_Max(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Finset_Max(builtin);
}
#ifdef __cplusplus
}
#endif
