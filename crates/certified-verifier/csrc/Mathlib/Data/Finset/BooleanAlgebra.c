// Lean compiler output
// Module: Mathlib.Data.Finset.BooleanAlgebra
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.Basic public import Mathlib.Data.Finset.Image public import Mathlib.Data.Fintype.Defs
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
lean_object* lp_mathlib_Finset_instGeneralizedBooleanAlgebra___redArg(lean_object*);
lean_object* lp_mathlib_Multiset_sub___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_ndunion___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_Multiset_decidableMem___aux__1___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_Fintype_decidableForallFintype___redArg(lean_object*, lean_object*);
uint8_t lp_mathlib_Finset_decidableDisjoint___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_boundedOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_boundedOrder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_booleanAlgebra___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_booleanAlgebra___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_booleanAlgebra___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_booleanAlgebra(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableCodisjoint___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableCodisjoint___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableCodisjoint___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableCodisjoint___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableCodisjoint(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableCodisjoint___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableIsCompl___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableIsCompl___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableIsCompl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableIsCompl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_boundedOrder___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_2_ = lean_box(0);
v___x_3_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3_, 0, v_inst_1_);
lean_ctor_set(v___x_3_, 1, v___x_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_boundedOrder(lean_object* v_00_u03b1_4_, lean_object* v_inst_5_){
_start:
{
lean_object* v___x_6_; 
v___x_6_ = lp_mathlib_Finset_boundedOrder___redArg(v_inst_5_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_booleanAlgebra___redArg___lam__0(lean_object* v_inst_7_, lean_object* v_inst_8_, lean_object* v_a_9_){
_start:
{
lean_object* v___x_10_; 
v___x_10_ = lp_mathlib_Multiset_sub___redArg(v_inst_7_, v_inst_8_, v_a_9_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_booleanAlgebra___redArg___lam__1(lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_x_13_, lean_object* v_y_14_){
_start:
{
lean_object* v___x_15_; lean_object* v___x_16_; 
lean_inc_ref(v_inst_11_);
v___x_15_ = lp_mathlib_Multiset_sub___redArg(v_inst_11_, v_inst_12_, v_x_13_);
v___x_16_ = lp_mathlib_Multiset_ndunion___redArg(v_inst_11_, v_y_14_, v___x_15_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_booleanAlgebra___redArg(lean_object* v_inst_17_, lean_object* v_inst_18_){
_start:
{
lean_object* v___x_19_; lean_object* v_toDistribLattice_20_; lean_object* v_toSDiff_21_; lean_object* v_toBot_22_; lean_object* v___f_23_; lean_object* v___f_24_; lean_object* v___x_25_; 
lean_inc_ref_n(v_inst_18_, 2);
v___x_19_ = lp_mathlib_Finset_instGeneralizedBooleanAlgebra___redArg(v_inst_18_);
v_toDistribLattice_20_ = lean_ctor_get(v___x_19_, 0);
lean_inc_ref(v_toDistribLattice_20_);
v_toSDiff_21_ = lean_ctor_get(v___x_19_, 1);
lean_inc(v_toSDiff_21_);
v_toBot_22_ = lean_ctor_get(v___x_19_, 2);
lean_inc(v_toBot_22_);
lean_dec_ref(v___x_19_);
lean_inc_n(v_inst_17_, 2);
v___f_23_ = lean_alloc_closure((void*)(lp_mathlib_Finset_booleanAlgebra___redArg___lam__0), 3, 2);
lean_closure_set(v___f_23_, 0, v_inst_18_);
lean_closure_set(v___f_23_, 1, v_inst_17_);
v___f_24_ = lean_alloc_closure((void*)(lp_mathlib_Finset_booleanAlgebra___redArg___lam__1), 4, 2);
lean_closure_set(v___f_24_, 0, v_inst_18_);
lean_closure_set(v___f_24_, 1, v_inst_17_);
v___x_25_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_25_, 0, v_toDistribLattice_20_);
lean_ctor_set(v___x_25_, 1, v___f_23_);
lean_ctor_set(v___x_25_, 2, v_toSDiff_21_);
lean_ctor_set(v___x_25_, 3, v___f_24_);
lean_ctor_set(v___x_25_, 4, v_inst_17_);
lean_ctor_set(v___x_25_, 5, v_toBot_22_);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_booleanAlgebra(lean_object* v_00_u03b1_26_, lean_object* v_inst_27_, lean_object* v_inst_28_){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lp_mathlib_Finset_booleanAlgebra___redArg(v_inst_27_, v_inst_28_);
return v___x_29_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableCodisjoint___redArg___lam__0(lean_object* v_inst_30_, lean_object* v_s_31_, lean_object* v_t_32_, lean_object* v_a_33_){
_start:
{
uint8_t v___x_34_; 
lean_inc(v_a_33_);
lean_inc_ref(v_inst_30_);
v___x_34_ = lp_mathlib_Multiset_decidableMem___aux__1___redArg(v_inst_30_, v_a_33_, v_s_31_);
if (v___x_34_ == 0)
{
uint8_t v___x_35_; 
v___x_35_ = lp_mathlib_Multiset_decidableMem___aux__1___redArg(v_inst_30_, v_a_33_, v_t_32_);
return v___x_35_;
}
else
{
lean_dec(v_a_33_);
lean_dec(v_t_32_);
lean_dec_ref(v_inst_30_);
return v___x_34_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableCodisjoint___redArg___lam__0___boxed(lean_object* v_inst_36_, lean_object* v_s_37_, lean_object* v_t_38_, lean_object* v_a_39_){
_start:
{
uint8_t v_res_40_; lean_object* v_r_41_; 
v_res_40_ = lp_mathlib_Finset_decidableCodisjoint___redArg___lam__0(v_inst_36_, v_s_37_, v_t_38_, v_a_39_);
v_r_41_ = lean_box(v_res_40_);
return v_r_41_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableCodisjoint___redArg(lean_object* v_s_42_, lean_object* v_t_43_, lean_object* v_inst_44_, lean_object* v_inst_45_){
_start:
{
lean_object* v___f_46_; uint8_t v___x_47_; 
v___f_46_ = lean_alloc_closure((void*)(lp_mathlib_Finset_decidableCodisjoint___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_46_, 0, v_inst_45_);
lean_closure_set(v___f_46_, 1, v_s_42_);
lean_closure_set(v___f_46_, 2, v_t_43_);
v___x_47_ = lp_mathlib_Fintype_decidableForallFintype___redArg(v___f_46_, v_inst_44_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableCodisjoint___redArg___boxed(lean_object* v_s_48_, lean_object* v_t_49_, lean_object* v_inst_50_, lean_object* v_inst_51_){
_start:
{
uint8_t v_res_52_; lean_object* v_r_53_; 
v_res_52_ = lp_mathlib_Finset_decidableCodisjoint___redArg(v_s_48_, v_t_49_, v_inst_50_, v_inst_51_);
v_r_53_ = lean_box(v_res_52_);
return v_r_53_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableCodisjoint(lean_object* v_00_u03b1_54_, lean_object* v_s_55_, lean_object* v_t_56_, lean_object* v_inst_57_, lean_object* v_inst_58_){
_start:
{
uint8_t v___x_59_; 
v___x_59_ = lp_mathlib_Finset_decidableCodisjoint___redArg(v_s_55_, v_t_56_, v_inst_57_, v_inst_58_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableCodisjoint___boxed(lean_object* v_00_u03b1_60_, lean_object* v_s_61_, lean_object* v_t_62_, lean_object* v_inst_63_, lean_object* v_inst_64_){
_start:
{
uint8_t v_res_65_; lean_object* v_r_66_; 
v_res_65_ = lp_mathlib_Finset_decidableCodisjoint(v_00_u03b1_60_, v_s_61_, v_t_62_, v_inst_63_, v_inst_64_);
v_r_66_ = lean_box(v_res_65_);
return v_r_66_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableIsCompl___redArg(lean_object* v_s_67_, lean_object* v_t_68_, lean_object* v_inst_69_, lean_object* v_inst_70_){
_start:
{
uint8_t v___x_71_; uint8_t v___x_72_; 
lean_inc_ref(v_inst_70_);
lean_inc(v_t_68_);
lean_inc(v_s_67_);
v___x_71_ = lp_mathlib_Finset_decidableCodisjoint___redArg(v_s_67_, v_t_68_, v_inst_69_, v_inst_70_);
v___x_72_ = lp_mathlib_Finset_decidableDisjoint___redArg(v_inst_70_, v_s_67_, v_t_68_);
if (v___x_72_ == 0)
{
return v___x_72_;
}
else
{
return v___x_71_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableIsCompl___redArg___boxed(lean_object* v_s_73_, lean_object* v_t_74_, lean_object* v_inst_75_, lean_object* v_inst_76_){
_start:
{
uint8_t v_res_77_; lean_object* v_r_78_; 
v_res_77_ = lp_mathlib_Finset_decidableIsCompl___redArg(v_s_73_, v_t_74_, v_inst_75_, v_inst_76_);
v_r_78_ = lean_box(v_res_77_);
return v_r_78_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableIsCompl(lean_object* v_00_u03b1_79_, lean_object* v_s_80_, lean_object* v_t_81_, lean_object* v_inst_82_, lean_object* v_inst_83_){
_start:
{
uint8_t v___x_84_; 
v___x_84_ = lp_mathlib_Finset_decidableIsCompl___redArg(v_s_80_, v_t_81_, v_inst_82_, v_inst_83_);
return v___x_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableIsCompl___boxed(lean_object* v_00_u03b1_85_, lean_object* v_s_86_, lean_object* v_t_87_, lean_object* v_inst_88_, lean_object* v_inst_89_){
_start:
{
uint8_t v_res_90_; lean_object* v_r_91_; 
v_res_90_ = lp_mathlib_Finset_decidableIsCompl(v_00_u03b1_85_, v_s_86_, v_t_87_, v_inst_88_, v_inst_89_);
v_r_91_ = lean_box(v_res_90_);
return v_r_91_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Image(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_BooleanAlgebra(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Finset_BooleanAlgebra(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Finset_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Image(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_BooleanAlgebra(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_BooleanAlgebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Finset_BooleanAlgebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Finset_BooleanAlgebra(builtin);
}
#ifdef __cplusplus
}
#endif
