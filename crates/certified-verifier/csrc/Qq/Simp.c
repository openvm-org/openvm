// Lean compiler output
// Module: Qq.Simp
// Imports: public import Init public meta import Init public import Qq.MetaM public import Lean.Meta.Tactic.Simp.Types
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
lean_object* lp_Qq_Qq_inferTypeQ(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_Simp_ResultQ_mk___redArg(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_Simp_ResultQ_mk___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_Simp_ResultQ_mk(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_Simp_ResultQ_mk___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_Simp_StepQ_done___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_Simp_StepQ_done(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_Simp_StepQ_done___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_Simp_StepQ_visit___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_Simp_StepQ_visit(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_Simp_StepQ_visit___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_Simp_StepQ_continue___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_Simp_StepQ_continue(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_Simp_StepQ_continue___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_Simp_Simproc_ofQ(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_Simp_Simproc_ofQ___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_Simp_ResultQ_mk___redArg(lean_object* v_expr_1_, lean_object* v_proof_x3f_2_, uint8_t v_cache_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_4_, 0, v_expr_1_);
lean_ctor_set(v___x_4_, 1, v_proof_x3f_2_);
lean_ctor_set_uint8(v___x_4_, sizeof(void*)*2, v_cache_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_Simp_ResultQ_mk___redArg___boxed(lean_object* v_expr_5_, lean_object* v_proof_x3f_6_, lean_object* v_cache_7_){
_start:
{
uint8_t v_cache_boxed_8_; lean_object* v_res_9_; 
v_cache_boxed_8_ = lean_unbox(v_cache_7_);
v_res_9_ = lp_Qq_Lean_Meta_Simp_ResultQ_mk___redArg(v_expr_5_, v_proof_x3f_6_, v_cache_boxed_8_);
return v_res_9_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_Simp_ResultQ_mk(lean_object* v_u_10_, lean_object* v_00_u03b1_11_, lean_object* v_e_12_, lean_object* v_expr_13_, lean_object* v_proof_x3f_14_, uint8_t v_cache_15_){
_start:
{
lean_object* v___x_16_; 
v___x_16_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_16_, 0, v_expr_13_);
lean_ctor_set(v___x_16_, 1, v_proof_x3f_14_);
lean_ctor_set_uint8(v___x_16_, sizeof(void*)*2, v_cache_15_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_Simp_ResultQ_mk___boxed(lean_object* v_u_17_, lean_object* v_00_u03b1_18_, lean_object* v_e_19_, lean_object* v_expr_20_, lean_object* v_proof_x3f_21_, lean_object* v_cache_22_){
_start:
{
uint8_t v_cache_boxed_23_; lean_object* v_res_24_; 
v_cache_boxed_23_ = lean_unbox(v_cache_22_);
v_res_24_ = lp_Qq_Lean_Meta_Simp_ResultQ_mk(v_u_17_, v_00_u03b1_18_, v_e_19_, v_expr_20_, v_proof_x3f_21_, v_cache_boxed_23_);
lean_dec_ref(v_e_19_);
lean_dec_ref(v_00_u03b1_18_);
lean_dec(v_u_17_);
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_Simp_StepQ_done___redArg(lean_object* v_r_25_){
_start:
{
lean_object* v___x_26_; 
v___x_26_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_26_, 0, v_r_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_Simp_StepQ_done(lean_object* v_u_27_, lean_object* v_00_u03b1_28_, lean_object* v_e_29_, lean_object* v_r_30_){
_start:
{
lean_object* v___x_31_; 
v___x_31_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_31_, 0, v_r_30_);
return v___x_31_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_Simp_StepQ_done___boxed(lean_object* v_u_32_, lean_object* v_00_u03b1_33_, lean_object* v_e_34_, lean_object* v_r_35_){
_start:
{
lean_object* v_res_36_; 
v_res_36_ = lp_Qq_Lean_Meta_Simp_StepQ_done(v_u_32_, v_00_u03b1_33_, v_e_34_, v_r_35_);
lean_dec_ref(v_e_34_);
lean_dec_ref(v_00_u03b1_33_);
lean_dec(v_u_32_);
return v_res_36_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_Simp_StepQ_visit___redArg(lean_object* v_r_37_){
_start:
{
lean_object* v___x_38_; 
v___x_38_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_38_, 0, v_r_37_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_Simp_StepQ_visit(lean_object* v_u_39_, lean_object* v_00_u03b1_40_, lean_object* v_e_41_, lean_object* v_r_42_){
_start:
{
lean_object* v___x_43_; 
v___x_43_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_43_, 0, v_r_42_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_Simp_StepQ_visit___boxed(lean_object* v_u_44_, lean_object* v_00_u03b1_45_, lean_object* v_e_46_, lean_object* v_r_47_){
_start:
{
lean_object* v_res_48_; 
v_res_48_ = lp_Qq_Lean_Meta_Simp_StepQ_visit(v_u_44_, v_00_u03b1_45_, v_e_46_, v_r_47_);
lean_dec_ref(v_e_46_);
lean_dec_ref(v_00_u03b1_45_);
lean_dec(v_u_44_);
return v_res_48_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_Simp_StepQ_continue___redArg(lean_object* v_r_49_){
_start:
{
lean_object* v___x_50_; 
v___x_50_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_50_, 0, v_r_49_);
return v___x_50_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_Simp_StepQ_continue(lean_object* v_u_51_, lean_object* v_00_u03b1_52_, lean_object* v_e_53_, lean_object* v_r_54_){
_start:
{
lean_object* v___x_55_; 
v___x_55_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_55_, 0, v_r_54_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_Simp_StepQ_continue___boxed(lean_object* v_u_56_, lean_object* v_00_u03b1_57_, lean_object* v_e_58_, lean_object* v_r_59_){
_start:
{
lean_object* v_res_60_; 
v_res_60_ = lp_Qq_Lean_Meta_Simp_StepQ_continue(v_u_56_, v_00_u03b1_57_, v_e_58_, v_r_59_);
lean_dec_ref(v_e_58_);
lean_dec_ref(v_00_u03b1_57_);
lean_dec(v_u_56_);
return v_res_60_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_Simp_Simproc_ofQ(lean_object* v_proc_61_, lean_object* v_e_62_, lean_object* v_a_63_, lean_object* v_a_64_, lean_object* v_a_65_, lean_object* v_a_66_, lean_object* v_a_67_, lean_object* v_a_68_, lean_object* v_a_69_){
_start:
{
lean_object* v___x_71_; 
v___x_71_ = lp_Qq_Qq_inferTypeQ(v_e_62_, v_a_66_, v_a_67_, v_a_68_, v_a_69_);
if (lean_obj_tag(v___x_71_) == 0)
{
lean_object* v_a_72_; lean_object* v_snd_73_; lean_object* v_fst_74_; lean_object* v_fst_75_; lean_object* v_snd_76_; lean_object* v___x_77_; 
v_a_72_ = lean_ctor_get(v___x_71_, 0);
lean_inc(v_a_72_);
lean_dec_ref_known(v___x_71_, 1);
v_snd_73_ = lean_ctor_get(v_a_72_, 1);
lean_inc(v_snd_73_);
v_fst_74_ = lean_ctor_get(v_a_72_, 0);
lean_inc(v_fst_74_);
lean_dec(v_a_72_);
v_fst_75_ = lean_ctor_get(v_snd_73_, 0);
lean_inc(v_fst_75_);
v_snd_76_ = lean_ctor_get(v_snd_73_, 1);
lean_inc(v_snd_76_);
lean_dec(v_snd_73_);
lean_inc(v_a_69_);
lean_inc_ref(v_a_68_);
lean_inc(v_a_67_);
lean_inc_ref(v_a_66_);
lean_inc(v_a_65_);
lean_inc_ref(v_a_64_);
lean_inc(v_a_63_);
v___x_77_ = lean_apply_11(v_proc_61_, v_fst_74_, v_fst_75_, v_snd_76_, v_a_63_, v_a_64_, v_a_65_, v_a_66_, v_a_67_, v_a_68_, v_a_69_, lean_box(0));
return v___x_77_;
}
else
{
lean_object* v_a_78_; lean_object* v___x_80_; uint8_t v_isShared_81_; uint8_t v_isSharedCheck_85_; 
lean_dec_ref(v_proc_61_);
v_a_78_ = lean_ctor_get(v___x_71_, 0);
v_isSharedCheck_85_ = !lean_is_exclusive(v___x_71_);
if (v_isSharedCheck_85_ == 0)
{
v___x_80_ = v___x_71_;
v_isShared_81_ = v_isSharedCheck_85_;
goto v_resetjp_79_;
}
else
{
lean_inc(v_a_78_);
lean_dec(v___x_71_);
v___x_80_ = lean_box(0);
v_isShared_81_ = v_isSharedCheck_85_;
goto v_resetjp_79_;
}
v_resetjp_79_:
{
lean_object* v___x_83_; 
if (v_isShared_81_ == 0)
{
v___x_83_ = v___x_80_;
goto v_reusejp_82_;
}
else
{
lean_object* v_reuseFailAlloc_84_; 
v_reuseFailAlloc_84_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_84_, 0, v_a_78_);
v___x_83_ = v_reuseFailAlloc_84_;
goto v_reusejp_82_;
}
v_reusejp_82_:
{
return v___x_83_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_Simp_Simproc_ofQ___boxed(lean_object* v_proc_86_, lean_object* v_e_87_, lean_object* v_a_88_, lean_object* v_a_89_, lean_object* v_a_90_, lean_object* v_a_91_, lean_object* v_a_92_, lean_object* v_a_93_, lean_object* v_a_94_, lean_object* v_a_95_){
_start:
{
lean_object* v_res_96_; 
v_res_96_ = lp_Qq_Lean_Meta_Simp_Simproc_ofQ(v_proc_86_, v_e_87_, v_a_88_, v_a_89_, v_a_90_, v_a_91_, v_a_92_, v_a_93_, v_a_94_);
lean_dec(v_a_94_);
lean_dec_ref(v_a_93_);
lean_dec(v_a_92_);
lean_dec_ref(v_a_91_);
lean_dec(v_a_90_);
lean_dec_ref(v_a_89_);
lean_dec(v_a_88_);
return v_res_96_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Qq_Qq_MetaM(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Simp_Types(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_Qq_Qq_Simp(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Qq_Qq_MetaM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Simp_Types(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_Qq_Qq_Simp(uint8_t builtin) {
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
lean_object* initialize_Qq_Qq_MetaM(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Simp_Types(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_Qq_Qq_Simp(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Qq_Qq_MetaM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Simp_Types(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Qq_Qq_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_Qq_Qq_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_Qq_Qq_Simp(builtin);
}
#ifdef __cplusplus
}
#endif
