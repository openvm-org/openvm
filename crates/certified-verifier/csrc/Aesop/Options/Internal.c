// Lean compiler output
// Module: Aesop.Options.Internal
// Imports: public import Init public meta import Init public import Aesop.Check public import Aesop.Options.Public
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
extern lean_object* lp_aesop_Aesop_instInhabitedOptions_default;
extern lean_object* lp_aesop_Aesop_Check_script_steps;
lean_object* lp_aesop_Aesop_Check_isEnabled___redArg(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_KVMap_instValueBool;
extern lean_object* lp_aesop_Aesop_Check_script;
extern lean_object* lp_aesop_Aesop_aesop_dev_generateScript;
lean_object* l_Lean_Option_get___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_instInhabitedOptions_x27_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedOptions_x27_default___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedOptions_x27_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedOptions_x27;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___redArg___lam__3(lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_aesop_Aesop_instInhabitedOptions_x27_default___closed__0(void){
_start:
{
lean_object* v___x_1_; uint8_t v___x_2_; lean_object* v___x_3_; lean_object* v___x_4_; 
v___x_1_ = lean_box(0);
v___x_2_ = 0;
v___x_3_ = lp_aesop_Aesop_instInhabitedOptions_default;
v___x_4_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_4_, 0, v___x_3_);
lean_ctor_set(v___x_4_, 1, v___x_1_);
lean_ctor_set_uint8(v___x_4_, sizeof(void*)*2, v___x_2_);
return v___x_4_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedOptions_x27_default(void){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedOptions_x27_default___closed__0, &lp_aesop_Aesop_instInhabitedOptions_x27_default___closed__0_once, _init_lp_aesop_Aesop_instInhabitedOptions_x27_default___closed__0);
return v___x_5_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedOptions_x27(void){
_start:
{
lean_object* v___x_6_; 
v___x_6_ = lp_aesop_Aesop_instInhabitedOptions_x27_default;
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___redArg___lam__0(lean_object* v_inst_7_, lean_object* v_inst_8_, lean_object* v_toPure_9_, uint8_t v_b_10_){
_start:
{
if (v_b_10_ == 0)
{
lean_object* v___x_11_; lean_object* v___x_12_; 
lean_dec(v_toPure_9_);
v___x_11_ = lp_aesop_Aesop_Check_script_steps;
v___x_12_ = lp_aesop_Aesop_Check_isEnabled___redArg(v_inst_7_, v_inst_8_, v___x_11_);
return v___x_12_;
}
else
{
lean_object* v___x_13_; lean_object* v___x_14_; 
lean_dec(v_inst_8_);
lean_dec_ref(v_inst_7_);
v___x_13_ = lean_box(v_b_10_);
v___x_14_ = lean_apply_2(v_toPure_9_, lean_box(0), v___x_13_);
return v___x_14_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___redArg___lam__0___boxed(lean_object* v_inst_15_, lean_object* v_inst_16_, lean_object* v_toPure_17_, lean_object* v_b_18_){
_start:
{
uint8_t v_b_boxed_19_; lean_object* v_res_20_; 
v_b_boxed_19_ = lean_unbox(v_b_18_);
v_res_20_ = lp_aesop_Aesop_Options_toOptions_x27___redArg___lam__0(v_inst_15_, v_inst_16_, v_toPure_17_, v_b_boxed_19_);
return v_res_20_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___redArg___lam__1(lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_toBind_23_, lean_object* v___f_24_, lean_object* v_toPure_25_, uint8_t v_b_26_){
_start:
{
if (v_b_26_ == 0)
{
lean_object* v___x_27_; lean_object* v___x_28_; lean_object* v___x_29_; 
lean_dec(v_toPure_25_);
v___x_27_ = lp_aesop_Aesop_Check_script;
v___x_28_ = lp_aesop_Aesop_Check_isEnabled___redArg(v_inst_21_, v_inst_22_, v___x_27_);
v___x_29_ = lean_apply_4(v_toBind_23_, lean_box(0), lean_box(0), v___x_28_, v___f_24_);
return v___x_29_;
}
else
{
lean_object* v___x_30_; lean_object* v___x_31_; 
lean_dec(v___f_24_);
lean_dec(v_toBind_23_);
lean_dec(v_inst_22_);
lean_dec_ref(v_inst_21_);
v___x_30_ = lean_box(v_b_26_);
v___x_31_ = lean_apply_2(v_toPure_25_, lean_box(0), v___x_30_);
return v___x_31_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___redArg___lam__1___boxed(lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_toBind_34_, lean_object* v___f_35_, lean_object* v_toPure_36_, lean_object* v_b_37_){
_start:
{
uint8_t v_b_boxed_38_; lean_object* v_res_39_; 
v_b_boxed_38_ = lean_unbox(v_b_37_);
v_res_39_ = lp_aesop_Aesop_Options_toOptions_x27___redArg___lam__1(v_inst_32_, v_inst_33_, v_toBind_34_, v___f_35_, v_toPure_36_, v_b_boxed_38_);
return v_res_39_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___redArg___lam__2(lean_object* v_opts_40_, lean_object* v_toPure_41_, lean_object* v_toBind_42_, lean_object* v___f_43_, uint8_t v_b_44_){
_start:
{
if (v_b_44_ == 0)
{
uint8_t v_traceScript_45_; lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; 
v_traceScript_45_ = lean_ctor_get_uint8(v_opts_40_, sizeof(void*)*6 + 6);
v___x_46_ = lean_box(v_traceScript_45_);
v___x_47_ = lean_apply_2(v_toPure_41_, lean_box(0), v___x_46_);
v___x_48_ = lean_apply_4(v_toBind_42_, lean_box(0), lean_box(0), v___x_47_, v___f_43_);
return v___x_48_;
}
else
{
lean_object* v___x_49_; lean_object* v___x_50_; 
lean_dec(v___f_43_);
lean_dec(v_toBind_42_);
v___x_49_ = lean_box(v_b_44_);
v___x_50_ = lean_apply_2(v_toPure_41_, lean_box(0), v___x_49_);
return v___x_50_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___redArg___lam__2___boxed(lean_object* v_opts_51_, lean_object* v_toPure_52_, lean_object* v_toBind_53_, lean_object* v___f_54_, lean_object* v_b_55_){
_start:
{
uint8_t v_b_boxed_56_; lean_object* v_res_57_; 
v_b_boxed_56_ = lean_unbox(v_b_55_);
v_res_57_ = lp_aesop_Aesop_Options_toOptions_x27___redArg___lam__2(v_opts_51_, v_toPure_52_, v_toBind_53_, v___f_54_, v_b_boxed_56_);
lean_dec_ref(v_opts_51_);
return v_res_57_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___redArg___lam__3(lean_object* v_opts_58_, lean_object* v_forwardMaxDepth_x3f_59_, lean_object* v_toPure_60_, uint8_t v_generateScript_61_){
_start:
{
lean_object* v___x_62_; lean_object* v___x_63_; 
v___x_62_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_62_, 0, v_opts_58_);
lean_ctor_set(v___x_62_, 1, v_forwardMaxDepth_x3f_59_);
lean_ctor_set_uint8(v___x_62_, sizeof(void*)*2, v_generateScript_61_);
v___x_63_ = lean_apply_2(v_toPure_60_, lean_box(0), v___x_62_);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___redArg___lam__3___boxed(lean_object* v_opts_64_, lean_object* v_forwardMaxDepth_x3f_65_, lean_object* v_toPure_66_, lean_object* v_generateScript_67_){
_start:
{
uint8_t v_generateScript_boxed_68_; lean_object* v_res_69_; 
v_generateScript_boxed_68_ = lean_unbox(v_generateScript_67_);
v_res_69_ = lp_aesop_Aesop_Options_toOptions_x27___redArg___lam__3(v_opts_64_, v_forwardMaxDepth_x3f_65_, v_toPure_66_, v_generateScript_boxed_68_);
return v_res_69_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___redArg___lam__4(lean_object* v___x_70_, lean_object* v_toPure_71_, lean_object* v_toBind_72_, lean_object* v___f_73_, lean_object* v___f_74_, lean_object* v_____do__lift_75_){
_start:
{
lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; 
v___x_76_ = lp_aesop_Aesop_aesop_dev_generateScript;
v___x_77_ = l_Lean_Option_get___redArg(v___x_70_, v_____do__lift_75_, v___x_76_);
v___x_78_ = lean_apply_2(v_toPure_71_, lean_box(0), v___x_77_);
lean_inc(v_toBind_72_);
v___x_79_ = lean_apply_4(v_toBind_72_, lean_box(0), lean_box(0), v___x_78_, v___f_73_);
v___x_80_ = lean_apply_4(v_toBind_72_, lean_box(0), lean_box(0), v___x_79_, v___f_74_);
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___redArg___lam__4___boxed(lean_object* v___x_81_, lean_object* v_toPure_82_, lean_object* v_toBind_83_, lean_object* v___f_84_, lean_object* v___f_85_, lean_object* v_____do__lift_86_){
_start:
{
lean_object* v_res_87_; 
v_res_87_ = lp_aesop_Aesop_Options_toOptions_x27___redArg___lam__4(v___x_81_, v_toPure_82_, v_toBind_83_, v___f_84_, v___f_85_, v_____do__lift_86_);
lean_dec_ref(v_____do__lift_86_);
return v_res_87_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___redArg(lean_object* v_inst_88_, lean_object* v_inst_89_, lean_object* v_opts_90_, lean_object* v_forwardMaxDepth_x3f_91_){
_start:
{
lean_object* v___x_92_; lean_object* v_toApplicative_93_; lean_object* v_toBind_94_; lean_object* v_toPure_95_; lean_object* v___f_96_; lean_object* v___f_97_; lean_object* v___f_98_; lean_object* v___f_99_; lean_object* v___f_100_; lean_object* v___x_101_; 
v___x_92_ = l_Lean_KVMap_instValueBool;
v_toApplicative_93_ = lean_ctor_get(v_inst_88_, 0);
v_toBind_94_ = lean_ctor_get(v_inst_88_, 1);
lean_inc_n(v_toBind_94_, 4);
v_toPure_95_ = lean_ctor_get(v_toApplicative_93_, 1);
lean_inc_n(v_toPure_95_, 5);
lean_inc_n(v_inst_89_, 2);
lean_inc_ref(v_inst_88_);
v___f_96_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Options_toOptions_x27___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_96_, 0, v_inst_88_);
lean_closure_set(v___f_96_, 1, v_inst_89_);
lean_closure_set(v___f_96_, 2, v_toPure_95_);
v___f_97_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Options_toOptions_x27___redArg___lam__1___boxed), 6, 5);
lean_closure_set(v___f_97_, 0, v_inst_88_);
lean_closure_set(v___f_97_, 1, v_inst_89_);
lean_closure_set(v___f_97_, 2, v_toBind_94_);
lean_closure_set(v___f_97_, 3, v___f_96_);
lean_closure_set(v___f_97_, 4, v_toPure_95_);
lean_inc_ref(v_opts_90_);
v___f_98_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Options_toOptions_x27___redArg___lam__2___boxed), 5, 4);
lean_closure_set(v___f_98_, 0, v_opts_90_);
lean_closure_set(v___f_98_, 1, v_toPure_95_);
lean_closure_set(v___f_98_, 2, v_toBind_94_);
lean_closure_set(v___f_98_, 3, v___f_97_);
v___f_99_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Options_toOptions_x27___redArg___lam__3___boxed), 4, 3);
lean_closure_set(v___f_99_, 0, v_opts_90_);
lean_closure_set(v___f_99_, 1, v_forwardMaxDepth_x3f_91_);
lean_closure_set(v___f_99_, 2, v_toPure_95_);
v___f_100_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Options_toOptions_x27___redArg___lam__4___boxed), 6, 5);
lean_closure_set(v___f_100_, 0, v___x_92_);
lean_closure_set(v___f_100_, 1, v_toPure_95_);
lean_closure_set(v___f_100_, 2, v_toBind_94_);
lean_closure_set(v___f_100_, 3, v___f_98_);
lean_closure_set(v___f_100_, 4, v___f_99_);
v___x_101_ = lean_apply_4(v_toBind_94_, lean_box(0), lean_box(0), v_inst_89_, v___f_100_);
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27(lean_object* v_m_102_, lean_object* v_inst_103_, lean_object* v_inst_104_, lean_object* v_opts_105_, lean_object* v_forwardMaxDepth_x3f_106_){
_start:
{
lean_object* v___x_107_; 
v___x_107_ = lp_aesop_Aesop_Options_toOptions_x27___redArg(v_inst_103_, v_inst_104_, v_opts_105_, v_forwardMaxDepth_x3f_106_);
return v___x_107_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Check(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Options_Public(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Options_Internal(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Check(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Options_Public(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_instInhabitedOptions_x27_default = _init_lp_aesop_Aesop_instInhabitedOptions_x27_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedOptions_x27_default);
lp_aesop_Aesop_instInhabitedOptions_x27 = _init_lp_aesop_Aesop_instInhabitedOptions_x27();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedOptions_x27);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Options_Internal(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Check(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Options_Public(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Options_Internal(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Check(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Options_Public(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Options_Internal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Options_Internal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Options_Internal(builtin);
}
#ifdef __cplusplus
}
#endif
