// Lean compiler output
// Module: Aesop.Search.Queue.Class
// Imports: public import Init public meta import Init public import Aesop.Tree.Data
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_Queue_init_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Queue_init_x27___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Queue_init_x27(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Queue_init_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Queue_init_x27___redArg(lean_object* v_inst_1_, lean_object* v_grefs_2_){
_start:
{
lean_object* v_init_4_; lean_object* v_addGoals_5_; lean_object* v___x_6_; lean_object* v___x_7_; 
v_init_4_ = lean_ctor_get(v_inst_1_, 0);
lean_inc_ref(v_init_4_);
v_addGoals_5_ = lean_ctor_get(v_inst_1_, 1);
lean_inc_ref(v_addGoals_5_);
lean_dec_ref(v_inst_1_);
v___x_6_ = lean_apply_1(v_init_4_, lean_box(0));
v___x_7_ = lean_apply_3(v_addGoals_5_, v___x_6_, v_grefs_2_, lean_box(0));
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Queue_init_x27___redArg___boxed(lean_object* v_inst_8_, lean_object* v_grefs_9_, lean_object* v_a_10_){
_start:
{
lean_object* v_res_11_; 
v_res_11_ = lp_aesop_Aesop_Queue_init_x27___redArg(v_inst_8_, v_grefs_9_);
return v_res_11_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Queue_init_x27(lean_object* v_Q_12_, lean_object* v_inst_13_, lean_object* v_grefs_14_){
_start:
{
lean_object* v___x_16_; 
v___x_16_ = lp_aesop_Aesop_Queue_init_x27___redArg(v_inst_13_, v_grefs_14_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Queue_init_x27___boxed(lean_object* v_Q_17_, lean_object* v_inst_18_, lean_object* v_grefs_19_, lean_object* v_a_20_){
_start:
{
lean_object* v_res_21_; 
v_res_21_ = lp_aesop_Aesop_Queue_init_x27(v_Q_17_, v_inst_18_, v_grefs_19_);
return v_res_21_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Tree_Data(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Search_Queue_Class(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tree_Data(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Search_Queue_Class(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Tree_Data(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Search_Queue_Class(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_Data(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Search_Queue_Class(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Search_Queue_Class(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Search_Queue_Class(builtin);
}
#ifdef __cplusplus
}
#endif
