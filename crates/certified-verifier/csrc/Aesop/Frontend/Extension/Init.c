// Lean compiler output
// Module: Aesop.Frontend.Extension.Init
// Imports: public import Init public meta import Init public import Aesop.RuleSet
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
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
static lean_once_cell_t lp_aesop_Aesop_instInhabitedDeclaredRuleSets_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedDeclaredRuleSets_default___closed__0;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedDeclaredRuleSets_default___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedDeclaredRuleSets_default___closed__1;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedDeclaredRuleSets_default___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedDeclaredRuleSets_default___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedDeclaredRuleSets_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedDeclaredRuleSets;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instEmptyCollectionDeclaredRuleSets;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Extension_Init_0__Aesop_initFn_00___x40_Aesop_Frontend_Extension_Init_1815514836____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Extension_Init_0__Aesop_initFn_00___x40_Aesop_Frontend_Extension_Init_1815514836____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_declaredRuleSetsRef;
LEAN_EXPORT lean_object* lp_aesop_Aesop_getDeclaredRuleSets();
LEAN_EXPORT lean_object* lp_aesop_Aesop_getDeclaredRuleSets___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getDefaultRuleSetNames();
LEAN_EXPORT lean_object* lp_aesop_Aesop_getDefaultRuleSetNames___boxed(lean_object*);
static lean_object* _init_lp_aesop_Aesop_instInhabitedDeclaredRuleSets_default___closed__0(void){
_start:
{
lean_object* v___x_1_; lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_1_ = lean_box(0);
v___x_2_ = lean_unsigned_to_nat(16u);
v___x_3_ = lean_mk_array(v___x_2_, v___x_1_);
return v___x_3_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedDeclaredRuleSets_default___closed__1(void){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_4_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedDeclaredRuleSets_default___closed__0, &lp_aesop_Aesop_instInhabitedDeclaredRuleSets_default___closed__0_once, _init_lp_aesop_Aesop_instInhabitedDeclaredRuleSets_default___closed__0);
v___x_5_ = lean_unsigned_to_nat(0u);
v___x_6_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_6_, 0, v___x_5_);
lean_ctor_set(v___x_6_, 1, v___x_4_);
return v___x_6_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedDeclaredRuleSets_default___closed__2(void){
_start:
{
lean_object* v___x_7_; lean_object* v___x_8_; 
v___x_7_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedDeclaredRuleSets_default___closed__1, &lp_aesop_Aesop_instInhabitedDeclaredRuleSets_default___closed__1_once, _init_lp_aesop_Aesop_instInhabitedDeclaredRuleSets_default___closed__1);
v___x_8_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_8_, 0, v___x_7_);
lean_ctor_set(v___x_8_, 1, v___x_7_);
return v___x_8_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedDeclaredRuleSets_default(void){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedDeclaredRuleSets_default___closed__2, &lp_aesop_Aesop_instInhabitedDeclaredRuleSets_default___closed__2_once, _init_lp_aesop_Aesop_instInhabitedDeclaredRuleSets_default___closed__2);
return v___x_9_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedDeclaredRuleSets(void){
_start:
{
lean_object* v___x_10_; 
v___x_10_ = lp_aesop_Aesop_instInhabitedDeclaredRuleSets_default;
return v___x_10_;
}
}
static lean_object* _init_lp_aesop_Aesop_instEmptyCollectionDeclaredRuleSets(void){
_start:
{
lean_object* v___x_11_; 
v___x_11_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedDeclaredRuleSets_default___closed__2, &lp_aesop_Aesop_instInhabitedDeclaredRuleSets_default___closed__2_once, _init_lp_aesop_Aesop_instInhabitedDeclaredRuleSets_default___closed__2);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Extension_Init_0__Aesop_initFn_00___x40_Aesop_Frontend_Extension_Init_1815514836____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_13_; lean_object* v___x_14_; lean_object* v___x_15_; 
v___x_13_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedDeclaredRuleSets_default___closed__2, &lp_aesop_Aesop_instInhabitedDeclaredRuleSets_default___closed__2_once, _init_lp_aesop_Aesop_instInhabitedDeclaredRuleSets_default___closed__2);
v___x_14_ = lean_st_mk_ref(v___x_13_);
v___x_15_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_15_, 0, v___x_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Extension_Init_0__Aesop_initFn_00___x40_Aesop_Frontend_Extension_Init_1815514836____hygCtx___hyg_2____boxed(lean_object* v_a_16_){
_start:
{
lean_object* v_res_17_; 
v_res_17_ = lp_aesop___private_Aesop_Frontend_Extension_Init_0__Aesop_initFn_00___x40_Aesop_Frontend_Extension_Init_1815514836____hygCtx___hyg_2_();
return v_res_17_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getDeclaredRuleSets(){
_start:
{
lean_object* v___x_19_; lean_object* v___x_20_; lean_object* v_ruleSets_21_; lean_object* v___x_22_; 
v___x_19_ = lp_aesop_Aesop_declaredRuleSetsRef;
v___x_20_ = lean_st_ref_get(v___x_19_);
v_ruleSets_21_ = lean_ctor_get(v___x_20_, 0);
lean_inc_ref(v_ruleSets_21_);
lean_dec(v___x_20_);
v___x_22_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_22_, 0, v_ruleSets_21_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getDeclaredRuleSets___boxed(lean_object* v_a_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_aesop_Aesop_getDeclaredRuleSets();
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getDefaultRuleSetNames(){
_start:
{
lean_object* v___x_26_; lean_object* v___x_27_; lean_object* v_defaultRuleSets_28_; lean_object* v___x_29_; 
v___x_26_ = lp_aesop_Aesop_declaredRuleSetsRef;
v___x_27_ = lean_st_ref_get(v___x_26_);
v_defaultRuleSets_28_ = lean_ctor_get(v___x_27_, 1);
lean_inc_ref(v_defaultRuleSets_28_);
lean_dec(v___x_27_);
v___x_29_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_29_, 0, v_defaultRuleSets_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getDefaultRuleSetNames___boxed(lean_object* v_a_30_){
_start:
{
lean_object* v_res_31_; 
v_res_31_ = lp_aesop_Aesop_getDefaultRuleSetNames();
return v_res_31_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_RuleSet(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Frontend_Extension_Init(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleSet(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_instInhabitedDeclaredRuleSets_default = _init_lp_aesop_Aesop_instInhabitedDeclaredRuleSets_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedDeclaredRuleSets_default);
lp_aesop_Aesop_instInhabitedDeclaredRuleSets = _init_lp_aesop_Aesop_instInhabitedDeclaredRuleSets();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedDeclaredRuleSets);
lp_aesop_Aesop_instEmptyCollectionDeclaredRuleSets = _init_lp_aesop_Aesop_instEmptyCollectionDeclaredRuleSets();
lean_mark_persistent(lp_aesop_Aesop_instEmptyCollectionDeclaredRuleSets);
res = lp_aesop___private_Aesop_Frontend_Extension_Init_0__Aesop_initFn_00___x40_Aesop_Frontend_Extension_Init_1815514836____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_aesop_Aesop_declaredRuleSetsRef = lean_io_result_get_value(res);
lean_mark_persistent(lp_aesop_Aesop_declaredRuleSetsRef);
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Frontend_Extension_Init(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_RuleSet(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Frontend_Extension_Init(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleSet(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Frontend_Extension_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Frontend_Extension_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Frontend_Extension_Init(builtin);
}
#ifdef __cplusplus
}
#endif
