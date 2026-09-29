// Lean compiler output
// Module: Aesop.RulePattern.Cache
// Imports: public import Init public meta import Init public import Aesop.Forward.Substitution public import Aesop.Rule.Name
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
static lean_once_cell_t lp_aesop_Aesop_instInhabitedRulePatternCache_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedRulePatternCache_default___closed__0;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedRulePatternCache_default___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedRulePatternCache_default___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedRulePatternCache_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedRulePatternCache;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instEmptyCollectionRulePatternCache;
static lean_object* _init_lp_aesop_Aesop_instInhabitedRulePatternCache_default___closed__0(void){
_start:
{
lean_object* v___x_1_; lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_1_ = lean_box(0);
v___x_2_ = lean_unsigned_to_nat(16u);
v___x_3_ = lean_mk_array(v___x_2_, v___x_1_);
return v___x_3_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedRulePatternCache_default___closed__1(void){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_4_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedRulePatternCache_default___closed__0, &lp_aesop_Aesop_instInhabitedRulePatternCache_default___closed__0_once, _init_lp_aesop_Aesop_instInhabitedRulePatternCache_default___closed__0);
v___x_5_ = lean_unsigned_to_nat(0u);
v___x_6_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_6_, 0, v___x_5_);
lean_ctor_set(v___x_6_, 1, v___x_4_);
return v___x_6_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedRulePatternCache_default(void){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedRulePatternCache_default___closed__1, &lp_aesop_Aesop_instInhabitedRulePatternCache_default___closed__1_once, _init_lp_aesop_Aesop_instInhabitedRulePatternCache_default___closed__1);
return v___x_7_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedRulePatternCache(void){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lp_aesop_Aesop_instInhabitedRulePatternCache_default;
return v___x_8_;
}
}
static lean_object* _init_lp_aesop_Aesop_instEmptyCollectionRulePatternCache(void){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedRulePatternCache_default___closed__1, &lp_aesop_Aesop_instInhabitedRulePatternCache_default___closed__1_once, _init_lp_aesop_Aesop_instInhabitedRulePatternCache_default___closed__1);
return v___x_9_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Forward_Substitution(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Rule_Name(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_RulePattern_Cache(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Forward_Substitution(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Rule_Name(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_instInhabitedRulePatternCache_default = _init_lp_aesop_Aesop_instInhabitedRulePatternCache_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedRulePatternCache_default);
lp_aesop_Aesop_instInhabitedRulePatternCache = _init_lp_aesop_Aesop_instInhabitedRulePatternCache();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedRulePatternCache);
lp_aesop_Aesop_instEmptyCollectionRulePatternCache = _init_lp_aesop_Aesop_instEmptyCollectionRulePatternCache();
lean_mark_persistent(lp_aesop_Aesop_instEmptyCollectionRulePatternCache);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_RulePattern_Cache(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Forward_Substitution(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Rule_Name(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_RulePattern_Cache(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Forward_Substitution(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Rule_Name(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RulePattern_Cache(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_RulePattern_Cache(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_RulePattern_Cache(builtin);
}
#ifdef __cplusplus
}
#endif
