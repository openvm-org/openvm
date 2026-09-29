// Lean compiler output
// Module: VM.Spec.Machine.Constants
// Imports: public import Init public meta import Init
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
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_PC__BITS;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_DEFAULT__PC__STEP;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_NUM__OPERANDS;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_LIMB__BITS;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_BLOCK__SIZE;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_ADDR__SPACE__OFFSET;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_PUBLIC__VALUES__AS;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_MAX__HINT__BUFFER__WORDS__BITS;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_DEFAULT__SUSPEND__EXIT__CODE;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_IMMEDIATES__AS;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_REGISTERS__AS;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_USER__MEMORY__AS;
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Machine_PC__BITS(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lean_unsigned_to_nat(30u);
return v___x_1_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Machine_DEFAULT__PC__STEP(void){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(4u);
return v___x_2_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Machine_NUM__OPERANDS(void){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_unsigned_to_nat(7u);
return v___x_3_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Machine_LIMB__BITS(void){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_unsigned_to_nat(8u);
return v___x_4_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Machine_BLOCK__SIZE(void){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lean_unsigned_to_nat(4u);
return v___x_5_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Machine_ADDR__SPACE__OFFSET(void){
_start:
{
lean_object* v___x_6_; 
v___x_6_ = lean_unsigned_to_nat(1u);
return v___x_6_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Machine_PUBLIC__VALUES__AS(void){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lean_unsigned_to_nat(3u);
return v___x_7_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Machine_MAX__HINT__BUFFER__WORDS__BITS(void){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lean_unsigned_to_nat(10u);
return v___x_8_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Machine_DEFAULT__SUSPEND__EXIT__CODE(void){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = lean_unsigned_to_nat(42u);
return v___x_9_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Machine_IMMEDIATES__AS(void){
_start:
{
lean_object* v___x_10_; 
v___x_10_ = lean_unsigned_to_nat(0u);
return v___x_10_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Machine_REGISTERS__AS(void){
_start:
{
lean_object* v___x_11_; 
v___x_11_ = lean_unsigned_to_nat(1u);
return v___x_11_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Machine_USER__MEMORY__AS(void){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lean_unsigned_to_nat(2u);
return v___x_12_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_openvm_x2dfv_VM_Spec_Machine_Constants(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
lean_initialize_runtime_module();
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_openvm_x2dfv_VM_Spec_Machine_PC__BITS = _init_lp_openvm_x2dfv_VM_Spec_Machine_PC__BITS();
lean_mark_persistent(lp_openvm_x2dfv_VM_Spec_Machine_PC__BITS);
lp_openvm_x2dfv_VM_Spec_Machine_DEFAULT__PC__STEP = _init_lp_openvm_x2dfv_VM_Spec_Machine_DEFAULT__PC__STEP();
lean_mark_persistent(lp_openvm_x2dfv_VM_Spec_Machine_DEFAULT__PC__STEP);
lp_openvm_x2dfv_VM_Spec_Machine_NUM__OPERANDS = _init_lp_openvm_x2dfv_VM_Spec_Machine_NUM__OPERANDS();
lean_mark_persistent(lp_openvm_x2dfv_VM_Spec_Machine_NUM__OPERANDS);
lp_openvm_x2dfv_VM_Spec_Machine_LIMB__BITS = _init_lp_openvm_x2dfv_VM_Spec_Machine_LIMB__BITS();
lean_mark_persistent(lp_openvm_x2dfv_VM_Spec_Machine_LIMB__BITS);
lp_openvm_x2dfv_VM_Spec_Machine_BLOCK__SIZE = _init_lp_openvm_x2dfv_VM_Spec_Machine_BLOCK__SIZE();
lean_mark_persistent(lp_openvm_x2dfv_VM_Spec_Machine_BLOCK__SIZE);
lp_openvm_x2dfv_VM_Spec_Machine_ADDR__SPACE__OFFSET = _init_lp_openvm_x2dfv_VM_Spec_Machine_ADDR__SPACE__OFFSET();
lean_mark_persistent(lp_openvm_x2dfv_VM_Spec_Machine_ADDR__SPACE__OFFSET);
lp_openvm_x2dfv_VM_Spec_Machine_PUBLIC__VALUES__AS = _init_lp_openvm_x2dfv_VM_Spec_Machine_PUBLIC__VALUES__AS();
lean_mark_persistent(lp_openvm_x2dfv_VM_Spec_Machine_PUBLIC__VALUES__AS);
lp_openvm_x2dfv_VM_Spec_Machine_MAX__HINT__BUFFER__WORDS__BITS = _init_lp_openvm_x2dfv_VM_Spec_Machine_MAX__HINT__BUFFER__WORDS__BITS();
lean_mark_persistent(lp_openvm_x2dfv_VM_Spec_Machine_MAX__HINT__BUFFER__WORDS__BITS);
lp_openvm_x2dfv_VM_Spec_Machine_DEFAULT__SUSPEND__EXIT__CODE = _init_lp_openvm_x2dfv_VM_Spec_Machine_DEFAULT__SUSPEND__EXIT__CODE();
lean_mark_persistent(lp_openvm_x2dfv_VM_Spec_Machine_DEFAULT__SUSPEND__EXIT__CODE);
lp_openvm_x2dfv_VM_Spec_Machine_IMMEDIATES__AS = _init_lp_openvm_x2dfv_VM_Spec_Machine_IMMEDIATES__AS();
lean_mark_persistent(lp_openvm_x2dfv_VM_Spec_Machine_IMMEDIATES__AS);
lp_openvm_x2dfv_VM_Spec_Machine_REGISTERS__AS = _init_lp_openvm_x2dfv_VM_Spec_Machine_REGISTERS__AS();
lean_mark_persistent(lp_openvm_x2dfv_VM_Spec_Machine_REGISTERS__AS);
lp_openvm_x2dfv_VM_Spec_Machine_USER__MEMORY__AS = _init_lp_openvm_x2dfv_VM_Spec_Machine_USER__MEMORY__AS();
lean_mark_persistent(lp_openvm_x2dfv_VM_Spec_Machine_USER__MEMORY__AS);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
