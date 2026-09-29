// Lean compiler output
// Module: Mathlib.Algebra.Group.Basic
// Imports: public import Init public meta import Init public import Aesop public import Mathlib.Algebra.Group.Defs public import Mathlib.Algebra.Notation.Defs public import Mathlib.Data.Int.Init public import Mathlib.Logic.Function.Iterate public import Mathlib.Tactic.SimpRw public import Mathlib.Tactic.SplitIfs
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
LEAN_EXPORT lean_object* lp_mathlib_DivisionMonoid_toDivInvOneMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionMonoid_toDivInvOneMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionMonoid_toDivInvOneMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionMonoid_toDivInvOneMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubtractionMonoid_toSubNegZeroMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubtractionMonoid_toSubNegZeroMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubtractionMonoid_toSubNegZeroMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubtractionMonoid_toSubNegZeroMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_toGrindNatModule___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_toGrindNatModule___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_toGrindNatModule(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_toGrindNatModule___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_toGrindIntModule___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_toGrindIntModule(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionMonoid_toDivInvOneMonoid___redArg(lean_object* v_inst_1_){
_start:
{
lean_inc_ref(v_inst_1_);
return v_inst_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionMonoid_toDivInvOneMonoid___redArg___boxed(lean_object* v_inst_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_DivisionMonoid_toDivInvOneMonoid___redArg(v_inst_2_);
lean_dec_ref(v_inst_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionMonoid_toDivInvOneMonoid(lean_object* v_00_u03b1_4_, lean_object* v_inst_5_){
_start:
{
lean_inc_ref(v_inst_5_);
return v_inst_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionMonoid_toDivInvOneMonoid___boxed(lean_object* v_00_u03b1_6_, lean_object* v_inst_7_){
_start:
{
lean_object* v_res_8_; 
v_res_8_ = lp_mathlib_DivisionMonoid_toDivInvOneMonoid(v_00_u03b1_6_, v_inst_7_);
lean_dec_ref(v_inst_7_);
return v_res_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubtractionMonoid_toSubNegZeroMonoid___redArg(lean_object* v_inst_9_){
_start:
{
lean_inc_ref(v_inst_9_);
return v_inst_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubtractionMonoid_toSubNegZeroMonoid___redArg___boxed(lean_object* v_inst_10_){
_start:
{
lean_object* v_res_11_; 
v_res_11_ = lp_mathlib_SubtractionMonoid_toSubNegZeroMonoid___redArg(v_inst_10_);
lean_dec_ref(v_inst_10_);
return v_res_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubtractionMonoid_toSubNegZeroMonoid(lean_object* v_00_u03b1_12_, lean_object* v_inst_13_){
_start:
{
lean_inc_ref(v_inst_13_);
return v_inst_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubtractionMonoid_toSubNegZeroMonoid___boxed(lean_object* v_00_u03b1_14_, lean_object* v_inst_15_){
_start:
{
lean_object* v_res_16_; 
v_res_16_ = lp_mathlib_SubtractionMonoid_toSubNegZeroMonoid(v_00_u03b1_14_, v_inst_15_);
lean_dec_ref(v_inst_15_);
return v_res_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_toGrindNatModule___redArg(lean_object* v_s_17_){
_start:
{
lean_object* v_toZero_18_; lean_object* v_toAdd_19_; lean_object* v_toNSMul_20_; lean_object* v___x_21_; lean_object* v___x_22_; 
v_toZero_18_ = lean_ctor_get(v_s_17_, 0);
v_toAdd_19_ = lean_ctor_get(v_s_17_, 1);
v_toNSMul_20_ = lean_ctor_get(v_s_17_, 2);
lean_inc(v_toAdd_19_);
lean_inc(v_toZero_18_);
v___x_21_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_21_, 0, v_toZero_18_);
lean_ctor_set(v___x_21_, 1, v_toAdd_19_);
lean_inc(v_toNSMul_20_);
v___x_22_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_22_, 0, v___x_21_);
lean_ctor_set(v___x_22_, 1, v_toNSMul_20_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_toGrindNatModule___redArg___boxed(lean_object* v_s_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_mathlib_AddCommMonoid_toGrindNatModule___redArg(v_s_23_);
lean_dec_ref(v_s_23_);
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_toGrindNatModule(lean_object* v_00_u03b1_25_, lean_object* v_s_26_){
_start:
{
lean_object* v___x_27_; 
v___x_27_ = lp_mathlib_AddCommMonoid_toGrindNatModule___redArg(v_s_26_);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_toGrindNatModule___boxed(lean_object* v_00_u03b1_28_, lean_object* v_s_29_){
_start:
{
lean_object* v_res_30_; 
v_res_30_ = lp_mathlib_AddCommMonoid_toGrindNatModule(v_00_u03b1_28_, v_s_29_);
lean_dec_ref(v_s_29_);
return v_res_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_toGrindIntModule___redArg(lean_object* v_s_31_){
_start:
{
lean_object* v_toAddMonoid_32_; lean_object* v_toNeg_33_; lean_object* v_toSub_34_; lean_object* v_toZSMul_35_; lean_object* v_toZero_36_; lean_object* v_toAdd_37_; lean_object* v_toNSMul_38_; lean_object* v___x_40_; uint8_t v_isShared_41_; uint8_t v_isSharedCheck_47_; 
v_toAddMonoid_32_ = lean_ctor_get(v_s_31_, 0);
lean_inc_ref(v_toAddMonoid_32_);
v_toNeg_33_ = lean_ctor_get(v_s_31_, 1);
lean_inc(v_toNeg_33_);
v_toSub_34_ = lean_ctor_get(v_s_31_, 2);
lean_inc(v_toSub_34_);
v_toZSMul_35_ = lean_ctor_get(v_s_31_, 3);
lean_inc(v_toZSMul_35_);
lean_dec_ref(v_s_31_);
v_toZero_36_ = lean_ctor_get(v_toAddMonoid_32_, 0);
v_toAdd_37_ = lean_ctor_get(v_toAddMonoid_32_, 1);
v_toNSMul_38_ = lean_ctor_get(v_toAddMonoid_32_, 2);
v_isSharedCheck_47_ = !lean_is_exclusive(v_toAddMonoid_32_);
if (v_isSharedCheck_47_ == 0)
{
v___x_40_ = v_toAddMonoid_32_;
v_isShared_41_ = v_isSharedCheck_47_;
goto v_resetjp_39_;
}
else
{
lean_inc(v_toNSMul_38_);
lean_inc(v_toAdd_37_);
lean_inc(v_toZero_36_);
lean_dec(v_toAddMonoid_32_);
v___x_40_ = lean_box(0);
v_isShared_41_ = v_isSharedCheck_47_;
goto v_resetjp_39_;
}
v_resetjp_39_:
{
lean_object* v___x_42_; lean_object* v___x_44_; 
v___x_42_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_42_, 0, v_toZero_36_);
lean_ctor_set(v___x_42_, 1, v_toAdd_37_);
if (v_isShared_41_ == 0)
{
lean_ctor_set(v___x_40_, 2, v_toSub_34_);
lean_ctor_set(v___x_40_, 1, v_toNeg_33_);
lean_ctor_set(v___x_40_, 0, v___x_42_);
v___x_44_ = v___x_40_;
goto v_reusejp_43_;
}
else
{
lean_object* v_reuseFailAlloc_46_; 
v_reuseFailAlloc_46_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_46_, 0, v___x_42_);
lean_ctor_set(v_reuseFailAlloc_46_, 1, v_toNeg_33_);
lean_ctor_set(v_reuseFailAlloc_46_, 2, v_toSub_34_);
v___x_44_ = v_reuseFailAlloc_46_;
goto v_reusejp_43_;
}
v_reusejp_43_:
{
lean_object* v___x_45_; 
v___x_45_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_45_, 0, v___x_44_);
lean_ctor_set(v___x_45_, 1, v_toNSMul_38_);
lean_ctor_set(v___x_45_, 2, v_toZSMul_35_);
return v___x_45_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_toGrindIntModule(lean_object* v_00_u03b1_48_, lean_object* v_s_49_){
_start:
{
lean_object* v___x_50_; 
v___x_50_ = lp_mathlib_AddCommGroup_toGrindIntModule___redArg(v_s_49_);
return v___x_50_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Notation_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Function_Iterate(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_SimpRw(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_SplitIfs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Notation_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Function_Iterate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_SimpRw(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_SplitIfs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Basic(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Notation_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Function_Iterate(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_SimpRw(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_SplitIfs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Notation_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Function_Iterate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_SimpRw(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_SplitIfs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
