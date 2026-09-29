// Lean compiler output
// Module: Mathlib.GroupTheory.OrderOfElement
// Imports: public import Init public meta import Init public import Mathlib.Algebra.CharP.Two public import Mathlib.Algebra.Group.Commute.Basic public import Mathlib.Algebra.Group.Pointwise.Set.Finite public import Mathlib.Algebra.Group.Subgroup.Finite public import Mathlib.Algebra.Group.TransferInstance public import Mathlib.Algebra.Module.NatInt public import Mathlib.Algebra.Order.Group.Action public import Mathlib.Algebra.Order.Ring.Abs public import Mathlib.Data.Int.ModEq public import Mathlib.Dynamics.PeriodicPts.Lemmas public import Mathlib.GroupTheory.Index public import Mathlib.NumberTheory.Divisors public import Mathlib.Order.Interval.Set.Infinite
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
LEAN_EXPORT lean_object* lp_mathlib_submonoidOfIdempotent(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_submonoidOfIdempotent___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addSubmonoidOfIdempotent(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addSubmonoidOfIdempotent___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_subgroupOfIdempotent(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_subgroupOfIdempotent___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addSubgroupOfIdempotent(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addSubgroupOfIdempotent___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_powCardSubgroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_powCardSubgroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_smulCardAddSubgroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_smulCardAddSubgroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_submonoidOfIdempotent(lean_object* v_M_1_, lean_object* v_inst_2_, lean_object* v_inst_3_, lean_object* v_S_4_, lean_object* v_hS1_5_, lean_object* v_hS2_6_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lean_box(0);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_submonoidOfIdempotent___boxed(lean_object* v_M_8_, lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_S_11_, lean_object* v_hS1_12_, lean_object* v_hS2_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_submonoidOfIdempotent(v_M_8_, v_inst_9_, v_inst_10_, v_S_11_, v_hS1_12_, v_hS2_13_);
lean_dec_ref(v_inst_9_);
return v_res_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addSubmonoidOfIdempotent(lean_object* v_M_15_, lean_object* v_inst_16_, lean_object* v_inst_17_, lean_object* v_S_18_, lean_object* v_hS1_19_, lean_object* v_hS2_20_){
_start:
{
lean_object* v___x_21_; 
v___x_21_ = lean_box(0);
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addSubmonoidOfIdempotent___boxed(lean_object* v_M_22_, lean_object* v_inst_23_, lean_object* v_inst_24_, lean_object* v_S_25_, lean_object* v_hS1_26_, lean_object* v_hS2_27_){
_start:
{
lean_object* v_res_28_; 
v_res_28_ = lp_mathlib_addSubmonoidOfIdempotent(v_M_22_, v_inst_23_, v_inst_24_, v_S_25_, v_hS1_26_, v_hS2_27_);
lean_dec_ref(v_inst_23_);
return v_res_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_subgroupOfIdempotent(lean_object* v_G_29_, lean_object* v_inst_30_, lean_object* v_inst_31_, lean_object* v_S_32_, lean_object* v_hS1_33_, lean_object* v_hS2_34_){
_start:
{
lean_object* v___x_35_; 
v___x_35_ = lean_box(0);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_subgroupOfIdempotent___boxed(lean_object* v_G_36_, lean_object* v_inst_37_, lean_object* v_inst_38_, lean_object* v_S_39_, lean_object* v_hS1_40_, lean_object* v_hS2_41_){
_start:
{
lean_object* v_res_42_; 
v_res_42_ = lp_mathlib_subgroupOfIdempotent(v_G_36_, v_inst_37_, v_inst_38_, v_S_39_, v_hS1_40_, v_hS2_41_);
lean_dec_ref(v_inst_37_);
return v_res_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addSubgroupOfIdempotent(lean_object* v_G_43_, lean_object* v_inst_44_, lean_object* v_inst_45_, lean_object* v_S_46_, lean_object* v_hS1_47_, lean_object* v_hS2_48_){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lean_box(0);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addSubgroupOfIdempotent___boxed(lean_object* v_G_50_, lean_object* v_inst_51_, lean_object* v_inst_52_, lean_object* v_S_53_, lean_object* v_hS1_54_, lean_object* v_hS2_55_){
_start:
{
lean_object* v_res_56_; 
v_res_56_ = lp_mathlib_addSubgroupOfIdempotent(v_G_50_, v_inst_51_, v_inst_52_, v_S_53_, v_hS1_54_, v_hS2_55_);
lean_dec_ref(v_inst_51_);
return v_res_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_powCardSubgroup(lean_object* v_G_57_, lean_object* v_inst_58_, lean_object* v_inst_59_, lean_object* v_S_60_, lean_object* v_hS_61_){
_start:
{
lean_object* v___x_62_; 
v___x_62_ = lean_box(0);
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_powCardSubgroup___boxed(lean_object* v_G_63_, lean_object* v_inst_64_, lean_object* v_inst_65_, lean_object* v_S_66_, lean_object* v_hS_67_){
_start:
{
lean_object* v_res_68_; 
v_res_68_ = lp_mathlib_powCardSubgroup(v_G_63_, v_inst_64_, v_inst_65_, v_S_66_, v_hS_67_);
lean_dec(v_inst_65_);
lean_dec_ref(v_inst_64_);
return v_res_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_smulCardAddSubgroup(lean_object* v_G_69_, lean_object* v_inst_70_, lean_object* v_inst_71_, lean_object* v_S_72_, lean_object* v_hS_73_){
_start:
{
lean_object* v___x_74_; 
v___x_74_ = lean_box(0);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_smulCardAddSubgroup___boxed(lean_object* v_G_75_, lean_object* v_inst_76_, lean_object* v_inst_77_, lean_object* v_S_78_, lean_object* v_hS_79_){
_start:
{
lean_object* v_res_80_; 
v_res_80_ = lp_mathlib_smulCardAddSubgroup(v_G_75_, v_inst_76_, v_inst_77_, v_S_78_, v_hS_79_);
lean_dec(v_inst_77_);
lean_dec_ref(v_inst_76_);
return v_res_80_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_CharP_Two(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Commute_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Finite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Finite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_TransferInstance(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_NatInt(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Action(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Abs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_ModEq(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Dynamics_PeriodicPts_Lemmas(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Index(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_NumberTheory_Divisors(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Interval_Set_Infinite(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_OrderOfElement(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_CharP_Two(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Commute_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Finite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Finite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_TransferInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_NatInt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Action(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Abs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_ModEq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Dynamics_PeriodicPts_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Index(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_NumberTheory_Divisors(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Interval_Set_Infinite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_GroupTheory_OrderOfElement(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_CharP_Two(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Commute_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Finite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Finite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_TransferInstance(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_NatInt(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Action(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Abs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_ModEq(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Dynamics_PeriodicPts_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Index(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_NumberTheory_Divisors(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Set_Infinite(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_OrderOfElement(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_CharP_Two(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Commute_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Finite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Finite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_TransferInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_NatInt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Action(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Ring_Abs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_ModEq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Dynamics_PeriodicPts_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Index(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_NumberTheory_Divisors(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Set_Infinite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_OrderOfElement(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_GroupTheory_OrderOfElement(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_GroupTheory_OrderOfElement(builtin);
}
#ifdef __cplusplus
}
#endif
