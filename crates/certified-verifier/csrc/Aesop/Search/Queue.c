// Lean compiler output
// Module: Aesop.Search.Queue
// Imports: public import Init public meta import Init public import Aesop.Search.Queue.Class public import Batteries.Data.BinomialHeap.Basic
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
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_st_ref_get(lean_object*);
extern lean_object* lp_aesop_Aesop_treeImpl;
double lp_aesop_Aesop_Goal_priority(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
uint8_t lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___redArg(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t lp_aesop_Aesop_Percent_instOrd___lam__0(double, double);
uint8_t lean_float_decLt(double, double);
double lean_float_sub(double, double);
double l_Float_ofScientific(lean_object*, uint8_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
size_t lean_usize_of_nat(lean_object*);
lean_object* l_Array_reverse___redArg(lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_instMonadBaseIO___aux__5___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_array_pop(lean_object*);
static lean_once_cell_t lp_aesop_Aesop_BestFirstQueue_ActiveGoal_le___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop_Aesop_BestFirstQueue_ActiveGoal_le___closed__0;
LEAN_EXPORT uint8_t lp_aesop_Aesop_BestFirstQueue_ActiveGoal_le(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BestFirstQueue_ActiveGoal_le___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BestFirstQueue_ActiveGoal_ofGoalRef(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BestFirstQueue_ActiveGoal_ofGoalRef___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BestFirstQueue_init;
LEAN_EXPORT lean_object* lp_aesop_Batteries_BinomialHeap_Imp_Heap_merge___at___00Aesop_BestFirstQueue_addGoals_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_BestFirstQueue_addGoals_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_BestFirstQueue_addGoals_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BestFirstQueue_addGoals(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BestFirstQueue_addGoals___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_BestFirstQueue_popGoal___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_BestFirstQueue_ActiveGoal_le___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_BestFirstQueue_popGoal___closed__0 = (const lean_object*)&lp_aesop_Aesop_BestFirstQueue_popGoal___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_BestFirstQueue_popGoal(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instQueueBestFirstQueue___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instQueueBestFirstQueue___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_instQueueBestFirstQueue___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instQueueBestFirstQueue___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instQueueBestFirstQueue___closed__0 = (const lean_object*)&lp_aesop_Aesop_instQueueBestFirstQueue___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_instQueueBestFirstQueue___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadBaseIO___aux__5___boxed, .m_arity = 3, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_aesop_Aesop_instQueueBestFirstQueue___closed__1 = (const lean_object*)&lp_aesop_Aesop_instQueueBestFirstQueue___closed__1_value;
static const lean_closure_object lp_aesop_Aesop_instQueueBestFirstQueue___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_BestFirstQueue_addGoals___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instQueueBestFirstQueue___closed__2 = (const lean_object*)&lp_aesop_Aesop_instQueueBestFirstQueue___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_instQueueBestFirstQueue___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instQueueBestFirstQueue___closed__1_value),((lean_object*)&lp_aesop_Aesop_instQueueBestFirstQueue___closed__2_value),((lean_object*)&lp_aesop_Aesop_instQueueBestFirstQueue___closed__0_value)}};
static const lean_object* lp_aesop_Aesop_instQueueBestFirstQueue___closed__3 = (const lean_object*)&lp_aesop_Aesop_instQueueBestFirstQueue___closed__3_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instQueueBestFirstQueue = (const lean_object*)&lp_aesop_Aesop_instQueueBestFirstQueue___closed__3_value;
static const lean_array_object lp_aesop_Aesop_LIFOQueue_init___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_LIFOQueue_init___closed__0 = (const lean_object*)&lp_aesop_Aesop_LIFOQueue_init___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_LIFOQueue_init = (const lean_object*)&lp_aesop_Aesop_LIFOQueue_init___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_LIFOQueue_addGoals(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_LIFOQueue_popGoal(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_LIFOQueue_instQueue___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_LIFOQueue_instQueue___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_LIFOQueue_instQueue___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_LIFOQueue_instQueue___lam__1___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_LIFOQueue_instQueue___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_LIFOQueue_instQueue___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_LIFOQueue_instQueue___closed__0 = (const lean_object*)&lp_aesop_Aesop_LIFOQueue_instQueue___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_LIFOQueue_instQueue___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_LIFOQueue_instQueue___lam__1___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_LIFOQueue_instQueue___closed__1 = (const lean_object*)&lp_aesop_Aesop_LIFOQueue_instQueue___closed__1_value;
static const lean_closure_object lp_aesop_Aesop_LIFOQueue_instQueue___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadBaseIO___aux__5___boxed, .m_arity = 3, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_LIFOQueue_init___closed__0_value)} };
static const lean_object* lp_aesop_Aesop_LIFOQueue_instQueue___closed__2 = (const lean_object*)&lp_aesop_Aesop_LIFOQueue_instQueue___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_LIFOQueue_instQueue___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_LIFOQueue_instQueue___closed__2_value),((lean_object*)&lp_aesop_Aesop_LIFOQueue_instQueue___closed__0_value),((lean_object*)&lp_aesop_Aesop_LIFOQueue_instQueue___closed__1_value)}};
static const lean_object* lp_aesop_Aesop_LIFOQueue_instQueue___closed__3 = (const lean_object*)&lp_aesop_Aesop_LIFOQueue_instQueue___closed__3_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_LIFOQueue_instQueue = (const lean_object*)&lp_aesop_Aesop_LIFOQueue_instQueue___closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_FIFOQueue_init___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_LIFOQueue_init___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop_Aesop_FIFOQueue_init___closed__0 = (const lean_object*)&lp_aesop_Aesop_FIFOQueue_init___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_FIFOQueue_init = (const lean_object*)&lp_aesop_Aesop_FIFOQueue_init___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_FIFOQueue_addGoals(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_FIFOQueue_addGoals___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_FIFOQueue_popGoal(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_FIFOQueue_instQueue___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_FIFOQueue_instQueue___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_FIFOQueue_instQueue___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_FIFOQueue_instQueue___lam__1___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_FIFOQueue_instQueue___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_FIFOQueue_instQueue___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_FIFOQueue_instQueue___closed__0 = (const lean_object*)&lp_aesop_Aesop_FIFOQueue_instQueue___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_FIFOQueue_instQueue___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_FIFOQueue_instQueue___lam__1___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_FIFOQueue_instQueue___closed__1 = (const lean_object*)&lp_aesop_Aesop_FIFOQueue_instQueue___closed__1_value;
static const lean_closure_object lp_aesop_Aesop_FIFOQueue_instQueue___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadBaseIO___aux__5___boxed, .m_arity = 3, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_FIFOQueue_init___closed__0_value)} };
static const lean_object* lp_aesop_Aesop_FIFOQueue_instQueue___closed__2 = (const lean_object*)&lp_aesop_Aesop_FIFOQueue_instQueue___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_FIFOQueue_instQueue___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_FIFOQueue_instQueue___closed__2_value),((lean_object*)&lp_aesop_Aesop_FIFOQueue_instQueue___closed__0_value),((lean_object*)&lp_aesop_Aesop_FIFOQueue_instQueue___closed__1_value)}};
static const lean_object* lp_aesop_Aesop_FIFOQueue_instQueue___closed__3 = (const lean_object*)&lp_aesop_Aesop_FIFOQueue_instQueue___closed__3_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_FIFOQueue_instQueue = (const lean_object*)&lp_aesop_Aesop_FIFOQueue_instQueue___closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_Options_queue___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_instQueueBestFirstQueue___closed__3_value)}};
static const lean_object* lp_aesop_Aesop_Options_queue___closed__0 = (const lean_object*)&lp_aesop_Aesop_Options_queue___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Options_queue___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_LIFOQueue_instQueue___closed__3_value)}};
static const lean_object* lp_aesop_Aesop_Options_queue___closed__1 = (const lean_object*)&lp_aesop_Aesop_Options_queue___closed__1_value;
static const lean_ctor_object lp_aesop_Aesop_Options_queue___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_FIFOQueue_instQueue___closed__3_value)}};
static const lean_object* lp_aesop_Aesop_Options_queue___closed__2 = (const lean_object*)&lp_aesop_Aesop_Options_queue___closed__2_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_queue(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_queue___boxed(lean_object*);
static double _init_lp_aesop_Aesop_BestFirstQueue_ActiveGoal_le___closed__0(void){
_start:
{
lean_object* v___x_1_; uint8_t v___x_2_; lean_object* v___x_3_; double v___x_4_; 
v___x_1_ = lean_unsigned_to_nat(5u);
v___x_2_ = 1;
v___x_3_ = lean_unsigned_to_nat(1u);
v___x_4_ = l_Float_ofScientific(v___x_3_, v___x_2_, v___x_1_);
return v___x_4_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_BestFirstQueue_ActiveGoal_le(lean_object* v_g_5_, lean_object* v_h_6_){
_start:
{
double v_priority_7_; lean_object* v_lastExpandedInIteration_8_; lean_object* v_addedInIteration_9_; double v_priority_10_; lean_object* v_lastExpandedInIteration_11_; lean_object* v_addedInIteration_12_; uint8_t v___y_14_; uint8_t v___x_18_; 
v_priority_7_ = lean_ctor_get_float(v_h_6_, sizeof(void*)*3);
v_lastExpandedInIteration_8_ = lean_ctor_get(v_h_6_, 1);
v_addedInIteration_9_ = lean_ctor_get(v_h_6_, 2);
v_priority_10_ = lean_ctor_get_float(v_g_5_, sizeof(void*)*3);
v_lastExpandedInIteration_11_ = lean_ctor_get(v_g_5_, 1);
v_addedInIteration_12_ = lean_ctor_get(v_g_5_, 2);
v___x_18_ = lp_aesop_Aesop_Percent_instOrd___lam__0(v_priority_7_, v_priority_10_);
if (v___x_18_ == 0)
{
uint8_t v___x_19_; 
v___x_19_ = 1;
return v___x_19_;
}
else
{
uint8_t v___x_20_; 
v___x_20_ = lean_float_decLt(v_priority_7_, v_priority_10_);
if (v___x_20_ == 0)
{
double v___x_21_; double v___x_22_; uint8_t v___x_23_; 
v___x_21_ = lean_float_sub(v_priority_7_, v_priority_10_);
v___x_22_ = lean_float_once(&lp_aesop_Aesop_BestFirstQueue_ActiveGoal_le___closed__0, &lp_aesop_Aesop_BestFirstQueue_ActiveGoal_le___closed__0_once, _init_lp_aesop_Aesop_BestFirstQueue_ActiveGoal_le___closed__0);
v___x_23_ = lean_float_decLt(v___x_21_, v___x_22_);
v___y_14_ = v___x_23_;
goto v___jp_13_;
}
else
{
double v___x_24_; lean_object* v___x_25_; lean_object* v___x_26_; double v___x_27_; uint8_t v___x_28_; 
v___x_24_ = lean_float_sub(v_priority_10_, v_priority_7_);
v___x_25_ = lean_unsigned_to_nat(1u);
v___x_26_ = lean_unsigned_to_nat(5u);
v___x_27_ = l_Float_ofScientific(v___x_25_, v___x_20_, v___x_26_);
v___x_28_ = lean_float_decLt(v___x_24_, v___x_27_);
v___y_14_ = v___x_28_;
goto v___jp_13_;
}
}
v___jp_13_:
{
if (v___y_14_ == 0)
{
return v___y_14_;
}
else
{
uint8_t v___x_15_; 
v___x_15_ = lean_nat_dec_le(v_lastExpandedInIteration_11_, v_lastExpandedInIteration_8_);
if (v___x_15_ == 0)
{
uint8_t v___x_16_; 
v___x_16_ = lean_nat_dec_eq(v_lastExpandedInIteration_11_, v_lastExpandedInIteration_8_);
if (v___x_16_ == 0)
{
return v___x_16_;
}
else
{
uint8_t v___x_17_; 
v___x_17_ = lean_nat_dec_le(v_addedInIteration_12_, v_addedInIteration_9_);
return v___x_17_;
}
}
else
{
return v___x_15_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BestFirstQueue_ActiveGoal_le___boxed(lean_object* v_g_29_, lean_object* v_h_30_){
_start:
{
uint8_t v_res_31_; lean_object* v_r_32_; 
v_res_31_ = lp_aesop_Aesop_BestFirstQueue_ActiveGoal_le(v_g_29_, v_h_30_);
lean_dec_ref(v_h_30_);
lean_dec_ref(v_g_29_);
v_r_32_ = lean_box(v_res_31_);
return v_r_32_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BestFirstQueue_ActiveGoal_ofGoalRef(lean_object* v_gref_33_){
_start:
{
lean_object* v___x_35_; lean_object* v___x_36_; lean_object* v_elimGoal_37_; double v___x_38_; lean_object* v___x_39_; lean_object* v_addedInIteration_40_; lean_object* v_lastExpandedInIteration_41_; lean_object* v___x_42_; 
v___x_35_ = lean_st_ref_get(v_gref_33_);
v___x_36_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_37_ = lean_ctor_get(v___x_36_, 1);
lean_inc(v___x_35_);
v___x_38_ = lp_aesop_Aesop_Goal_priority(v___x_35_);
lean_inc_ref(v_elimGoal_37_);
v___x_39_ = lean_apply_1(v_elimGoal_37_, v___x_35_);
v_addedInIteration_40_ = lean_ctor_get(v___x_39_, 10);
lean_inc(v_addedInIteration_40_);
v_lastExpandedInIteration_41_ = lean_ctor_get(v___x_39_, 11);
lean_inc(v_lastExpandedInIteration_41_);
lean_dec_ref(v___x_39_);
v___x_42_ = lean_alloc_ctor(0, 3, 8);
lean_ctor_set(v___x_42_, 0, v_gref_33_);
lean_ctor_set(v___x_42_, 1, v_lastExpandedInIteration_41_);
lean_ctor_set(v___x_42_, 2, v_addedInIteration_40_);
lean_ctor_set_float(v___x_42_, sizeof(void*)*3, v___x_38_);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BestFirstQueue_ActiveGoal_ofGoalRef___boxed(lean_object* v_gref_43_, lean_object* v_a_44_){
_start:
{
lean_object* v_res_45_; 
v_res_45_ = lp_aesop_Aesop_BestFirstQueue_ActiveGoal_ofGoalRef(v_gref_43_);
return v_res_45_;
}
}
static lean_object* _init_lp_aesop_Aesop_BestFirstQueue_init(void){
_start:
{
lean_object* v___x_46_; 
v___x_46_ = lean_box(0);
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Batteries_BinomialHeap_Imp_Heap_merge___at___00Aesop_BestFirstQueue_addGoals_spec__0(lean_object* v_x_47_, lean_object* v_x_48_){
_start:
{
if (lean_obj_tag(v_x_47_) == 0)
{
return v_x_48_;
}
else
{
if (lean_obj_tag(v_x_48_) == 0)
{
return v_x_47_;
}
else
{
lean_object* v_rank_49_; lean_object* v_val_50_; lean_object* v_node_51_; lean_object* v_next_52_; lean_object* v_rank_53_; lean_object* v_val_54_; lean_object* v_node_55_; lean_object* v_next_56_; lean_object* v_fst_58_; lean_object* v_snd_59_; uint8_t v___x_73_; 
v_rank_49_ = lean_ctor_get(v_x_47_, 0);
v_val_50_ = lean_ctor_get(v_x_47_, 1);
v_node_51_ = lean_ctor_get(v_x_47_, 2);
v_next_52_ = lean_ctor_get(v_x_47_, 3);
v_rank_53_ = lean_ctor_get(v_x_48_, 0);
v_val_54_ = lean_ctor_get(v_x_48_, 1);
v_node_55_ = lean_ctor_get(v_x_48_, 2);
v_next_56_ = lean_ctor_get(v_x_48_, 3);
v___x_73_ = lean_nat_dec_lt(v_rank_49_, v_rank_53_);
if (v___x_73_ == 0)
{
lean_object* v___x_75_; uint8_t v_isShared_76_; uint8_t v_isSharedCheck_85_; 
lean_inc(v_next_56_);
lean_inc(v_node_55_);
lean_inc(v_val_54_);
lean_inc(v_rank_53_);
v_isSharedCheck_85_ = !lean_is_exclusive(v_x_48_);
if (v_isSharedCheck_85_ == 0)
{
lean_object* v_unused_86_; lean_object* v_unused_87_; lean_object* v_unused_88_; lean_object* v_unused_89_; 
v_unused_86_ = lean_ctor_get(v_x_48_, 3);
lean_dec(v_unused_86_);
v_unused_87_ = lean_ctor_get(v_x_48_, 2);
lean_dec(v_unused_87_);
v_unused_88_ = lean_ctor_get(v_x_48_, 1);
lean_dec(v_unused_88_);
v_unused_89_ = lean_ctor_get(v_x_48_, 0);
lean_dec(v_unused_89_);
v___x_75_ = v_x_48_;
v_isShared_76_ = v_isSharedCheck_85_;
goto v_resetjp_74_;
}
else
{
lean_dec(v_x_48_);
v___x_75_ = lean_box(0);
v_isShared_76_ = v_isSharedCheck_85_;
goto v_resetjp_74_;
}
v_resetjp_74_:
{
uint8_t v___x_77_; 
v___x_77_ = lean_nat_dec_lt(v_rank_53_, v_rank_49_);
if (v___x_77_ == 0)
{
uint8_t v___x_78_; 
lean_inc(v_next_52_);
lean_inc(v_node_51_);
lean_inc(v_val_50_);
lean_inc(v_rank_49_);
lean_del_object(v___x_75_);
lean_dec(v_rank_53_);
lean_dec_ref_known(v_x_47_, 4);
v___x_78_ = lp_aesop_Aesop_BestFirstQueue_ActiveGoal_le(v_val_50_, v_val_54_);
if (v___x_78_ == 0)
{
lean_object* v___x_79_; 
v___x_79_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_79_, 0, v_val_50_);
lean_ctor_set(v___x_79_, 1, v_node_51_);
lean_ctor_set(v___x_79_, 2, v_node_55_);
v_fst_58_ = v_val_54_;
v_snd_59_ = v___x_79_;
goto v___jp_57_;
}
else
{
lean_object* v___x_80_; 
v___x_80_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_80_, 0, v_val_54_);
lean_ctor_set(v___x_80_, 1, v_node_55_);
lean_ctor_set(v___x_80_, 2, v_node_51_);
v_fst_58_ = v_val_50_;
v_snd_59_ = v___x_80_;
goto v___jp_57_;
}
}
else
{
lean_object* v___x_81_; lean_object* v___x_83_; 
v___x_81_ = lp_aesop_Batteries_BinomialHeap_Imp_Heap_merge___at___00Aesop_BestFirstQueue_addGoals_spec__0(v_x_47_, v_next_56_);
if (v_isShared_76_ == 0)
{
lean_ctor_set(v___x_75_, 3, v___x_81_);
v___x_83_ = v___x_75_;
goto v_reusejp_82_;
}
else
{
lean_object* v_reuseFailAlloc_84_; 
v_reuseFailAlloc_84_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v_reuseFailAlloc_84_, 0, v_rank_53_);
lean_ctor_set(v_reuseFailAlloc_84_, 1, v_val_54_);
lean_ctor_set(v_reuseFailAlloc_84_, 2, v_node_55_);
lean_ctor_set(v_reuseFailAlloc_84_, 3, v___x_81_);
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
else
{
lean_object* v___x_91_; uint8_t v_isShared_92_; uint8_t v_isSharedCheck_97_; 
lean_inc(v_next_52_);
lean_inc(v_node_51_);
lean_inc(v_val_50_);
lean_inc(v_rank_49_);
v_isSharedCheck_97_ = !lean_is_exclusive(v_x_47_);
if (v_isSharedCheck_97_ == 0)
{
lean_object* v_unused_98_; lean_object* v_unused_99_; lean_object* v_unused_100_; lean_object* v_unused_101_; 
v_unused_98_ = lean_ctor_get(v_x_47_, 3);
lean_dec(v_unused_98_);
v_unused_99_ = lean_ctor_get(v_x_47_, 2);
lean_dec(v_unused_99_);
v_unused_100_ = lean_ctor_get(v_x_47_, 1);
lean_dec(v_unused_100_);
v_unused_101_ = lean_ctor_get(v_x_47_, 0);
lean_dec(v_unused_101_);
v___x_91_ = v_x_47_;
v_isShared_92_ = v_isSharedCheck_97_;
goto v_resetjp_90_;
}
else
{
lean_dec(v_x_47_);
v___x_91_ = lean_box(0);
v_isShared_92_ = v_isSharedCheck_97_;
goto v_resetjp_90_;
}
v_resetjp_90_:
{
lean_object* v___x_93_; lean_object* v___x_95_; 
v___x_93_ = lp_aesop_Batteries_BinomialHeap_Imp_Heap_merge___at___00Aesop_BestFirstQueue_addGoals_spec__0(v_next_52_, v_x_48_);
if (v_isShared_92_ == 0)
{
lean_ctor_set(v___x_91_, 3, v___x_93_);
v___x_95_ = v___x_91_;
goto v_reusejp_94_;
}
else
{
lean_object* v_reuseFailAlloc_96_; 
v_reuseFailAlloc_96_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v_reuseFailAlloc_96_, 0, v_rank_49_);
lean_ctor_set(v_reuseFailAlloc_96_, 1, v_val_50_);
lean_ctor_set(v_reuseFailAlloc_96_, 2, v_node_51_);
lean_ctor_set(v_reuseFailAlloc_96_, 3, v___x_93_);
v___x_95_ = v_reuseFailAlloc_96_;
goto v_reusejp_94_;
}
v_reusejp_94_:
{
return v___x_95_;
}
}
}
v___jp_57_:
{
lean_object* v___x_60_; lean_object* v_r_61_; uint8_t v___x_62_; 
v___x_60_ = lean_unsigned_to_nat(1u);
v_r_61_ = lean_nat_add(v_rank_49_, v___x_60_);
lean_dec(v_rank_49_);
v___x_62_ = lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___redArg(v_next_52_, v_r_61_);
if (v___x_62_ == 0)
{
uint8_t v___x_63_; 
v___x_63_ = lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___redArg(v_next_56_, v_r_61_);
if (v___x_63_ == 0)
{
lean_object* v___x_64_; lean_object* v___x_65_; 
v___x_64_ = lp_aesop_Batteries_BinomialHeap_Imp_Heap_merge___at___00Aesop_BestFirstQueue_addGoals_spec__0(v_next_52_, v_next_56_);
v___x_65_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v___x_65_, 0, v_r_61_);
lean_ctor_set(v___x_65_, 1, v_fst_58_);
lean_ctor_set(v___x_65_, 2, v_snd_59_);
lean_ctor_set(v___x_65_, 3, v___x_64_);
return v___x_65_;
}
else
{
lean_object* v___x_66_; 
v___x_66_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v___x_66_, 0, v_r_61_);
lean_ctor_set(v___x_66_, 1, v_fst_58_);
lean_ctor_set(v___x_66_, 2, v_snd_59_);
lean_ctor_set(v___x_66_, 3, v_next_56_);
v_x_47_ = v_next_52_;
v_x_48_ = v___x_66_;
goto _start;
}
}
else
{
uint8_t v___x_68_; 
v___x_68_ = lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___redArg(v_next_56_, v_r_61_);
if (v___x_68_ == 0)
{
lean_object* v___x_69_; 
v___x_69_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v___x_69_, 0, v_r_61_);
lean_ctor_set(v___x_69_, 1, v_fst_58_);
lean_ctor_set(v___x_69_, 2, v_snd_59_);
lean_ctor_set(v___x_69_, 3, v_next_52_);
v_x_47_ = v___x_69_;
v_x_48_ = v_next_56_;
goto _start;
}
else
{
lean_object* v___x_71_; lean_object* v___x_72_; 
v___x_71_ = lp_aesop_Batteries_BinomialHeap_Imp_Heap_merge___at___00Aesop_BestFirstQueue_addGoals_spec__0(v_next_52_, v_next_56_);
v___x_72_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v___x_72_, 0, v_r_61_);
lean_ctor_set(v___x_72_, 1, v_fst_58_);
lean_ctor_set(v___x_72_, 2, v_snd_59_);
lean_ctor_set(v___x_72_, 3, v___x_71_);
return v___x_72_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_BestFirstQueue_addGoals_spec__1(lean_object* v_as_102_, size_t v_i_103_, size_t v_stop_104_, lean_object* v_b_105_){
_start:
{
uint8_t v___x_107_; 
v___x_107_ = lean_usize_dec_eq(v_i_103_, v_stop_104_);
if (v___x_107_ == 0)
{
lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; size_t v___x_115_; size_t v___x_116_; 
v___x_108_ = lean_array_uget_borrowed(v_as_102_, v_i_103_);
lean_inc(v___x_108_);
v___x_109_ = lp_aesop_Aesop_BestFirstQueue_ActiveGoal_ofGoalRef(v___x_108_);
v___x_110_ = lean_unsigned_to_nat(0u);
v___x_111_ = lean_box(0);
v___x_112_ = lean_box(0);
v___x_113_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v___x_113_, 0, v___x_110_);
lean_ctor_set(v___x_113_, 1, v___x_109_);
lean_ctor_set(v___x_113_, 2, v___x_111_);
lean_ctor_set(v___x_113_, 3, v___x_112_);
v___x_114_ = lp_aesop_Batteries_BinomialHeap_Imp_Heap_merge___at___00Aesop_BestFirstQueue_addGoals_spec__0(v___x_113_, v_b_105_);
v___x_115_ = ((size_t)1ULL);
v___x_116_ = lean_usize_add(v_i_103_, v___x_115_);
v_i_103_ = v___x_116_;
v_b_105_ = v___x_114_;
goto _start;
}
else
{
return v_b_105_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_BestFirstQueue_addGoals_spec__1___boxed(lean_object* v_as_118_, lean_object* v_i_119_, lean_object* v_stop_120_, lean_object* v_b_121_, lean_object* v___y_122_){
_start:
{
size_t v_i_boxed_123_; size_t v_stop_boxed_124_; lean_object* v_res_125_; 
v_i_boxed_123_ = lean_unbox_usize(v_i_119_);
lean_dec(v_i_119_);
v_stop_boxed_124_ = lean_unbox_usize(v_stop_120_);
lean_dec(v_stop_120_);
v_res_125_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_BestFirstQueue_addGoals_spec__1(v_as_118_, v_i_boxed_123_, v_stop_boxed_124_, v_b_121_);
lean_dec_ref(v_as_118_);
return v_res_125_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BestFirstQueue_addGoals(lean_object* v_q_126_, lean_object* v_grefs_127_){
_start:
{
lean_object* v___x_129_; lean_object* v___x_130_; uint8_t v___x_131_; 
v___x_129_ = lean_unsigned_to_nat(0u);
v___x_130_ = lean_array_get_size(v_grefs_127_);
v___x_131_ = lean_nat_dec_lt(v___x_129_, v___x_130_);
if (v___x_131_ == 0)
{
return v_q_126_;
}
else
{
uint8_t v___x_132_; 
v___x_132_ = lean_nat_dec_le(v___x_130_, v___x_130_);
if (v___x_132_ == 0)
{
if (v___x_131_ == 0)
{
return v_q_126_;
}
else
{
size_t v___x_133_; size_t v___x_134_; lean_object* v___x_135_; 
v___x_133_ = ((size_t)0ULL);
v___x_134_ = lean_usize_of_nat(v___x_130_);
v___x_135_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_BestFirstQueue_addGoals_spec__1(v_grefs_127_, v___x_133_, v___x_134_, v_q_126_);
return v___x_135_;
}
}
else
{
size_t v___x_136_; size_t v___x_137_; lean_object* v___x_138_; 
v___x_136_ = ((size_t)0ULL);
v___x_137_ = lean_usize_of_nat(v___x_130_);
v___x_138_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_BestFirstQueue_addGoals_spec__1(v_grefs_127_, v___x_136_, v___x_137_, v_q_126_);
return v___x_138_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BestFirstQueue_addGoals___boxed(lean_object* v_q_139_, lean_object* v_grefs_140_, lean_object* v_a_141_){
_start:
{
lean_object* v_res_142_; 
v_res_142_ = lp_aesop_Aesop_BestFirstQueue_addGoals(v_q_139_, v_grefs_140_);
lean_dec_ref(v_grefs_140_);
return v_res_142_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BestFirstQueue_popGoal(lean_object* v_q_144_){
_start:
{
lean_object* v___x_145_; lean_object* v___x_146_; 
v___x_145_ = ((lean_object*)(lp_aesop_Aesop_BestFirstQueue_popGoal___closed__0));
lean_inc(v_q_144_);
v___x_146_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg(v___x_145_, v_q_144_);
if (lean_obj_tag(v___x_146_) == 0)
{
lean_object* v___x_147_; lean_object* v___x_148_; 
v___x_147_ = lean_box(0);
v___x_148_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_148_, 0, v___x_147_);
lean_ctor_set(v___x_148_, 1, v_q_144_);
return v___x_148_;
}
else
{
lean_object* v_val_149_; lean_object* v___x_151_; uint8_t v_isShared_152_; uint8_t v_isSharedCheck_166_; 
lean_dec(v_q_144_);
v_val_149_ = lean_ctor_get(v___x_146_, 0);
v_isSharedCheck_166_ = !lean_is_exclusive(v___x_146_);
if (v_isSharedCheck_166_ == 0)
{
v___x_151_ = v___x_146_;
v_isShared_152_ = v_isSharedCheck_166_;
goto v_resetjp_150_;
}
else
{
lean_inc(v_val_149_);
lean_dec(v___x_146_);
v___x_151_ = lean_box(0);
v_isShared_152_ = v_isSharedCheck_166_;
goto v_resetjp_150_;
}
v_resetjp_150_:
{
lean_object* v_fst_153_; lean_object* v_snd_154_; lean_object* v___x_156_; uint8_t v_isShared_157_; uint8_t v_isSharedCheck_165_; 
v_fst_153_ = lean_ctor_get(v_val_149_, 0);
v_snd_154_ = lean_ctor_get(v_val_149_, 1);
v_isSharedCheck_165_ = !lean_is_exclusive(v_val_149_);
if (v_isSharedCheck_165_ == 0)
{
v___x_156_ = v_val_149_;
v_isShared_157_ = v_isSharedCheck_165_;
goto v_resetjp_155_;
}
else
{
lean_inc(v_snd_154_);
lean_inc(v_fst_153_);
lean_dec(v_val_149_);
v___x_156_ = lean_box(0);
v_isShared_157_ = v_isSharedCheck_165_;
goto v_resetjp_155_;
}
v_resetjp_155_:
{
lean_object* v_goal_158_; lean_object* v___x_160_; 
v_goal_158_ = lean_ctor_get(v_fst_153_, 0);
lean_inc(v_goal_158_);
lean_dec(v_fst_153_);
if (v_isShared_152_ == 0)
{
lean_ctor_set(v___x_151_, 0, v_goal_158_);
v___x_160_ = v___x_151_;
goto v_reusejp_159_;
}
else
{
lean_object* v_reuseFailAlloc_164_; 
v_reuseFailAlloc_164_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_164_, 0, v_goal_158_);
v___x_160_ = v_reuseFailAlloc_164_;
goto v_reusejp_159_;
}
v_reusejp_159_:
{
lean_object* v___x_162_; 
if (v_isShared_157_ == 0)
{
lean_ctor_set(v___x_156_, 0, v___x_160_);
v___x_162_ = v___x_156_;
goto v_reusejp_161_;
}
else
{
lean_object* v_reuseFailAlloc_163_; 
v_reuseFailAlloc_163_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_163_, 0, v___x_160_);
lean_ctor_set(v_reuseFailAlloc_163_, 1, v_snd_154_);
v___x_162_ = v_reuseFailAlloc_163_;
goto v_reusejp_161_;
}
v_reusejp_161_:
{
return v___x_162_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instQueueBestFirstQueue___lam__0(lean_object* v_q_167_){
_start:
{
lean_object* v___x_169_; 
v___x_169_ = lp_aesop_Aesop_BestFirstQueue_popGoal(v_q_167_);
return v___x_169_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instQueueBestFirstQueue___lam__0___boxed(lean_object* v_q_170_, lean_object* v___y_171_){
_start:
{
lean_object* v_res_172_; 
v_res_172_ = lp_aesop_Aesop_instQueueBestFirstQueue___lam__0(v_q_170_);
return v_res_172_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_LIFOQueue_addGoals(lean_object* v_q_185_, lean_object* v_grefs_186_){
_start:
{
lean_object* v___x_187_; lean_object* v___x_188_; 
v___x_187_ = l_Array_reverse___redArg(v_grefs_186_);
v___x_188_ = l_Array_append___redArg(v_q_185_, v___x_187_);
lean_dec_ref(v___x_187_);
return v___x_188_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_LIFOQueue_popGoal(lean_object* v_q_189_){
_start:
{
lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; uint8_t v___x_193_; 
v___x_190_ = lean_array_get_size(v_q_189_);
v___x_191_ = lean_unsigned_to_nat(1u);
v___x_192_ = lean_nat_sub(v___x_190_, v___x_191_);
v___x_193_ = lean_nat_dec_lt(v___x_192_, v___x_190_);
if (v___x_193_ == 0)
{
lean_object* v___x_194_; lean_object* v___x_195_; 
lean_dec(v___x_192_);
v___x_194_ = lean_box(0);
v___x_195_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_195_, 0, v___x_194_);
lean_ctor_set(v___x_195_, 1, v_q_189_);
return v___x_195_;
}
else
{
lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; 
v___x_196_ = lean_array_fget_borrowed(v_q_189_, v___x_192_);
lean_dec(v___x_192_);
lean_inc(v___x_196_);
v___x_197_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_197_, 0, v___x_196_);
v___x_198_ = lean_array_pop(v_q_189_);
v___x_199_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_199_, 0, v___x_197_);
lean_ctor_set(v___x_199_, 1, v___x_198_);
return v___x_199_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_LIFOQueue_instQueue___lam__0(lean_object* v_q_200_, lean_object* v_grefs_201_){
_start:
{
lean_object* v___x_203_; 
v___x_203_ = lp_aesop_Aesop_LIFOQueue_addGoals(v_q_200_, v_grefs_201_);
return v___x_203_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_LIFOQueue_instQueue___lam__0___boxed(lean_object* v_q_204_, lean_object* v_grefs_205_, lean_object* v___y_206_){
_start:
{
lean_object* v_res_207_; 
v_res_207_ = lp_aesop_Aesop_LIFOQueue_instQueue___lam__0(v_q_204_, v_grefs_205_);
return v_res_207_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_LIFOQueue_instQueue___lam__1(lean_object* v_q_208_){
_start:
{
lean_object* v___x_210_; 
v___x_210_ = lp_aesop_Aesop_LIFOQueue_popGoal(v_q_208_);
return v___x_210_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_LIFOQueue_instQueue___lam__1___boxed(lean_object* v_q_211_, lean_object* v___y_212_){
_start:
{
lean_object* v_res_213_; 
v_res_213_ = lp_aesop_Aesop_LIFOQueue_instQueue___lam__1(v_q_211_);
return v_res_213_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_FIFOQueue_addGoals(lean_object* v_q_227_, lean_object* v_grefs_228_){
_start:
{
lean_object* v_goals_229_; lean_object* v_pos_230_; lean_object* v___x_232_; uint8_t v_isShared_233_; uint8_t v_isSharedCheck_238_; 
v_goals_229_ = lean_ctor_get(v_q_227_, 0);
v_pos_230_ = lean_ctor_get(v_q_227_, 1);
v_isSharedCheck_238_ = !lean_is_exclusive(v_q_227_);
if (v_isSharedCheck_238_ == 0)
{
v___x_232_ = v_q_227_;
v_isShared_233_ = v_isSharedCheck_238_;
goto v_resetjp_231_;
}
else
{
lean_inc(v_pos_230_);
lean_inc(v_goals_229_);
lean_dec(v_q_227_);
v___x_232_ = lean_box(0);
v_isShared_233_ = v_isSharedCheck_238_;
goto v_resetjp_231_;
}
v_resetjp_231_:
{
lean_object* v___x_234_; lean_object* v___x_236_; 
v___x_234_ = l_Array_append___redArg(v_goals_229_, v_grefs_228_);
if (v_isShared_233_ == 0)
{
lean_ctor_set(v___x_232_, 0, v___x_234_);
v___x_236_ = v___x_232_;
goto v_reusejp_235_;
}
else
{
lean_object* v_reuseFailAlloc_237_; 
v_reuseFailAlloc_237_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_237_, 0, v___x_234_);
lean_ctor_set(v_reuseFailAlloc_237_, 1, v_pos_230_);
v___x_236_ = v_reuseFailAlloc_237_;
goto v_reusejp_235_;
}
v_reusejp_235_:
{
return v___x_236_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_FIFOQueue_addGoals___boxed(lean_object* v_q_239_, lean_object* v_grefs_240_){
_start:
{
lean_object* v_res_241_; 
v_res_241_ = lp_aesop_Aesop_FIFOQueue_addGoals(v_q_239_, v_grefs_240_);
lean_dec_ref(v_grefs_240_);
return v_res_241_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_FIFOQueue_popGoal(lean_object* v_q_242_){
_start:
{
lean_object* v_goals_243_; lean_object* v_pos_244_; lean_object* v___x_245_; uint8_t v___x_246_; 
v_goals_243_ = lean_ctor_get(v_q_242_, 0);
v_pos_244_ = lean_ctor_get(v_q_242_, 1);
v___x_245_ = lean_array_get_size(v_goals_243_);
v___x_246_ = lean_nat_dec_lt(v_pos_244_, v___x_245_);
if (v___x_246_ == 0)
{
lean_object* v___x_247_; lean_object* v___x_248_; 
v___x_247_ = lean_box(0);
v___x_248_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_248_, 0, v___x_247_);
lean_ctor_set(v___x_248_, 1, v_q_242_);
return v___x_248_;
}
else
{
lean_object* v___x_250_; uint8_t v_isShared_251_; uint8_t v_isSharedCheck_260_; 
lean_inc(v_pos_244_);
lean_inc_ref(v_goals_243_);
v_isSharedCheck_260_ = !lean_is_exclusive(v_q_242_);
if (v_isSharedCheck_260_ == 0)
{
lean_object* v_unused_261_; lean_object* v_unused_262_; 
v_unused_261_ = lean_ctor_get(v_q_242_, 1);
lean_dec(v_unused_261_);
v_unused_262_ = lean_ctor_get(v_q_242_, 0);
lean_dec(v_unused_262_);
v___x_250_ = v_q_242_;
v_isShared_251_ = v_isSharedCheck_260_;
goto v_resetjp_249_;
}
else
{
lean_dec(v_q_242_);
v___x_250_ = lean_box(0);
v_isShared_251_ = v_isSharedCheck_260_;
goto v_resetjp_249_;
}
v_resetjp_249_:
{
lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_257_; 
v___x_252_ = lean_array_fget_borrowed(v_goals_243_, v_pos_244_);
lean_inc(v___x_252_);
v___x_253_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_253_, 0, v___x_252_);
v___x_254_ = lean_unsigned_to_nat(1u);
v___x_255_ = lean_nat_add(v_pos_244_, v___x_254_);
lean_dec(v_pos_244_);
if (v_isShared_251_ == 0)
{
lean_ctor_set(v___x_250_, 1, v___x_255_);
v___x_257_ = v___x_250_;
goto v_reusejp_256_;
}
else
{
lean_object* v_reuseFailAlloc_259_; 
v_reuseFailAlloc_259_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_259_, 0, v_goals_243_);
lean_ctor_set(v_reuseFailAlloc_259_, 1, v___x_255_);
v___x_257_ = v_reuseFailAlloc_259_;
goto v_reusejp_256_;
}
v_reusejp_256_:
{
lean_object* v___x_258_; 
v___x_258_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_258_, 0, v___x_253_);
lean_ctor_set(v___x_258_, 1, v___x_257_);
return v___x_258_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_FIFOQueue_instQueue___lam__0(lean_object* v_q_263_, lean_object* v_grefs_264_){
_start:
{
lean_object* v___x_266_; 
v___x_266_ = lp_aesop_Aesop_FIFOQueue_addGoals(v_q_263_, v_grefs_264_);
return v___x_266_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_FIFOQueue_instQueue___lam__0___boxed(lean_object* v_q_267_, lean_object* v_grefs_268_, lean_object* v___y_269_){
_start:
{
lean_object* v_res_270_; 
v_res_270_ = lp_aesop_Aesop_FIFOQueue_instQueue___lam__0(v_q_267_, v_grefs_268_);
lean_dec_ref(v_grefs_268_);
return v_res_270_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_FIFOQueue_instQueue___lam__1(lean_object* v_q_271_){
_start:
{
lean_object* v___x_273_; 
v___x_273_ = lp_aesop_Aesop_FIFOQueue_popGoal(v_q_271_);
return v___x_273_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_FIFOQueue_instQueue___lam__1___boxed(lean_object* v_q_274_, lean_object* v___y_275_){
_start:
{
lean_object* v_res_276_; 
v_res_276_ = lp_aesop_Aesop_FIFOQueue_instQueue___lam__1(v_q_274_);
return v_res_276_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_queue(lean_object* v_opts_292_){
_start:
{
uint8_t v_strategy_293_; 
v_strategy_293_ = lean_ctor_get_uint8(v_opts_292_, sizeof(void*)*6);
switch(v_strategy_293_)
{
case 0:
{
lean_object* v___x_294_; 
v___x_294_ = ((lean_object*)(lp_aesop_Aesop_Options_queue___closed__0));
return v___x_294_;
}
case 1:
{
lean_object* v___x_295_; 
v___x_295_ = ((lean_object*)(lp_aesop_Aesop_Options_queue___closed__1));
return v___x_295_;
}
default: 
{
lean_object* v___x_296_; 
v___x_296_ = ((lean_object*)(lp_aesop_Aesop_Options_queue___closed__2));
return v___x_296_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_queue___boxed(lean_object* v_opts_297_){
_start:
{
lean_object* v_res_298_; 
v_res_298_ = lp_aesop_Aesop_Options_queue(v_opts_297_);
lean_dec_ref(v_opts_297_);
return v_res_298_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Search_Queue_Class(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Data_BinomialHeap_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Search_Queue(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Search_Queue_Class(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_BinomialHeap_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_BestFirstQueue_init = _init_lp_aesop_Aesop_BestFirstQueue_init();
lean_mark_persistent(lp_aesop_Aesop_BestFirstQueue_init);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Search_Queue(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Search_Queue_Class(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_BinomialHeap_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Search_Queue(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Search_Queue_Class(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_BinomialHeap_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Search_Queue(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Search_Queue(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Search_Queue(builtin);
}
#ifdef __cplusplus
}
#endif
