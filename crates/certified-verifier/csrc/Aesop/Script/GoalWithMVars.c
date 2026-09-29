// Lean compiler output
// Module: Aesop.Script.GoalWithMVars
// Imports: public import Init public meta import Init public import Lean.Meta.Basic import Lean.Meta.CollectMVars
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
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Name_reprPrec___boxed(lean_object*, lean_object*);
lean_object* l_Lean_Name_reprPrec(lean_object*, lean_object*);
extern lean_object* l_Std_Format_defWidth;
lean_object* l_Std_Format_pretty(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Array_repr___redArg(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* l_Std_DHashMap_Internal_AssocList_foldlM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getMVarDependencies(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_instInhabitedGoalWithMVars_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedGoalWithMVars_default___closed__0;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedGoalWithMVars_default___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedGoalWithMVars_default___closed__1;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedGoalWithMVars_default___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedGoalWithMVars_default___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedGoalWithMVars_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedGoalWithMVars;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instReprGoalWithMVars___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instReprGoalWithMVars___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "{ goal := "};
static const lean_object* lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__0 = (const lean_object*)&lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__0_value;
static const lean_string_object lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = ", mvars := "};
static const lean_object* lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__1 = (const lean_object*)&lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__1_value;
static const lean_string_object lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " }"};
static const lean_object* lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__2 = (const lean_object*)&lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__2_value;
static const lean_closure_object lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__3 = (const lean_object*)&lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__3_value;
static const lean_closure_object lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__4 = (const lean_object*)&lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__4_value;
static const lean_closure_object lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__5 = (const lean_object*)&lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__5_value;
static const lean_closure_object lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__6 = (const lean_object*)&lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__6_value;
static const lean_closure_object lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__7 = (const lean_object*)&lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__7_value;
static const lean_closure_object lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__8 = (const lean_object*)&lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__8_value;
static const lean_closure_object lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__9 = (const lean_object*)&lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__9_value;
static const lean_ctor_object lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__3_value),((lean_object*)&lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__4_value)}};
static const lean_object* lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__10 = (const lean_object*)&lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__10_value;
static const lean_ctor_object lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__10_value),((lean_object*)&lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__5_value),((lean_object*)&lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__6_value),((lean_object*)&lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__7_value),((lean_object*)&lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__8_value)}};
static const lean_object* lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__11 = (const lean_object*)&lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__11_value;
static const lean_ctor_object lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__11_value),((lean_object*)&lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__9_value)}};
static const lean_object* lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__12 = (const lean_object*)&lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__12_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instReprGoalWithMVars___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instReprGoalWithMVars___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_instReprGoalWithMVars___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instReprGoalWithMVars___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instReprGoalWithMVars___closed__0 = (const lean_object*)&lp_aesop_Aesop_instReprGoalWithMVars___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_instReprGoalWithMVars___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Name_reprPrec___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instReprGoalWithMVars___closed__1 = (const lean_object*)&lp_aesop_Aesop_instReprGoalWithMVars___closed__1_value;
static const lean_closure_object lp_aesop_Aesop_instReprGoalWithMVars___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instReprGoalWithMVars___lam__2___boxed, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_aesop_Aesop_instReprGoalWithMVars___closed__1_value),((lean_object*)&lp_aesop_Aesop_instReprGoalWithMVars___closed__0_value)} };
static const lean_object* lp_aesop_Aesop_instReprGoalWithMVars___closed__2 = (const lean_object*)&lp_aesop_Aesop_instReprGoalWithMVars___closed__2_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instReprGoalWithMVars = (const lean_object*)&lp_aesop_Aesop_instReprGoalWithMVars___closed__2_value;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqGoalWithMVars___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqGoalWithMVars___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_instBEqGoalWithMVars___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instBEqGoalWithMVars___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instBEqGoalWithMVars___closed__0 = (const lean_object*)&lp_aesop_Aesop_instBEqGoalWithMVars___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instBEqGoalWithMVars = (const lean_object*)&lp_aesop_Aesop_instBEqGoalWithMVars___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalWithMVars_ofMVarId(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalWithMVars_ofMVarId___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_aesop_Aesop_instInhabitedGoalWithMVars_default___closed__0(void){
_start:
{
lean_object* v___x_1_; lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_1_ = lean_box(0);
v___x_2_ = lean_unsigned_to_nat(16u);
v___x_3_ = lean_mk_array(v___x_2_, v___x_1_);
return v___x_3_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedGoalWithMVars_default___closed__1(void){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_4_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedGoalWithMVars_default___closed__0, &lp_aesop_Aesop_instInhabitedGoalWithMVars_default___closed__0_once, _init_lp_aesop_Aesop_instInhabitedGoalWithMVars_default___closed__0);
v___x_5_ = lean_unsigned_to_nat(0u);
v___x_6_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_6_, 0, v___x_5_);
lean_ctor_set(v___x_6_, 1, v___x_4_);
return v___x_6_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedGoalWithMVars_default___closed__2(void){
_start:
{
lean_object* v___x_7_; lean_object* v___x_8_; lean_object* v___x_9_; 
v___x_7_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedGoalWithMVars_default___closed__1, &lp_aesop_Aesop_instInhabitedGoalWithMVars_default___closed__1_once, _init_lp_aesop_Aesop_instInhabitedGoalWithMVars_default___closed__1);
v___x_8_ = lean_box(0);
v___x_9_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_9_, 0, v___x_8_);
lean_ctor_set(v___x_9_, 1, v___x_7_);
return v___x_9_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedGoalWithMVars_default(void){
_start:
{
lean_object* v___x_10_; 
v___x_10_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedGoalWithMVars_default___closed__2, &lp_aesop_Aesop_instInhabitedGoalWithMVars_default___closed__2_once, _init_lp_aesop_Aesop_instInhabitedGoalWithMVars_default___closed__2);
return v___x_10_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedGoalWithMVars(void){
_start:
{
lean_object* v___x_11_; 
v___x_11_ = lp_aesop_Aesop_instInhabitedGoalWithMVars_default;
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instReprGoalWithMVars___lam__0(lean_object* v_x1_12_, lean_object* v_x2_13_, lean_object* v_x3_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lean_array_push(v_x1_12_, v_x2_13_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instReprGoalWithMVars___lam__1(lean_object* v___x_16_, lean_object* v___f_17_, lean_object* v_acc_18_, lean_object* v_l_19_){
_start:
{
lean_object* v___x_20_; 
v___x_20_ = l_Std_DHashMap_Internal_AssocList_foldlM___redArg(v___x_16_, v___f_17_, v_acc_18_, v_l_19_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instReprGoalWithMVars___lam__2(lean_object* v___f_43_, lean_object* v___f_44_, lean_object* v_x_45_, lean_object* v_x_46_){
_start:
{
lean_object* v_goal_47_; lean_object* v_mvars_48_; lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v_size_53_; lean_object* v_buckets_54_; lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___y_60_; lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; uint8_t v___x_70_; 
v_goal_47_ = lean_ctor_get(v_x_45_, 0);
lean_inc(v_goal_47_);
v_mvars_48_ = lean_ctor_get(v_x_45_, 1);
lean_inc_ref(v_mvars_48_);
lean_dec_ref(v_x_45_);
v___x_49_ = lean_unsigned_to_nat(0u);
v___x_50_ = l_Lean_Name_reprPrec(v_goal_47_, v___x_49_);
v___x_51_ = l_Std_Format_defWidth;
v___x_52_ = l_Std_Format_pretty(v___x_50_, v___x_51_, v___x_49_, v___x_49_);
v_size_53_ = lean_ctor_get(v_mvars_48_, 0);
lean_inc(v_size_53_);
v_buckets_54_ = lean_ctor_get(v_mvars_48_, 1);
lean_inc_ref(v_buckets_54_);
lean_dec_ref(v_mvars_48_);
v___x_55_ = ((lean_object*)(lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__0));
v___x_56_ = lean_string_append(v___x_55_, v___x_52_);
lean_dec_ref(v___x_52_);
v___x_57_ = ((lean_object*)(lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__1));
v___x_58_ = lean_string_append(v___x_56_, v___x_57_);
v___x_67_ = lean_mk_empty_array_with_capacity(v_size_53_);
lean_dec(v_size_53_);
v___x_68_ = ((lean_object*)(lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__12));
v___x_69_ = lean_array_get_size(v_buckets_54_);
v___x_70_ = lean_nat_dec_lt(v___x_49_, v___x_69_);
if (v___x_70_ == 0)
{
lean_dec_ref(v_buckets_54_);
lean_dec_ref(v___f_44_);
v___y_60_ = v___x_67_;
goto v___jp_59_;
}
else
{
lean_object* v___f_71_; uint8_t v___x_72_; 
v___f_71_ = lean_alloc_closure((void*)(lp_aesop_Aesop_instReprGoalWithMVars___lam__1), 4, 2);
lean_closure_set(v___f_71_, 0, v___x_68_);
lean_closure_set(v___f_71_, 1, v___f_44_);
v___x_72_ = lean_nat_dec_le(v___x_69_, v___x_69_);
if (v___x_72_ == 0)
{
if (v___x_70_ == 0)
{
lean_dec_ref(v___f_71_);
lean_dec_ref(v_buckets_54_);
v___y_60_ = v___x_67_;
goto v___jp_59_;
}
else
{
size_t v___x_73_; size_t v___x_74_; lean_object* v___x_75_; 
v___x_73_ = ((size_t)0ULL);
v___x_74_ = lean_usize_of_nat(v___x_69_);
v___x_75_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_68_, v___f_71_, v_buckets_54_, v___x_73_, v___x_74_, v___x_67_);
v___y_60_ = v___x_75_;
goto v___jp_59_;
}
}
else
{
size_t v___x_76_; size_t v___x_77_; lean_object* v___x_78_; 
v___x_76_ = ((size_t)0ULL);
v___x_77_ = lean_usize_of_nat(v___x_69_);
v___x_78_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_68_, v___f_71_, v_buckets_54_, v___x_76_, v___x_77_, v___x_67_);
v___y_60_ = v___x_78_;
goto v___jp_59_;
}
}
v___jp_59_:
{
lean_object* v___x_61_; lean_object* v___x_62_; lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; 
v___x_61_ = l_Array_repr___redArg(v___f_43_, v___y_60_);
v___x_62_ = l_Std_Format_pretty(v___x_61_, v___x_51_, v___x_49_, v___x_49_);
v___x_63_ = lean_string_append(v___x_58_, v___x_62_);
lean_dec_ref(v___x_62_);
v___x_64_ = ((lean_object*)(lp_aesop_Aesop_instReprGoalWithMVars___lam__2___closed__2));
v___x_65_ = lean_string_append(v___x_63_, v___x_64_);
v___x_66_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_66_, 0, v___x_65_);
return v___x_66_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instReprGoalWithMVars___lam__2___boxed(lean_object* v___f_79_, lean_object* v___f_80_, lean_object* v_x_81_, lean_object* v_x_82_){
_start:
{
lean_object* v_res_83_; 
v_res_83_ = lp_aesop_Aesop_instReprGoalWithMVars___lam__2(v___f_79_, v___f_80_, v_x_81_, v_x_82_);
lean_dec(v_x_82_);
return v_res_83_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqGoalWithMVars___lam__0(lean_object* v_g_u2081_90_, lean_object* v_g_u2082_91_){
_start:
{
lean_object* v_goal_92_; lean_object* v_goal_93_; uint8_t v___x_94_; 
v_goal_92_ = lean_ctor_get(v_g_u2081_90_, 0);
v_goal_93_ = lean_ctor_get(v_g_u2082_91_, 0);
v___x_94_ = l_Lean_instBEqMVarId_beq(v_goal_92_, v_goal_93_);
return v___x_94_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqGoalWithMVars___lam__0___boxed(lean_object* v_g_u2081_95_, lean_object* v_g_u2082_96_){
_start:
{
uint8_t v_res_97_; lean_object* v_r_98_; 
v_res_97_ = lp_aesop_Aesop_instBEqGoalWithMVars___lam__0(v_g_u2081_95_, v_g_u2082_96_);
lean_dec_ref(v_g_u2082_96_);
lean_dec_ref(v_g_u2081_95_);
v_r_98_ = lean_box(v_res_97_);
return v_r_98_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalWithMVars_ofMVarId(lean_object* v_goal_101_, lean_object* v_a_102_, lean_object* v_a_103_, lean_object* v_a_104_, lean_object* v_a_105_){
_start:
{
uint8_t v___x_107_; lean_object* v___x_108_; 
v___x_107_ = 0;
lean_inc(v_goal_101_);
v___x_108_ = l_Lean_MVarId_getMVarDependencies(v_goal_101_, v___x_107_, v_a_102_, v_a_103_, v_a_104_, v_a_105_);
if (lean_obj_tag(v___x_108_) == 0)
{
lean_object* v_a_109_; lean_object* v___x_111_; uint8_t v_isShared_112_; uint8_t v_isSharedCheck_117_; 
v_a_109_ = lean_ctor_get(v___x_108_, 0);
v_isSharedCheck_117_ = !lean_is_exclusive(v___x_108_);
if (v_isSharedCheck_117_ == 0)
{
v___x_111_ = v___x_108_;
v_isShared_112_ = v_isSharedCheck_117_;
goto v_resetjp_110_;
}
else
{
lean_inc(v_a_109_);
lean_dec(v___x_108_);
v___x_111_ = lean_box(0);
v_isShared_112_ = v_isSharedCheck_117_;
goto v_resetjp_110_;
}
v_resetjp_110_:
{
lean_object* v___x_113_; lean_object* v___x_115_; 
v___x_113_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_113_, 0, v_goal_101_);
lean_ctor_set(v___x_113_, 1, v_a_109_);
if (v_isShared_112_ == 0)
{
lean_ctor_set(v___x_111_, 0, v___x_113_);
v___x_115_ = v___x_111_;
goto v_reusejp_114_;
}
else
{
lean_object* v_reuseFailAlloc_116_; 
v_reuseFailAlloc_116_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_116_, 0, v___x_113_);
v___x_115_ = v_reuseFailAlloc_116_;
goto v_reusejp_114_;
}
v_reusejp_114_:
{
return v___x_115_;
}
}
}
else
{
lean_object* v_a_118_; lean_object* v___x_120_; uint8_t v_isShared_121_; uint8_t v_isSharedCheck_125_; 
lean_dec(v_goal_101_);
v_a_118_ = lean_ctor_get(v___x_108_, 0);
v_isSharedCheck_125_ = !lean_is_exclusive(v___x_108_);
if (v_isSharedCheck_125_ == 0)
{
v___x_120_ = v___x_108_;
v_isShared_121_ = v_isSharedCheck_125_;
goto v_resetjp_119_;
}
else
{
lean_inc(v_a_118_);
lean_dec(v___x_108_);
v___x_120_ = lean_box(0);
v_isShared_121_ = v_isSharedCheck_125_;
goto v_resetjp_119_;
}
v_resetjp_119_:
{
lean_object* v___x_123_; 
if (v_isShared_121_ == 0)
{
v___x_123_ = v___x_120_;
goto v_reusejp_122_;
}
else
{
lean_object* v_reuseFailAlloc_124_; 
v_reuseFailAlloc_124_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_124_, 0, v_a_118_);
v___x_123_ = v_reuseFailAlloc_124_;
goto v_reusejp_122_;
}
v_reusejp_122_:
{
return v___x_123_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalWithMVars_ofMVarId___boxed(lean_object* v_goal_126_, lean_object* v_a_127_, lean_object* v_a_128_, lean_object* v_a_129_, lean_object* v_a_130_, lean_object* v_a_131_){
_start:
{
lean_object* v_res_132_; 
v_res_132_ = lp_aesop_Aesop_GoalWithMVars_ofMVarId(v_goal_126_, v_a_127_, v_a_128_, v_a_129_, v_a_130_);
lean_dec(v_a_130_);
lean_dec_ref(v_a_129_);
lean_dec(v_a_128_);
lean_dec_ref(v_a_127_);
return v_res_132_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Basic(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_CollectMVars(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Script_GoalWithMVars(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_CollectMVars(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_instInhabitedGoalWithMVars_default = _init_lp_aesop_Aesop_instInhabitedGoalWithMVars_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedGoalWithMVars_default);
lp_aesop_Aesop_instInhabitedGoalWithMVars = _init_lp_aesop_Aesop_instInhabitedGoalWithMVars();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedGoalWithMVars);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Script_GoalWithMVars(uint8_t builtin) {
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
lean_object* initialize_Lean_Meta_Basic(uint8_t builtin);
lean_object* initialize_Lean_Meta_CollectMVars(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Script_GoalWithMVars(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_CollectMVars(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Script_GoalWithMVars(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Script_GoalWithMVars(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Script_GoalWithMVars(builtin);
}
#ifdef __cplusplus
}
#endif
