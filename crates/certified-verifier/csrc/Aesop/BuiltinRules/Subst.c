// Lean compiler output
// Module: Aesop.BuiltinRules.Subst
// Imports: public import Init public meta import Init public import Aesop.Frontend.Attribute public meta import Aesop.RuleTac.Forward.Basic
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
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_FVarId_getUserName___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_get_size(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l_Lean_Expr_fvar___override(lean_object*);
lean_object* l_Lean_Meta_mkPropExt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_replaceFVarS(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_FVarId_getType___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Expr_isAppOfArity(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_appFn_x21(lean_object*);
lean_object* l_Lean_Expr_appArg_x21(lean_object*);
uint8_t l_Lean_Expr_isFVar(lean_object*);
lean_object* lean_whnf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* l_Lean_LocalDecl_fvarId(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Meta_getLocalDeclFromUserName(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_subst_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_hideForwardImplDetailHyps(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Script_TacticBuilder_substFVars_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* lp_aesop_Aesop_mvarIdToSubgoal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_BuiltinRules_matchSubstitutableIff_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Iff"};
static const lean_object* lp_aesop_Aesop_BuiltinRules_matchSubstitutableIff_x3f___closed__0 = (const lean_object*)&lp_aesop_Aesop_BuiltinRules_matchSubstitutableIff_x3f___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_BuiltinRules_matchSubstitutableIff_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_BuiltinRules_matchSubstitutableIff_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(19, 54, 203, 28, 77, 25, 163, 137)}};
static const lean_object* lp_aesop_Aesop_BuiltinRules_matchSubstitutableIff_x3f___closed__1 = (const lean_object*)&lp_aesop_Aesop_BuiltinRules_matchSubstitutableIff_x3f___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_matchSubstitutableIff_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_matchSubstitutableIff_x3f___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_prepareIff_x3f_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_prepareIff_x3f_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_prepareIff_x3f_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_prepareIff_x3f_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_prepareIff_x3f_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_prepareIff_x3f_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_prepareIff_x3f___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_prepareIff_x3f___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_prepareIff_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_prepareIff_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_BuiltinRules_prepareIffs_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_BuiltinRules_prepareIffs_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_prepareIffs(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_prepareIffs___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep___at___00Aesop_BuiltinRules_substEqs_x3f_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep___at___00Aesop_BuiltinRules_substEqs_x3f_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep___at___00Aesop_BuiltinRules_substEqs_x3f_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep___at___00Aesop_BuiltinRules_substEqs_x3f_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_BuiltinRules_substEqs_x3f_spec__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_BuiltinRules_substEqs_x3f_spec__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_BuiltinRules_substEqs_x3f_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_BuiltinRules_substEqs_x3f_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_substEqs_x3f_spec__0___redArg(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_substEqs_x3f_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_substEqs_x3f_spec__0(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_substEqs_x3f_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_aesop_Aesop_BuiltinRules_substEqs_x3f___boxed__const__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + sizeof(size_t)*1, .m_other = 0, .m_tag = 0}, .m_objs = {(lean_object*)(size_t)(0ULL)}};
LEAN_EXPORT const lean_object* lp_aesop_Aesop_BuiltinRules_substEqs_x3f___boxed__const__1 = (const lean_object*)&lp_aesop_Aesop_BuiltinRules_substEqs_x3f___boxed__const__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_substEqs_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_substEqs_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_substEqsAndIffs_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_substEqsAndIffs_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_subst_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_subst_spec__2___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_subst_spec__2___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_subst_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_subst_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_subst_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_subst_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_subst_spec__3___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_subst_spec__3___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_subst_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_subst_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_subst_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_subst_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_BuiltinRules_subst_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_BuiltinRules_subst_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_subst_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_subst_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_subst_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "unexpected index match location"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_subst_spec__1___closed__0 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_subst_spec__1___closed__0_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_subst_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_subst_spec__1___closed__1;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_subst_spec__1(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_subst_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_BuiltinRules_subst___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "no suitable hypothesis found"};
static const lean_object* lp_aesop_Aesop_BuiltinRules_subst___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_BuiltinRules_subst___lam__0___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_BuiltinRules_subst___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_BuiltinRules_subst___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_subst___lam__0(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_subst___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_subst(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_subst___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_subst_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_subst_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_matchSubstitutableIff_x3f(lean_object* v_e_4_){
_start:
{
lean_object* v___x_5_; lean_object* v___x_6_; uint8_t v___x_7_; 
v___x_5_ = ((lean_object*)(lp_aesop_Aesop_BuiltinRules_matchSubstitutableIff_x3f___closed__1));
v___x_6_ = lean_unsigned_to_nat(2u);
v___x_7_ = l_Lean_Expr_isAppOfArity(v_e_4_, v___x_5_, v___x_6_);
if (v___x_7_ == 0)
{
lean_object* v___x_8_; 
v___x_8_ = lean_box(0);
return v___x_8_;
}
else
{
lean_object* v___x_9_; lean_object* v___x_10_; lean_object* v___x_11_; lean_object* v___x_12_; uint8_t v___y_14_; uint8_t v___x_17_; 
v___x_9_ = l_Lean_Expr_appFn_x21(v_e_4_);
v___x_10_ = l_Lean_Expr_appArg_x21(v___x_9_);
lean_dec_ref(v___x_9_);
v___x_11_ = l_Lean_Expr_appArg_x21(v_e_4_);
lean_inc_ref(v___x_11_);
lean_inc_ref(v___x_10_);
v___x_12_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_12_, 0, v___x_10_);
lean_ctor_set(v___x_12_, 1, v___x_11_);
v___x_17_ = l_Lean_Expr_isFVar(v___x_10_);
lean_dec_ref(v___x_10_);
if (v___x_17_ == 0)
{
uint8_t v___x_18_; 
v___x_18_ = l_Lean_Expr_isFVar(v___x_11_);
lean_dec_ref(v___x_11_);
v___y_14_ = v___x_18_;
goto v___jp_13_;
}
else
{
lean_dec_ref(v___x_11_);
v___y_14_ = v___x_17_;
goto v___jp_13_;
}
v___jp_13_:
{
if (v___y_14_ == 0)
{
lean_object* v___x_15_; 
lean_dec_ref_known(v___x_12_, 2);
v___x_15_ = lean_box(0);
return v___x_15_;
}
else
{
lean_object* v___x_16_; 
v___x_16_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_16_, 0, v___x_12_);
return v___x_16_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_matchSubstitutableIff_x3f___boxed(lean_object* v_e_19_){
_start:
{
lean_object* v_res_20_; 
v_res_20_ = lp_aesop_Aesop_BuiltinRules_matchSubstitutableIff_x3f(v_e_19_);
lean_dec_ref(v_e_19_);
return v_res_20_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_prepareIff_x3f_spec__0___redArg___lam__0(lean_object* v_x_21_, lean_object* v___y_22_, lean_object* v___y_23_, lean_object* v___y_24_, lean_object* v___y_25_, lean_object* v___y_26_, lean_object* v___y_27_){
_start:
{
lean_object* v___x_29_; 
lean_inc(v___y_23_);
lean_inc(v___y_22_);
v___x_29_ = lean_apply_7(v_x_21_, v___y_22_, v___y_23_, v___y_24_, v___y_25_, v___y_26_, v___y_27_, lean_box(0));
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_prepareIff_x3f_spec__0___redArg___lam__0___boxed(lean_object* v_x_30_, lean_object* v___y_31_, lean_object* v___y_32_, lean_object* v___y_33_, lean_object* v___y_34_, lean_object* v___y_35_, lean_object* v___y_36_, lean_object* v___y_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_prepareIff_x3f_spec__0___redArg___lam__0(v_x_30_, v___y_31_, v___y_32_, v___y_33_, v___y_34_, v___y_35_, v___y_36_);
lean_dec(v___y_32_);
lean_dec(v___y_31_);
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_prepareIff_x3f_spec__0___redArg(lean_object* v_mvarId_39_, lean_object* v_x_40_, lean_object* v___y_41_, lean_object* v___y_42_, lean_object* v___y_43_, lean_object* v___y_44_, lean_object* v___y_45_, lean_object* v___y_46_){
_start:
{
lean_object* v___f_48_; lean_object* v___x_49_; 
lean_inc(v___y_42_);
lean_inc(v___y_41_);
v___f_48_ = lean_alloc_closure((void*)(lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_prepareIff_x3f_spec__0___redArg___lam__0___boxed), 8, 3);
lean_closure_set(v___f_48_, 0, v_x_40_);
lean_closure_set(v___f_48_, 1, v___y_41_);
lean_closure_set(v___f_48_, 2, v___y_42_);
v___x_49_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_39_, v___f_48_, v___y_43_, v___y_44_, v___y_45_, v___y_46_);
if (lean_obj_tag(v___x_49_) == 0)
{
return v___x_49_;
}
else
{
lean_object* v_a_50_; lean_object* v___x_52_; uint8_t v_isShared_53_; uint8_t v_isSharedCheck_57_; 
v_a_50_ = lean_ctor_get(v___x_49_, 0);
v_isSharedCheck_57_ = !lean_is_exclusive(v___x_49_);
if (v_isSharedCheck_57_ == 0)
{
v___x_52_ = v___x_49_;
v_isShared_53_ = v_isSharedCheck_57_;
goto v_resetjp_51_;
}
else
{
lean_inc(v_a_50_);
lean_dec(v___x_49_);
v___x_52_ = lean_box(0);
v_isShared_53_ = v_isSharedCheck_57_;
goto v_resetjp_51_;
}
v_resetjp_51_:
{
lean_object* v___x_55_; 
if (v_isShared_53_ == 0)
{
v___x_55_ = v___x_52_;
goto v_reusejp_54_;
}
else
{
lean_object* v_reuseFailAlloc_56_; 
v_reuseFailAlloc_56_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_56_, 0, v_a_50_);
v___x_55_ = v_reuseFailAlloc_56_;
goto v_reusejp_54_;
}
v_reusejp_54_:
{
return v___x_55_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_prepareIff_x3f_spec__0___redArg___boxed(lean_object* v_mvarId_58_, lean_object* v_x_59_, lean_object* v___y_60_, lean_object* v___y_61_, lean_object* v___y_62_, lean_object* v___y_63_, lean_object* v___y_64_, lean_object* v___y_65_, lean_object* v___y_66_){
_start:
{
lean_object* v_res_67_; 
v_res_67_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_prepareIff_x3f_spec__0___redArg(v_mvarId_58_, v_x_59_, v___y_60_, v___y_61_, v___y_62_, v___y_63_, v___y_64_, v___y_65_);
lean_dec(v___y_65_);
lean_dec_ref(v___y_64_);
lean_dec(v___y_63_);
lean_dec_ref(v___y_62_);
lean_dec(v___y_61_);
lean_dec(v___y_60_);
return v_res_67_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_prepareIff_x3f_spec__0(lean_object* v_00_u03b1_68_, lean_object* v_mvarId_69_, lean_object* v_x_70_, lean_object* v___y_71_, lean_object* v___y_72_, lean_object* v___y_73_, lean_object* v___y_74_, lean_object* v___y_75_, lean_object* v___y_76_){
_start:
{
lean_object* v___x_78_; 
v___x_78_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_prepareIff_x3f_spec__0___redArg(v_mvarId_69_, v_x_70_, v___y_71_, v___y_72_, v___y_73_, v___y_74_, v___y_75_, v___y_76_);
return v___x_78_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_prepareIff_x3f_spec__0___boxed(lean_object* v_00_u03b1_79_, lean_object* v_mvarId_80_, lean_object* v_x_81_, lean_object* v___y_82_, lean_object* v___y_83_, lean_object* v___y_84_, lean_object* v___y_85_, lean_object* v___y_86_, lean_object* v___y_87_, lean_object* v___y_88_){
_start:
{
lean_object* v_res_89_; 
v_res_89_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_prepareIff_x3f_spec__0(v_00_u03b1_79_, v_mvarId_80_, v_x_81_, v___y_82_, v___y_83_, v___y_84_, v___y_85_, v___y_86_, v___y_87_);
lean_dec(v___y_87_);
lean_dec_ref(v___y_86_);
lean_dec(v___y_85_);
lean_dec_ref(v___y_84_);
lean_dec(v___y_83_);
lean_dec(v___y_82_);
return v_res_89_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_prepareIff_x3f___lam__0(lean_object* v_fvarId_90_, lean_object* v_mvarId_91_, lean_object* v___y_92_, lean_object* v___y_93_, lean_object* v___y_94_, lean_object* v___y_95_, lean_object* v___y_96_, lean_object* v___y_97_){
_start:
{
lean_object* v_a_100_; lean_object* v___x_192_; 
lean_inc(v_fvarId_90_);
v___x_192_ = l_Lean_FVarId_getType___redArg(v_fvarId_90_, v___y_94_, v___y_96_, v___y_97_);
if (lean_obj_tag(v___x_192_) == 0)
{
lean_object* v_a_193_; lean_object* v___x_194_; 
v_a_193_ = lean_ctor_get(v___x_192_, 0);
lean_inc(v_a_193_);
lean_dec_ref_known(v___x_192_, 1);
v___x_194_ = lp_aesop_Aesop_BuiltinRules_matchSubstitutableIff_x3f(v_a_193_);
if (lean_obj_tag(v___x_194_) == 0)
{
lean_object* v___x_195_; 
lean_inc(v___y_97_);
lean_inc_ref(v___y_96_);
lean_inc(v___y_95_);
lean_inc_ref(v___y_94_);
v___x_195_ = lean_whnf(v_a_193_, v___y_94_, v___y_95_, v___y_96_, v___y_97_);
if (lean_obj_tag(v___x_195_) == 0)
{
lean_object* v_a_196_; lean_object* v___x_197_; 
v_a_196_ = lean_ctor_get(v___x_195_, 0);
lean_inc(v_a_196_);
lean_dec_ref_known(v___x_195_, 1);
v___x_197_ = lp_aesop_Aesop_BuiltinRules_matchSubstitutableIff_x3f(v_a_196_);
lean_dec(v_a_196_);
v_a_100_ = v___x_197_;
goto v___jp_99_;
}
else
{
lean_object* v_a_198_; lean_object* v___x_200_; uint8_t v_isShared_201_; uint8_t v_isSharedCheck_205_; 
lean_dec(v_mvarId_91_);
lean_dec(v_fvarId_90_);
v_a_198_ = lean_ctor_get(v___x_195_, 0);
v_isSharedCheck_205_ = !lean_is_exclusive(v___x_195_);
if (v_isSharedCheck_205_ == 0)
{
v___x_200_ = v___x_195_;
v_isShared_201_ = v_isSharedCheck_205_;
goto v_resetjp_199_;
}
else
{
lean_inc(v_a_198_);
lean_dec(v___x_195_);
v___x_200_ = lean_box(0);
v_isShared_201_ = v_isSharedCheck_205_;
goto v_resetjp_199_;
}
v_resetjp_199_:
{
lean_object* v___x_203_; 
if (v_isShared_201_ == 0)
{
v___x_203_ = v___x_200_;
goto v_reusejp_202_;
}
else
{
lean_object* v_reuseFailAlloc_204_; 
v_reuseFailAlloc_204_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_204_, 0, v_a_198_);
v___x_203_ = v_reuseFailAlloc_204_;
goto v_reusejp_202_;
}
v_reusejp_202_:
{
return v___x_203_;
}
}
}
}
else
{
lean_dec(v_a_193_);
v_a_100_ = v___x_194_;
goto v___jp_99_;
}
}
else
{
lean_object* v_a_206_; lean_object* v___x_208_; uint8_t v_isShared_209_; uint8_t v_isSharedCheck_213_; 
lean_dec(v_mvarId_91_);
lean_dec(v_fvarId_90_);
v_a_206_ = lean_ctor_get(v___x_192_, 0);
v_isSharedCheck_213_ = !lean_is_exclusive(v___x_192_);
if (v_isSharedCheck_213_ == 0)
{
v___x_208_ = v___x_192_;
v_isShared_209_ = v_isSharedCheck_213_;
goto v_resetjp_207_;
}
else
{
lean_inc(v_a_206_);
lean_dec(v___x_192_);
v___x_208_ = lean_box(0);
v_isShared_209_ = v_isSharedCheck_213_;
goto v_resetjp_207_;
}
v_resetjp_207_:
{
lean_object* v___x_211_; 
if (v_isShared_209_ == 0)
{
v___x_211_ = v___x_208_;
goto v_reusejp_210_;
}
else
{
lean_object* v_reuseFailAlloc_212_; 
v_reuseFailAlloc_212_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_212_, 0, v_a_206_);
v___x_211_ = v_reuseFailAlloc_212_;
goto v_reusejp_210_;
}
v_reusejp_210_:
{
return v___x_211_;
}
}
}
v___jp_99_:
{
if (lean_obj_tag(v_a_100_) == 1)
{
lean_object* v_val_101_; lean_object* v___x_103_; uint8_t v_isShared_104_; uint8_t v_isSharedCheck_189_; 
v_val_101_ = lean_ctor_get(v_a_100_, 0);
v_isSharedCheck_189_ = !lean_is_exclusive(v_a_100_);
if (v_isSharedCheck_189_ == 0)
{
v___x_103_ = v_a_100_;
v_isShared_104_ = v_isSharedCheck_189_;
goto v_resetjp_102_;
}
else
{
lean_inc(v_val_101_);
lean_dec(v_a_100_);
v___x_103_ = lean_box(0);
v_isShared_104_ = v_isSharedCheck_189_;
goto v_resetjp_102_;
}
v_resetjp_102_:
{
lean_object* v_fst_105_; lean_object* v_snd_106_; lean_object* v___x_107_; lean_object* v___x_108_; 
v_fst_105_ = lean_ctor_get(v_val_101_, 0);
lean_inc(v_fst_105_);
v_snd_106_ = lean_ctor_get(v_val_101_, 1);
lean_inc(v_snd_106_);
lean_dec(v_val_101_);
lean_inc(v_fvarId_90_);
v___x_107_ = l_Lean_Expr_fvar___override(v_fvarId_90_);
v___x_108_ = l_Lean_Meta_mkPropExt(v___x_107_, v___y_94_, v___y_95_, v___y_96_, v___y_97_);
if (lean_obj_tag(v___x_108_) == 0)
{
lean_object* v_a_109_; lean_object* v___x_110_; 
v_a_109_ = lean_ctor_get(v___x_108_, 0);
lean_inc(v_a_109_);
lean_dec_ref_known(v___x_108_, 1);
v___x_110_ = l_Lean_Meta_mkEq(v_fst_105_, v_snd_106_, v___y_94_, v___y_95_, v___y_96_, v___y_97_);
if (lean_obj_tag(v___x_110_) == 0)
{
lean_object* v_a_111_; lean_object* v___x_112_; 
v_a_111_ = lean_ctor_get(v___x_110_, 0);
lean_inc(v_a_111_);
lean_dec_ref_known(v___x_110_, 1);
v___x_112_ = l_Lean_Meta_saveState___redArg(v___y_95_, v___y_97_);
if (lean_obj_tag(v___x_112_) == 0)
{
lean_object* v_a_113_; lean_object* v___x_114_; 
v_a_113_ = lean_ctor_get(v___x_112_, 0);
lean_inc(v_a_113_);
lean_dec_ref_known(v___x_112_, 1);
v___x_114_ = lp_aesop_Aesop_replaceFVarS(v_mvarId_91_, v_fvarId_90_, v_a_111_, v_a_109_, v___y_92_, v___y_93_, v___y_94_, v___y_95_, v___y_96_, v___y_97_);
if (lean_obj_tag(v___x_114_) == 0)
{
lean_object* v_a_115_; lean_object* v___x_117_; uint8_t v_isShared_118_; uint8_t v_isSharedCheck_156_; 
v_a_115_ = lean_ctor_get(v___x_114_, 0);
v_isSharedCheck_156_ = !lean_is_exclusive(v___x_114_);
if (v_isSharedCheck_156_ == 0)
{
v___x_117_ = v___x_114_;
v_isShared_118_ = v_isSharedCheck_156_;
goto v_resetjp_116_;
}
else
{
lean_inc(v_a_115_);
lean_dec(v___x_114_);
v___x_117_ = lean_box(0);
v_isShared_118_ = v_isSharedCheck_156_;
goto v_resetjp_116_;
}
v_resetjp_116_:
{
lean_object* v_snd_119_; lean_object* v_snd_120_; uint8_t v___x_121_; 
v_snd_119_ = lean_ctor_get(v_a_115_, 1);
lean_inc(v_snd_119_);
v_snd_120_ = lean_ctor_get(v_snd_119_, 1);
v___x_121_ = lean_unbox(v_snd_120_);
if (v___x_121_ == 0)
{
lean_object* v___x_122_; 
lean_dec(v_snd_119_);
lean_del_object(v___x_117_);
lean_dec(v_a_115_);
lean_del_object(v___x_103_);
v___x_122_ = l_Lean_Meta_SavedState_restore___redArg(v_a_113_, v___y_95_, v___y_97_);
lean_dec(v_a_113_);
if (lean_obj_tag(v___x_122_) == 0)
{
lean_object* v___x_124_; uint8_t v_isShared_125_; uint8_t v_isSharedCheck_130_; 
v_isSharedCheck_130_ = !lean_is_exclusive(v___x_122_);
if (v_isSharedCheck_130_ == 0)
{
lean_object* v_unused_131_; 
v_unused_131_ = lean_ctor_get(v___x_122_, 0);
lean_dec(v_unused_131_);
v___x_124_ = v___x_122_;
v_isShared_125_ = v_isSharedCheck_130_;
goto v_resetjp_123_;
}
else
{
lean_dec(v___x_122_);
v___x_124_ = lean_box(0);
v_isShared_125_ = v_isSharedCheck_130_;
goto v_resetjp_123_;
}
v_resetjp_123_:
{
lean_object* v___x_126_; lean_object* v___x_128_; 
v___x_126_ = lean_box(0);
if (v_isShared_125_ == 0)
{
lean_ctor_set(v___x_124_, 0, v___x_126_);
v___x_128_ = v___x_124_;
goto v_reusejp_127_;
}
else
{
lean_object* v_reuseFailAlloc_129_; 
v_reuseFailAlloc_129_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_129_, 0, v___x_126_);
v___x_128_ = v_reuseFailAlloc_129_;
goto v_reusejp_127_;
}
v_reusejp_127_:
{
return v___x_128_;
}
}
}
else
{
lean_object* v_a_132_; lean_object* v___x_134_; uint8_t v_isShared_135_; uint8_t v_isSharedCheck_139_; 
v_a_132_ = lean_ctor_get(v___x_122_, 0);
v_isSharedCheck_139_ = !lean_is_exclusive(v___x_122_);
if (v_isSharedCheck_139_ == 0)
{
v___x_134_ = v___x_122_;
v_isShared_135_ = v_isSharedCheck_139_;
goto v_resetjp_133_;
}
else
{
lean_inc(v_a_132_);
lean_dec(v___x_122_);
v___x_134_ = lean_box(0);
v_isShared_135_ = v_isSharedCheck_139_;
goto v_resetjp_133_;
}
v_resetjp_133_:
{
lean_object* v___x_137_; 
if (v_isShared_135_ == 0)
{
v___x_137_ = v___x_134_;
goto v_reusejp_136_;
}
else
{
lean_object* v_reuseFailAlloc_138_; 
v_reuseFailAlloc_138_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_138_, 0, v_a_132_);
v___x_137_ = v_reuseFailAlloc_138_;
goto v_reusejp_136_;
}
v_reusejp_136_:
{
return v___x_137_;
}
}
}
}
else
{
lean_object* v_fst_140_; lean_object* v_fst_141_; lean_object* v___x_143_; uint8_t v_isShared_144_; uint8_t v_isSharedCheck_154_; 
lean_dec(v_a_113_);
v_fst_140_ = lean_ctor_get(v_a_115_, 0);
lean_inc(v_fst_140_);
lean_dec(v_a_115_);
v_fst_141_ = lean_ctor_get(v_snd_119_, 0);
v_isSharedCheck_154_ = !lean_is_exclusive(v_snd_119_);
if (v_isSharedCheck_154_ == 0)
{
lean_object* v_unused_155_; 
v_unused_155_ = lean_ctor_get(v_snd_119_, 1);
lean_dec(v_unused_155_);
v___x_143_ = v_snd_119_;
v_isShared_144_ = v_isSharedCheck_154_;
goto v_resetjp_142_;
}
else
{
lean_inc(v_fst_141_);
lean_dec(v_snd_119_);
v___x_143_ = lean_box(0);
v_isShared_144_ = v_isSharedCheck_154_;
goto v_resetjp_142_;
}
v_resetjp_142_:
{
lean_object* v___x_146_; 
if (v_isShared_144_ == 0)
{
lean_ctor_set(v___x_143_, 1, v_fst_141_);
lean_ctor_set(v___x_143_, 0, v_fst_140_);
v___x_146_ = v___x_143_;
goto v_reusejp_145_;
}
else
{
lean_object* v_reuseFailAlloc_153_; 
v_reuseFailAlloc_153_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_153_, 0, v_fst_140_);
lean_ctor_set(v_reuseFailAlloc_153_, 1, v_fst_141_);
v___x_146_ = v_reuseFailAlloc_153_;
goto v_reusejp_145_;
}
v_reusejp_145_:
{
lean_object* v___x_148_; 
if (v_isShared_104_ == 0)
{
lean_ctor_set(v___x_103_, 0, v___x_146_);
v___x_148_ = v___x_103_;
goto v_reusejp_147_;
}
else
{
lean_object* v_reuseFailAlloc_152_; 
v_reuseFailAlloc_152_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_152_, 0, v___x_146_);
v___x_148_ = v_reuseFailAlloc_152_;
goto v_reusejp_147_;
}
v_reusejp_147_:
{
lean_object* v___x_150_; 
if (v_isShared_118_ == 0)
{
lean_ctor_set(v___x_117_, 0, v___x_148_);
v___x_150_ = v___x_117_;
goto v_reusejp_149_;
}
else
{
lean_object* v_reuseFailAlloc_151_; 
v_reuseFailAlloc_151_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_151_, 0, v___x_148_);
v___x_150_ = v_reuseFailAlloc_151_;
goto v_reusejp_149_;
}
v_reusejp_149_:
{
return v___x_150_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_157_; lean_object* v___x_159_; uint8_t v_isShared_160_; uint8_t v_isSharedCheck_164_; 
lean_dec(v_a_113_);
lean_del_object(v___x_103_);
v_a_157_ = lean_ctor_get(v___x_114_, 0);
v_isSharedCheck_164_ = !lean_is_exclusive(v___x_114_);
if (v_isSharedCheck_164_ == 0)
{
v___x_159_ = v___x_114_;
v_isShared_160_ = v_isSharedCheck_164_;
goto v_resetjp_158_;
}
else
{
lean_inc(v_a_157_);
lean_dec(v___x_114_);
v___x_159_ = lean_box(0);
v_isShared_160_ = v_isSharedCheck_164_;
goto v_resetjp_158_;
}
v_resetjp_158_:
{
lean_object* v___x_162_; 
if (v_isShared_160_ == 0)
{
v___x_162_ = v___x_159_;
goto v_reusejp_161_;
}
else
{
lean_object* v_reuseFailAlloc_163_; 
v_reuseFailAlloc_163_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_163_, 0, v_a_157_);
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
else
{
lean_object* v_a_165_; lean_object* v___x_167_; uint8_t v_isShared_168_; uint8_t v_isSharedCheck_172_; 
lean_dec(v_a_111_);
lean_dec(v_a_109_);
lean_del_object(v___x_103_);
lean_dec(v_mvarId_91_);
lean_dec(v_fvarId_90_);
v_a_165_ = lean_ctor_get(v___x_112_, 0);
v_isSharedCheck_172_ = !lean_is_exclusive(v___x_112_);
if (v_isSharedCheck_172_ == 0)
{
v___x_167_ = v___x_112_;
v_isShared_168_ = v_isSharedCheck_172_;
goto v_resetjp_166_;
}
else
{
lean_inc(v_a_165_);
lean_dec(v___x_112_);
v___x_167_ = lean_box(0);
v_isShared_168_ = v_isSharedCheck_172_;
goto v_resetjp_166_;
}
v_resetjp_166_:
{
lean_object* v___x_170_; 
if (v_isShared_168_ == 0)
{
v___x_170_ = v___x_167_;
goto v_reusejp_169_;
}
else
{
lean_object* v_reuseFailAlloc_171_; 
v_reuseFailAlloc_171_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_171_, 0, v_a_165_);
v___x_170_ = v_reuseFailAlloc_171_;
goto v_reusejp_169_;
}
v_reusejp_169_:
{
return v___x_170_;
}
}
}
}
else
{
lean_object* v_a_173_; lean_object* v___x_175_; uint8_t v_isShared_176_; uint8_t v_isSharedCheck_180_; 
lean_dec(v_a_109_);
lean_del_object(v___x_103_);
lean_dec(v_mvarId_91_);
lean_dec(v_fvarId_90_);
v_a_173_ = lean_ctor_get(v___x_110_, 0);
v_isSharedCheck_180_ = !lean_is_exclusive(v___x_110_);
if (v_isSharedCheck_180_ == 0)
{
v___x_175_ = v___x_110_;
v_isShared_176_ = v_isSharedCheck_180_;
goto v_resetjp_174_;
}
else
{
lean_inc(v_a_173_);
lean_dec(v___x_110_);
v___x_175_ = lean_box(0);
v_isShared_176_ = v_isSharedCheck_180_;
goto v_resetjp_174_;
}
v_resetjp_174_:
{
lean_object* v___x_178_; 
if (v_isShared_176_ == 0)
{
v___x_178_ = v___x_175_;
goto v_reusejp_177_;
}
else
{
lean_object* v_reuseFailAlloc_179_; 
v_reuseFailAlloc_179_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_179_, 0, v_a_173_);
v___x_178_ = v_reuseFailAlloc_179_;
goto v_reusejp_177_;
}
v_reusejp_177_:
{
return v___x_178_;
}
}
}
}
else
{
lean_object* v_a_181_; lean_object* v___x_183_; uint8_t v_isShared_184_; uint8_t v_isSharedCheck_188_; 
lean_dec(v_snd_106_);
lean_dec(v_fst_105_);
lean_del_object(v___x_103_);
lean_dec(v_mvarId_91_);
lean_dec(v_fvarId_90_);
v_a_181_ = lean_ctor_get(v___x_108_, 0);
v_isSharedCheck_188_ = !lean_is_exclusive(v___x_108_);
if (v_isSharedCheck_188_ == 0)
{
v___x_183_ = v___x_108_;
v_isShared_184_ = v_isSharedCheck_188_;
goto v_resetjp_182_;
}
else
{
lean_inc(v_a_181_);
lean_dec(v___x_108_);
v___x_183_ = lean_box(0);
v_isShared_184_ = v_isSharedCheck_188_;
goto v_resetjp_182_;
}
v_resetjp_182_:
{
lean_object* v___x_186_; 
if (v_isShared_184_ == 0)
{
v___x_186_ = v___x_183_;
goto v_reusejp_185_;
}
else
{
lean_object* v_reuseFailAlloc_187_; 
v_reuseFailAlloc_187_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_187_, 0, v_a_181_);
v___x_186_ = v_reuseFailAlloc_187_;
goto v_reusejp_185_;
}
v_reusejp_185_:
{
return v___x_186_;
}
}
}
}
}
else
{
lean_object* v___x_190_; lean_object* v___x_191_; 
lean_dec(v_a_100_);
lean_dec(v_mvarId_91_);
lean_dec(v_fvarId_90_);
v___x_190_ = lean_box(0);
v___x_191_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_191_, 0, v___x_190_);
return v___x_191_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_prepareIff_x3f___lam__0___boxed(lean_object* v_fvarId_214_, lean_object* v_mvarId_215_, lean_object* v___y_216_, lean_object* v___y_217_, lean_object* v___y_218_, lean_object* v___y_219_, lean_object* v___y_220_, lean_object* v___y_221_, lean_object* v___y_222_){
_start:
{
lean_object* v_res_223_; 
v_res_223_ = lp_aesop_Aesop_BuiltinRules_prepareIff_x3f___lam__0(v_fvarId_214_, v_mvarId_215_, v___y_216_, v___y_217_, v___y_218_, v___y_219_, v___y_220_, v___y_221_);
lean_dec(v___y_221_);
lean_dec_ref(v___y_220_);
lean_dec(v___y_219_);
lean_dec_ref(v___y_218_);
lean_dec(v___y_217_);
lean_dec(v___y_216_);
return v_res_223_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_prepareIff_x3f(lean_object* v_mvarId_224_, lean_object* v_fvarId_225_, lean_object* v_a_226_, lean_object* v_a_227_, lean_object* v_a_228_, lean_object* v_a_229_, lean_object* v_a_230_, lean_object* v_a_231_){
_start:
{
lean_object* v___f_233_; lean_object* v___x_234_; 
lean_inc(v_mvarId_224_);
v___f_233_ = lean_alloc_closure((void*)(lp_aesop_Aesop_BuiltinRules_prepareIff_x3f___lam__0___boxed), 9, 2);
lean_closure_set(v___f_233_, 0, v_fvarId_225_);
lean_closure_set(v___f_233_, 1, v_mvarId_224_);
v___x_234_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_prepareIff_x3f_spec__0___redArg(v_mvarId_224_, v___f_233_, v_a_226_, v_a_227_, v_a_228_, v_a_229_, v_a_230_, v_a_231_);
return v___x_234_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_prepareIff_x3f___boxed(lean_object* v_mvarId_235_, lean_object* v_fvarId_236_, lean_object* v_a_237_, lean_object* v_a_238_, lean_object* v_a_239_, lean_object* v_a_240_, lean_object* v_a_241_, lean_object* v_a_242_, lean_object* v_a_243_){
_start:
{
lean_object* v_res_244_; 
v_res_244_ = lp_aesop_Aesop_BuiltinRules_prepareIff_x3f(v_mvarId_235_, v_fvarId_236_, v_a_237_, v_a_238_, v_a_239_, v_a_240_, v_a_241_, v_a_242_);
lean_dec(v_a_242_);
lean_dec_ref(v_a_241_);
lean_dec(v_a_240_);
lean_dec_ref(v_a_239_);
lean_dec(v_a_238_);
lean_dec(v_a_237_);
return v_res_244_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_BuiltinRules_prepareIffs_spec__0(lean_object* v_as_245_, size_t v_sz_246_, size_t v_i_247_, lean_object* v_b_248_, lean_object* v___y_249_, lean_object* v___y_250_, lean_object* v___y_251_, lean_object* v___y_252_, lean_object* v___y_253_, lean_object* v___y_254_){
_start:
{
uint8_t v___x_256_; 
v___x_256_ = lean_usize_dec_lt(v_i_247_, v_sz_246_);
if (v___x_256_ == 0)
{
lean_object* v___x_257_; 
v___x_257_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_257_, 0, v_b_248_);
return v___x_257_;
}
else
{
lean_object* v_fst_258_; lean_object* v_snd_259_; lean_object* v___x_261_; uint8_t v_isShared_262_; uint8_t v_isSharedCheck_294_; 
v_fst_258_ = lean_ctor_get(v_b_248_, 0);
v_snd_259_ = lean_ctor_get(v_b_248_, 1);
v_isSharedCheck_294_ = !lean_is_exclusive(v_b_248_);
if (v_isSharedCheck_294_ == 0)
{
v___x_261_ = v_b_248_;
v_isShared_262_ = v_isSharedCheck_294_;
goto v_resetjp_260_;
}
else
{
lean_inc(v_snd_259_);
lean_inc(v_fst_258_);
lean_dec(v_b_248_);
v___x_261_ = lean_box(0);
v_isShared_262_ = v_isSharedCheck_294_;
goto v_resetjp_260_;
}
v_resetjp_260_:
{
lean_object* v_a_263_; lean_object* v___x_264_; 
v_a_263_ = lean_array_uget_borrowed(v_as_245_, v_i_247_);
lean_inc(v_a_263_);
lean_inc(v_fst_258_);
v___x_264_ = lp_aesop_Aesop_BuiltinRules_prepareIff_x3f(v_fst_258_, v_a_263_, v___y_249_, v___y_250_, v___y_251_, v___y_252_, v___y_253_, v___y_254_);
if (lean_obj_tag(v___x_264_) == 0)
{
lean_object* v_a_265_; lean_object* v_a_267_; 
v_a_265_ = lean_ctor_get(v___x_264_, 0);
lean_inc(v_a_265_);
lean_dec_ref_known(v___x_264_, 1);
if (lean_obj_tag(v_a_265_) == 1)
{
lean_object* v_val_271_; lean_object* v_fst_272_; lean_object* v_snd_273_; lean_object* v___x_275_; uint8_t v_isShared_276_; uint8_t v_isSharedCheck_281_; 
lean_del_object(v___x_261_);
lean_dec(v_fst_258_);
v_val_271_ = lean_ctor_get(v_a_265_, 0);
lean_inc(v_val_271_);
lean_dec_ref_known(v_a_265_, 1);
v_fst_272_ = lean_ctor_get(v_val_271_, 0);
v_snd_273_ = lean_ctor_get(v_val_271_, 1);
v_isSharedCheck_281_ = !lean_is_exclusive(v_val_271_);
if (v_isSharedCheck_281_ == 0)
{
v___x_275_ = v_val_271_;
v_isShared_276_ = v_isSharedCheck_281_;
goto v_resetjp_274_;
}
else
{
lean_inc(v_snd_273_);
lean_inc(v_fst_272_);
lean_dec(v_val_271_);
v___x_275_ = lean_box(0);
v_isShared_276_ = v_isSharedCheck_281_;
goto v_resetjp_274_;
}
v_resetjp_274_:
{
lean_object* v___x_277_; lean_object* v___x_279_; 
v___x_277_ = lean_array_push(v_snd_259_, v_snd_273_);
if (v_isShared_276_ == 0)
{
lean_ctor_set(v___x_275_, 1, v___x_277_);
v___x_279_ = v___x_275_;
goto v_reusejp_278_;
}
else
{
lean_object* v_reuseFailAlloc_280_; 
v_reuseFailAlloc_280_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_280_, 0, v_fst_272_);
lean_ctor_set(v_reuseFailAlloc_280_, 1, v___x_277_);
v___x_279_ = v_reuseFailAlloc_280_;
goto v_reusejp_278_;
}
v_reusejp_278_:
{
v_a_267_ = v___x_279_;
goto v___jp_266_;
}
}
}
else
{
lean_object* v___x_282_; lean_object* v___x_284_; 
lean_dec(v_a_265_);
lean_inc(v_a_263_);
v___x_282_ = lean_array_push(v_snd_259_, v_a_263_);
if (v_isShared_262_ == 0)
{
lean_ctor_set(v___x_261_, 1, v___x_282_);
v___x_284_ = v___x_261_;
goto v_reusejp_283_;
}
else
{
lean_object* v_reuseFailAlloc_285_; 
v_reuseFailAlloc_285_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_285_, 0, v_fst_258_);
lean_ctor_set(v_reuseFailAlloc_285_, 1, v___x_282_);
v___x_284_ = v_reuseFailAlloc_285_;
goto v_reusejp_283_;
}
v_reusejp_283_:
{
v_a_267_ = v___x_284_;
goto v___jp_266_;
}
}
v___jp_266_:
{
size_t v___x_268_; size_t v___x_269_; 
v___x_268_ = ((size_t)1ULL);
v___x_269_ = lean_usize_add(v_i_247_, v___x_268_);
v_i_247_ = v___x_269_;
v_b_248_ = v_a_267_;
goto _start;
}
}
else
{
lean_object* v_a_286_; lean_object* v___x_288_; uint8_t v_isShared_289_; uint8_t v_isSharedCheck_293_; 
lean_del_object(v___x_261_);
lean_dec(v_snd_259_);
lean_dec(v_fst_258_);
v_a_286_ = lean_ctor_get(v___x_264_, 0);
v_isSharedCheck_293_ = !lean_is_exclusive(v___x_264_);
if (v_isSharedCheck_293_ == 0)
{
v___x_288_ = v___x_264_;
v_isShared_289_ = v_isSharedCheck_293_;
goto v_resetjp_287_;
}
else
{
lean_inc(v_a_286_);
lean_dec(v___x_264_);
v___x_288_ = lean_box(0);
v_isShared_289_ = v_isSharedCheck_293_;
goto v_resetjp_287_;
}
v_resetjp_287_:
{
lean_object* v___x_291_; 
if (v_isShared_289_ == 0)
{
v___x_291_ = v___x_288_;
goto v_reusejp_290_;
}
else
{
lean_object* v_reuseFailAlloc_292_; 
v_reuseFailAlloc_292_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_292_, 0, v_a_286_);
v___x_291_ = v_reuseFailAlloc_292_;
goto v_reusejp_290_;
}
v_reusejp_290_:
{
return v___x_291_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_BuiltinRules_prepareIffs_spec__0___boxed(lean_object* v_as_295_, lean_object* v_sz_296_, lean_object* v_i_297_, lean_object* v_b_298_, lean_object* v___y_299_, lean_object* v___y_300_, lean_object* v___y_301_, lean_object* v___y_302_, lean_object* v___y_303_, lean_object* v___y_304_, lean_object* v___y_305_){
_start:
{
size_t v_sz_boxed_306_; size_t v_i_boxed_307_; lean_object* v_res_308_; 
v_sz_boxed_306_ = lean_unbox_usize(v_sz_296_);
lean_dec(v_sz_296_);
v_i_boxed_307_ = lean_unbox_usize(v_i_297_);
lean_dec(v_i_297_);
v_res_308_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_BuiltinRules_prepareIffs_spec__0(v_as_295_, v_sz_boxed_306_, v_i_boxed_307_, v_b_298_, v___y_299_, v___y_300_, v___y_301_, v___y_302_, v___y_303_, v___y_304_);
lean_dec(v___y_304_);
lean_dec_ref(v___y_303_);
lean_dec(v___y_302_);
lean_dec_ref(v___y_301_);
lean_dec(v___y_300_);
lean_dec(v___y_299_);
lean_dec_ref(v_as_295_);
return v_res_308_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_prepareIffs(lean_object* v_mvarId_309_, lean_object* v_fvarIds_310_, lean_object* v_a_311_, lean_object* v_a_312_, lean_object* v_a_313_, lean_object* v_a_314_, lean_object* v_a_315_, lean_object* v_a_316_){
_start:
{
lean_object* v___x_318_; lean_object* v_newFVarIds_319_; lean_object* v___x_320_; size_t v_sz_321_; size_t v___x_322_; lean_object* v___x_323_; 
v___x_318_ = lean_array_get_size(v_fvarIds_310_);
v_newFVarIds_319_ = lean_mk_empty_array_with_capacity(v___x_318_);
v___x_320_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_320_, 0, v_mvarId_309_);
lean_ctor_set(v___x_320_, 1, v_newFVarIds_319_);
v_sz_321_ = lean_array_size(v_fvarIds_310_);
v___x_322_ = ((size_t)0ULL);
v___x_323_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_BuiltinRules_prepareIffs_spec__0(v_fvarIds_310_, v_sz_321_, v___x_322_, v___x_320_, v_a_311_, v_a_312_, v_a_313_, v_a_314_, v_a_315_, v_a_316_);
if (lean_obj_tag(v___x_323_) == 0)
{
lean_object* v_a_324_; lean_object* v___x_326_; uint8_t v_isShared_327_; uint8_t v_isSharedCheck_340_; 
v_a_324_ = lean_ctor_get(v___x_323_, 0);
v_isSharedCheck_340_ = !lean_is_exclusive(v___x_323_);
if (v_isSharedCheck_340_ == 0)
{
v___x_326_ = v___x_323_;
v_isShared_327_ = v_isSharedCheck_340_;
goto v_resetjp_325_;
}
else
{
lean_inc(v_a_324_);
lean_dec(v___x_323_);
v___x_326_ = lean_box(0);
v_isShared_327_ = v_isSharedCheck_340_;
goto v_resetjp_325_;
}
v_resetjp_325_:
{
lean_object* v_fst_328_; lean_object* v_snd_329_; lean_object* v___x_331_; uint8_t v_isShared_332_; uint8_t v_isSharedCheck_339_; 
v_fst_328_ = lean_ctor_get(v_a_324_, 0);
v_snd_329_ = lean_ctor_get(v_a_324_, 1);
v_isSharedCheck_339_ = !lean_is_exclusive(v_a_324_);
if (v_isSharedCheck_339_ == 0)
{
v___x_331_ = v_a_324_;
v_isShared_332_ = v_isSharedCheck_339_;
goto v_resetjp_330_;
}
else
{
lean_inc(v_snd_329_);
lean_inc(v_fst_328_);
lean_dec(v_a_324_);
v___x_331_ = lean_box(0);
v_isShared_332_ = v_isSharedCheck_339_;
goto v_resetjp_330_;
}
v_resetjp_330_:
{
lean_object* v___x_334_; 
if (v_isShared_332_ == 0)
{
v___x_334_ = v___x_331_;
goto v_reusejp_333_;
}
else
{
lean_object* v_reuseFailAlloc_338_; 
v_reuseFailAlloc_338_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_338_, 0, v_fst_328_);
lean_ctor_set(v_reuseFailAlloc_338_, 1, v_snd_329_);
v___x_334_ = v_reuseFailAlloc_338_;
goto v_reusejp_333_;
}
v_reusejp_333_:
{
lean_object* v___x_336_; 
if (v_isShared_327_ == 0)
{
lean_ctor_set(v___x_326_, 0, v___x_334_);
v___x_336_ = v___x_326_;
goto v_reusejp_335_;
}
else
{
lean_object* v_reuseFailAlloc_337_; 
v_reuseFailAlloc_337_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_337_, 0, v___x_334_);
v___x_336_ = v_reuseFailAlloc_337_;
goto v_reusejp_335_;
}
v_reusejp_335_:
{
return v___x_336_;
}
}
}
}
}
else
{
return v___x_323_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_prepareIffs___boxed(lean_object* v_mvarId_341_, lean_object* v_fvarIds_342_, lean_object* v_a_343_, lean_object* v_a_344_, lean_object* v_a_345_, lean_object* v_a_346_, lean_object* v_a_347_, lean_object* v_a_348_, lean_object* v_a_349_){
_start:
{
lean_object* v_res_350_; 
v_res_350_ = lp_aesop_Aesop_BuiltinRules_prepareIffs(v_mvarId_341_, v_fvarIds_342_, v_a_343_, v_a_344_, v_a_345_, v_a_346_, v_a_347_, v_a_348_);
lean_dec(v_a_348_);
lean_dec_ref(v_a_347_);
lean_dec(v_a_346_);
lean_dec_ref(v_a_345_);
lean_dec(v_a_344_);
lean_dec(v_a_343_);
lean_dec_ref(v_fvarIds_342_);
return v_res_350_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep___at___00Aesop_BuiltinRules_substEqs_x3f_spec__2___redArg(lean_object* v_step_351_, lean_object* v___y_352_){
_start:
{
lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; 
v___x_354_ = lean_st_ref_take(v___y_352_);
v___x_355_ = lean_array_push(v___x_354_, v_step_351_);
v___x_356_ = lean_st_ref_set(v___y_352_, v___x_355_);
v___x_357_ = lean_box(0);
v___x_358_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_358_, 0, v___x_357_);
return v___x_358_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep___at___00Aesop_BuiltinRules_substEqs_x3f_spec__2___redArg___boxed(lean_object* v_step_359_, lean_object* v___y_360_, lean_object* v___y_361_){
_start:
{
lean_object* v_res_362_; 
v_res_362_ = lp_aesop_Aesop_recordScriptStep___at___00Aesop_BuiltinRules_substEqs_x3f_spec__2___redArg(v_step_359_, v___y_360_);
lean_dec(v___y_360_);
return v_res_362_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep___at___00Aesop_BuiltinRules_substEqs_x3f_spec__2(lean_object* v_step_363_, lean_object* v___y_364_, lean_object* v___y_365_, lean_object* v___y_366_, lean_object* v___y_367_, lean_object* v___y_368_, lean_object* v___y_369_){
_start:
{
lean_object* v___x_371_; 
v___x_371_ = lp_aesop_Aesop_recordScriptStep___at___00Aesop_BuiltinRules_substEqs_x3f_spec__2___redArg(v_step_363_, v___y_364_);
return v___x_371_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep___at___00Aesop_BuiltinRules_substEqs_x3f_spec__2___boxed(lean_object* v_step_372_, lean_object* v___y_373_, lean_object* v___y_374_, lean_object* v___y_375_, lean_object* v___y_376_, lean_object* v___y_377_, lean_object* v___y_378_, lean_object* v___y_379_){
_start:
{
lean_object* v_res_380_; 
v_res_380_ = lp_aesop_Aesop_recordScriptStep___at___00Aesop_BuiltinRules_substEqs_x3f_spec__2(v_step_372_, v___y_373_, v___y_374_, v___y_375_, v___y_376_, v___y_377_, v___y_378_);
lean_dec(v___y_378_);
lean_dec_ref(v___y_377_);
lean_dec(v___y_376_);
lean_dec_ref(v___y_375_);
lean_dec(v___y_374_);
lean_dec(v___y_373_);
return v_res_380_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_BuiltinRules_substEqs_x3f_spec__1___lam__0(lean_object* v_a_381_, lean_object* v___y_382_, lean_object* v___y_383_, lean_object* v___y_384_, lean_object* v___y_385_, lean_object* v___y_386_, lean_object* v___y_387_){
_start:
{
lean_object* v___x_389_; 
v___x_389_ = l_Lean_Meta_getLocalDeclFromUserName(v_a_381_, v___y_384_, v___y_385_, v___y_386_, v___y_387_);
return v___x_389_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_BuiltinRules_substEqs_x3f_spec__1___lam__0___boxed(lean_object* v_a_390_, lean_object* v___y_391_, lean_object* v___y_392_, lean_object* v___y_393_, lean_object* v___y_394_, lean_object* v___y_395_, lean_object* v___y_396_, lean_object* v___y_397_){
_start:
{
lean_object* v_res_398_; 
v_res_398_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_BuiltinRules_substEqs_x3f_spec__1___lam__0(v_a_390_, v___y_391_, v___y_392_, v___y_393_, v___y_394_, v___y_395_, v___y_396_);
lean_dec(v___y_396_);
lean_dec_ref(v___y_395_);
lean_dec(v___y_394_);
lean_dec_ref(v___y_393_);
lean_dec(v___y_392_);
lean_dec(v___y_391_);
return v_res_398_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_BuiltinRules_substEqs_x3f_spec__1(lean_object* v_as_399_, size_t v_sz_400_, size_t v_i_401_, lean_object* v_b_402_, lean_object* v___y_403_, lean_object* v___y_404_, lean_object* v___y_405_, lean_object* v___y_406_, lean_object* v___y_407_, lean_object* v___y_408_){
_start:
{
uint8_t v___x_410_; 
v___x_410_ = lean_usize_dec_lt(v_i_401_, v_sz_400_);
if (v___x_410_ == 0)
{
lean_object* v___x_411_; 
v___x_411_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_411_, 0, v_b_402_);
return v___x_411_;
}
else
{
lean_object* v_fst_412_; lean_object* v_snd_413_; lean_object* v___x_415_; uint8_t v_isShared_416_; uint8_t v_isSharedCheck_453_; 
v_fst_412_ = lean_ctor_get(v_b_402_, 0);
v_snd_413_ = lean_ctor_get(v_b_402_, 1);
v_isSharedCheck_453_ = !lean_is_exclusive(v_b_402_);
if (v_isSharedCheck_453_ == 0)
{
v___x_415_ = v_b_402_;
v_isShared_416_ = v_isSharedCheck_453_;
goto v_resetjp_414_;
}
else
{
lean_inc(v_snd_413_);
lean_inc(v_fst_412_);
lean_dec(v_b_402_);
v___x_415_ = lean_box(0);
v_isShared_416_ = v_isSharedCheck_453_;
goto v_resetjp_414_;
}
v_resetjp_414_:
{
lean_object* v_a_417_; lean_object* v___f_418_; lean_object* v___x_419_; 
v_a_417_ = lean_array_uget_borrowed(v_as_399_, v_i_401_);
lean_inc(v_a_417_);
v___f_418_ = lean_alloc_closure((void*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_BuiltinRules_substEqs_x3f_spec__1___lam__0___boxed), 8, 1);
lean_closure_set(v___f_418_, 0, v_a_417_);
lean_inc(v_fst_412_);
v___x_419_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_prepareIff_x3f_spec__0___redArg(v_fst_412_, v___f_418_, v___y_403_, v___y_404_, v___y_405_, v___y_406_, v___y_407_, v___y_408_);
if (lean_obj_tag(v___x_419_) == 0)
{
lean_object* v_a_420_; lean_object* v___x_421_; lean_object* v___x_422_; 
v_a_420_ = lean_ctor_get(v___x_419_, 0);
lean_inc(v_a_420_);
lean_dec_ref_known(v___x_419_, 1);
v___x_421_ = l_Lean_LocalDecl_fvarId(v_a_420_);
lean_dec(v_a_420_);
lean_inc(v_fst_412_);
v___x_422_ = l_Lean_Meta_subst_x3f(v_fst_412_, v___x_421_, v___y_405_, v___y_406_, v___y_407_, v___y_408_);
if (lean_obj_tag(v___x_422_) == 0)
{
lean_object* v_a_423_; lean_object* v_a_425_; 
v_a_423_ = lean_ctor_get(v___x_422_, 0);
lean_inc(v_a_423_);
lean_dec_ref_known(v___x_422_, 1);
if (lean_obj_tag(v_a_423_) == 1)
{
lean_object* v_val_429_; lean_object* v___x_430_; lean_object* v___x_432_; 
lean_dec(v_fst_412_);
v_val_429_ = lean_ctor_get(v_a_423_, 0);
lean_inc(v_val_429_);
lean_dec_ref_known(v_a_423_, 1);
lean_inc(v_a_417_);
v___x_430_ = lean_array_push(v_snd_413_, v_a_417_);
if (v_isShared_416_ == 0)
{
lean_ctor_set(v___x_415_, 1, v___x_430_);
lean_ctor_set(v___x_415_, 0, v_val_429_);
v___x_432_ = v___x_415_;
goto v_reusejp_431_;
}
else
{
lean_object* v_reuseFailAlloc_433_; 
v_reuseFailAlloc_433_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_433_, 0, v_val_429_);
lean_ctor_set(v_reuseFailAlloc_433_, 1, v___x_430_);
v___x_432_ = v_reuseFailAlloc_433_;
goto v_reusejp_431_;
}
v_reusejp_431_:
{
v_a_425_ = v___x_432_;
goto v___jp_424_;
}
}
else
{
lean_object* v___x_435_; 
lean_dec(v_a_423_);
if (v_isShared_416_ == 0)
{
v___x_435_ = v___x_415_;
goto v_reusejp_434_;
}
else
{
lean_object* v_reuseFailAlloc_436_; 
v_reuseFailAlloc_436_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_436_, 0, v_fst_412_);
lean_ctor_set(v_reuseFailAlloc_436_, 1, v_snd_413_);
v___x_435_ = v_reuseFailAlloc_436_;
goto v_reusejp_434_;
}
v_reusejp_434_:
{
v_a_425_ = v___x_435_;
goto v___jp_424_;
}
}
v___jp_424_:
{
size_t v___x_426_; size_t v___x_427_; 
v___x_426_ = ((size_t)1ULL);
v___x_427_ = lean_usize_add(v_i_401_, v___x_426_);
v_i_401_ = v___x_427_;
v_b_402_ = v_a_425_;
goto _start;
}
}
else
{
lean_object* v_a_437_; lean_object* v___x_439_; uint8_t v_isShared_440_; uint8_t v_isSharedCheck_444_; 
lean_del_object(v___x_415_);
lean_dec(v_snd_413_);
lean_dec(v_fst_412_);
v_a_437_ = lean_ctor_get(v___x_422_, 0);
v_isSharedCheck_444_ = !lean_is_exclusive(v___x_422_);
if (v_isSharedCheck_444_ == 0)
{
v___x_439_ = v___x_422_;
v_isShared_440_ = v_isSharedCheck_444_;
goto v_resetjp_438_;
}
else
{
lean_inc(v_a_437_);
lean_dec(v___x_422_);
v___x_439_ = lean_box(0);
v_isShared_440_ = v_isSharedCheck_444_;
goto v_resetjp_438_;
}
v_resetjp_438_:
{
lean_object* v___x_442_; 
if (v_isShared_440_ == 0)
{
v___x_442_ = v___x_439_;
goto v_reusejp_441_;
}
else
{
lean_object* v_reuseFailAlloc_443_; 
v_reuseFailAlloc_443_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_443_, 0, v_a_437_);
v___x_442_ = v_reuseFailAlloc_443_;
goto v_reusejp_441_;
}
v_reusejp_441_:
{
return v___x_442_;
}
}
}
}
else
{
lean_object* v_a_445_; lean_object* v___x_447_; uint8_t v_isShared_448_; uint8_t v_isSharedCheck_452_; 
lean_del_object(v___x_415_);
lean_dec(v_snd_413_);
lean_dec(v_fst_412_);
v_a_445_ = lean_ctor_get(v___x_419_, 0);
v_isSharedCheck_452_ = !lean_is_exclusive(v___x_419_);
if (v_isSharedCheck_452_ == 0)
{
v___x_447_ = v___x_419_;
v_isShared_448_ = v_isSharedCheck_452_;
goto v_resetjp_446_;
}
else
{
lean_inc(v_a_445_);
lean_dec(v___x_419_);
v___x_447_ = lean_box(0);
v_isShared_448_ = v_isSharedCheck_452_;
goto v_resetjp_446_;
}
v_resetjp_446_:
{
lean_object* v___x_450_; 
if (v_isShared_448_ == 0)
{
v___x_450_ = v___x_447_;
goto v_reusejp_449_;
}
else
{
lean_object* v_reuseFailAlloc_451_; 
v_reuseFailAlloc_451_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_451_, 0, v_a_445_);
v___x_450_ = v_reuseFailAlloc_451_;
goto v_reusejp_449_;
}
v_reusejp_449_:
{
return v___x_450_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_BuiltinRules_substEqs_x3f_spec__1___boxed(lean_object* v_as_454_, lean_object* v_sz_455_, lean_object* v_i_456_, lean_object* v_b_457_, lean_object* v___y_458_, lean_object* v___y_459_, lean_object* v___y_460_, lean_object* v___y_461_, lean_object* v___y_462_, lean_object* v___y_463_, lean_object* v___y_464_){
_start:
{
size_t v_sz_boxed_465_; size_t v_i_boxed_466_; lean_object* v_res_467_; 
v_sz_boxed_465_ = lean_unbox_usize(v_sz_455_);
lean_dec(v_sz_455_);
v_i_boxed_466_ = lean_unbox_usize(v_i_456_);
lean_dec(v_i_456_);
v_res_467_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_BuiltinRules_substEqs_x3f_spec__1(v_as_454_, v_sz_boxed_465_, v_i_boxed_466_, v_b_457_, v___y_458_, v___y_459_, v___y_460_, v___y_461_, v___y_462_, v___y_463_);
lean_dec(v___y_463_);
lean_dec_ref(v___y_462_);
lean_dec(v___y_461_);
lean_dec_ref(v___y_460_);
lean_dec(v___y_459_);
lean_dec(v___y_458_);
lean_dec_ref(v_as_454_);
return v_res_467_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_substEqs_x3f_spec__0___redArg(size_t v_sz_468_, size_t v_i_469_, lean_object* v_bs_470_, lean_object* v___y_471_, lean_object* v___y_472_, lean_object* v___y_473_){
_start:
{
uint8_t v___x_475_; 
v___x_475_ = lean_usize_dec_lt(v_i_469_, v_sz_468_);
if (v___x_475_ == 0)
{
lean_object* v___x_476_; 
v___x_476_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_476_, 0, v_bs_470_);
return v___x_476_;
}
else
{
lean_object* v_v_477_; lean_object* v___x_478_; 
v_v_477_ = lean_array_uget_borrowed(v_bs_470_, v_i_469_);
lean_inc(v_v_477_);
v___x_478_ = l_Lean_FVarId_getUserName___redArg(v_v_477_, v___y_471_, v___y_472_, v___y_473_);
if (lean_obj_tag(v___x_478_) == 0)
{
lean_object* v_a_479_; lean_object* v___x_480_; lean_object* v_bs_x27_481_; size_t v___x_482_; size_t v___x_483_; lean_object* v___x_484_; 
v_a_479_ = lean_ctor_get(v___x_478_, 0);
lean_inc(v_a_479_);
lean_dec_ref_known(v___x_478_, 1);
v___x_480_ = lean_unsigned_to_nat(0u);
v_bs_x27_481_ = lean_array_uset(v_bs_470_, v_i_469_, v___x_480_);
v___x_482_ = ((size_t)1ULL);
v___x_483_ = lean_usize_add(v_i_469_, v___x_482_);
v___x_484_ = lean_array_uset(v_bs_x27_481_, v_i_469_, v_a_479_);
v_i_469_ = v___x_483_;
v_bs_470_ = v___x_484_;
goto _start;
}
else
{
lean_object* v_a_486_; lean_object* v___x_488_; uint8_t v_isShared_489_; uint8_t v_isSharedCheck_493_; 
lean_dec_ref(v_bs_470_);
v_a_486_ = lean_ctor_get(v___x_478_, 0);
v_isSharedCheck_493_ = !lean_is_exclusive(v___x_478_);
if (v_isSharedCheck_493_ == 0)
{
v___x_488_ = v___x_478_;
v_isShared_489_ = v_isSharedCheck_493_;
goto v_resetjp_487_;
}
else
{
lean_inc(v_a_486_);
lean_dec(v___x_478_);
v___x_488_ = lean_box(0);
v_isShared_489_ = v_isSharedCheck_493_;
goto v_resetjp_487_;
}
v_resetjp_487_:
{
lean_object* v___x_491_; 
if (v_isShared_489_ == 0)
{
v___x_491_ = v___x_488_;
goto v_reusejp_490_;
}
else
{
lean_object* v_reuseFailAlloc_492_; 
v_reuseFailAlloc_492_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_492_, 0, v_a_486_);
v___x_491_ = v_reuseFailAlloc_492_;
goto v_reusejp_490_;
}
v_reusejp_490_:
{
return v___x_491_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_substEqs_x3f_spec__0___redArg___boxed(lean_object* v_sz_494_, lean_object* v_i_495_, lean_object* v_bs_496_, lean_object* v___y_497_, lean_object* v___y_498_, lean_object* v___y_499_, lean_object* v___y_500_){
_start:
{
size_t v_sz_boxed_501_; size_t v_i_boxed_502_; lean_object* v_res_503_; 
v_sz_boxed_501_ = lean_unbox_usize(v_sz_494_);
lean_dec(v_sz_494_);
v_i_boxed_502_ = lean_unbox_usize(v_i_495_);
lean_dec(v_i_495_);
v_res_503_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_substEqs_x3f_spec__0___redArg(v_sz_boxed_501_, v_i_boxed_502_, v_bs_496_, v___y_497_, v___y_498_, v___y_499_);
lean_dec(v___y_499_);
lean_dec_ref(v___y_498_);
lean_dec_ref(v___y_497_);
return v_res_503_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_substEqs_x3f_spec__0(size_t v_sz_504_, size_t v_i_505_, lean_object* v_bs_506_, lean_object* v___y_507_, lean_object* v___y_508_, lean_object* v___y_509_, lean_object* v___y_510_, lean_object* v___y_511_, lean_object* v___y_512_){
_start:
{
lean_object* v___x_514_; 
v___x_514_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_substEqs_x3f_spec__0___redArg(v_sz_504_, v_i_505_, v_bs_506_, v___y_509_, v___y_511_, v___y_512_);
return v___x_514_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_substEqs_x3f_spec__0___boxed(lean_object* v_sz_515_, lean_object* v_i_516_, lean_object* v_bs_517_, lean_object* v___y_518_, lean_object* v___y_519_, lean_object* v___y_520_, lean_object* v___y_521_, lean_object* v___y_522_, lean_object* v___y_523_, lean_object* v___y_524_){
_start:
{
size_t v_sz_boxed_525_; size_t v_i_boxed_526_; lean_object* v_res_527_; 
v_sz_boxed_525_ = lean_unbox_usize(v_sz_515_);
lean_dec(v_sz_515_);
v_i_boxed_526_ = lean_unbox_usize(v_i_516_);
lean_dec(v_i_516_);
v_res_527_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_substEqs_x3f_spec__0(v_sz_boxed_525_, v_i_boxed_526_, v_bs_517_, v___y_518_, v___y_519_, v___y_520_, v___y_521_, v___y_522_, v___y_523_);
lean_dec(v___y_523_);
lean_dec_ref(v___y_522_);
lean_dec(v___y_521_);
lean_dec_ref(v___y_520_);
lean_dec(v___y_519_);
lean_dec(v___y_518_);
return v_res_527_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_substEqs_x3f(lean_object* v_goal_530_, lean_object* v_fvarIds_531_, lean_object* v_a_532_, lean_object* v_a_533_, lean_object* v_a_534_, lean_object* v_a_535_, lean_object* v_a_536_, lean_object* v_a_537_){
_start:
{
lean_object* v___x_539_; 
v___x_539_ = l_Lean_Meta_saveState___redArg(v_a_535_, v_a_537_);
if (lean_obj_tag(v___x_539_) == 0)
{
lean_object* v_a_540_; size_t v_sz_541_; size_t v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_545_; lean_object* v___x_546_; 
v_a_540_ = lean_ctor_get(v___x_539_, 0);
lean_inc(v_a_540_);
lean_dec_ref_known(v___x_539_, 1);
v_sz_541_ = lean_array_size(v_fvarIds_531_);
v___x_542_ = ((size_t)0ULL);
v___x_543_ = lean_box_usize(v_sz_541_);
v___x_544_ = ((lean_object*)(lp_aesop_Aesop_BuiltinRules_substEqs_x3f___boxed__const__1));
v___x_545_ = lean_alloc_closure((void*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_substEqs_x3f_spec__0___boxed), 10, 3);
lean_closure_set(v___x_545_, 0, v___x_543_);
lean_closure_set(v___x_545_, 1, v___x_544_);
lean_closure_set(v___x_545_, 2, v_fvarIds_531_);
lean_inc(v_goal_530_);
v___x_546_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_prepareIff_x3f_spec__0___redArg(v_goal_530_, v___x_545_, v_a_532_, v_a_533_, v_a_534_, v_a_535_, v_a_536_, v_a_537_);
if (lean_obj_tag(v___x_546_) == 0)
{
lean_object* v_a_547_; lean_object* v___x_548_; lean_object* v___x_549_; lean_object* v___x_550_; size_t v_sz_551_; lean_object* v___x_552_; 
v_a_547_ = lean_ctor_get(v___x_546_, 0);
lean_inc(v_a_547_);
lean_dec_ref_known(v___x_546_, 1);
v___x_548_ = lean_array_get_size(v_a_547_);
v___x_549_ = lean_mk_empty_array_with_capacity(v___x_548_);
lean_inc(v_goal_530_);
v___x_550_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_550_, 0, v_goal_530_);
lean_ctor_set(v___x_550_, 1, v___x_549_);
v_sz_551_ = lean_array_size(v_a_547_);
v___x_552_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_BuiltinRules_substEqs_x3f_spec__1(v_a_547_, v_sz_551_, v___x_542_, v___x_550_, v_a_532_, v_a_533_, v_a_534_, v_a_535_, v_a_536_, v_a_537_);
lean_dec(v_a_547_);
if (lean_obj_tag(v___x_552_) == 0)
{
lean_object* v_a_553_; lean_object* v___x_555_; uint8_t v_isShared_556_; uint8_t v_isSharedCheck_600_; 
v_a_553_ = lean_ctor_get(v___x_552_, 0);
v_isSharedCheck_600_ = !lean_is_exclusive(v___x_552_);
if (v_isSharedCheck_600_ == 0)
{
v___x_555_ = v___x_552_;
v_isShared_556_ = v_isSharedCheck_600_;
goto v_resetjp_554_;
}
else
{
lean_inc(v_a_553_);
lean_dec(v___x_552_);
v___x_555_ = lean_box(0);
v_isShared_556_ = v_isSharedCheck_600_;
goto v_resetjp_554_;
}
v_resetjp_554_:
{
lean_object* v_fst_557_; lean_object* v_snd_558_; uint8_t v___x_559_; 
v_fst_557_ = lean_ctor_get(v_a_553_, 0);
lean_inc(v_fst_557_);
v_snd_558_ = lean_ctor_get(v_a_553_, 1);
lean_inc(v_snd_558_);
lean_dec(v_a_553_);
v___x_559_ = l_Lean_instBEqMVarId_beq(v_fst_557_, v_goal_530_);
if (v___x_559_ == 0)
{
lean_object* v___x_560_; 
lean_del_object(v___x_555_);
v___x_560_ = lp_aesop_Aesop_hideForwardImplDetailHyps(v_fst_557_, v_a_534_, v_a_535_, v_a_536_, v_a_537_);
if (lean_obj_tag(v___x_560_) == 0)
{
lean_object* v_a_561_; lean_object* v___x_562_; 
v_a_561_ = lean_ctor_get(v___x_560_, 0);
lean_inc(v_a_561_);
lean_dec_ref_known(v___x_560_, 1);
v___x_562_ = l_Lean_Meta_saveState___redArg(v_a_535_, v_a_537_);
if (lean_obj_tag(v___x_562_) == 0)
{
lean_object* v_a_563_; lean_object* v___x_564_; lean_object* v___x_565_; lean_object* v___x_566_; lean_object* v___x_567_; lean_object* v___x_568_; lean_object* v___x_569_; lean_object* v___x_570_; lean_object* v___x_572_; uint8_t v_isShared_573_; uint8_t v_isSharedCheck_578_; 
v_a_563_ = lean_ctor_get(v___x_562_, 0);
lean_inc(v_a_563_);
lean_dec_ref_known(v___x_562_, 1);
v___x_564_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_TacticBuilder_substFVars_x27___boxed), 6, 1);
lean_closure_set(v___x_564_, 0, v_snd_558_);
v___x_565_ = lean_unsigned_to_nat(1u);
v___x_566_ = lean_mk_empty_array_with_capacity(v___x_565_);
lean_inc_ref(v___x_566_);
v___x_567_ = lean_array_push(v___x_566_, v___x_564_);
lean_inc(v_a_561_);
v___x_568_ = lean_array_push(v___x_566_, v_a_561_);
v___x_569_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_569_, 0, v_a_540_);
lean_ctor_set(v___x_569_, 1, v_goal_530_);
lean_ctor_set(v___x_569_, 2, v___x_567_);
lean_ctor_set(v___x_569_, 3, v_a_563_);
lean_ctor_set(v___x_569_, 4, v___x_568_);
v___x_570_ = lp_aesop_Aesop_recordScriptStep___at___00Aesop_BuiltinRules_substEqs_x3f_spec__2___redArg(v___x_569_, v_a_532_);
v_isSharedCheck_578_ = !lean_is_exclusive(v___x_570_);
if (v_isSharedCheck_578_ == 0)
{
lean_object* v_unused_579_; 
v_unused_579_ = lean_ctor_get(v___x_570_, 0);
lean_dec(v_unused_579_);
v___x_572_ = v___x_570_;
v_isShared_573_ = v_isSharedCheck_578_;
goto v_resetjp_571_;
}
else
{
lean_dec(v___x_570_);
v___x_572_ = lean_box(0);
v_isShared_573_ = v_isSharedCheck_578_;
goto v_resetjp_571_;
}
v_resetjp_571_:
{
lean_object* v___x_574_; lean_object* v___x_576_; 
v___x_574_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_574_, 0, v_a_561_);
if (v_isShared_573_ == 0)
{
lean_ctor_set(v___x_572_, 0, v___x_574_);
v___x_576_ = v___x_572_;
goto v_reusejp_575_;
}
else
{
lean_object* v_reuseFailAlloc_577_; 
v_reuseFailAlloc_577_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_577_, 0, v___x_574_);
v___x_576_ = v_reuseFailAlloc_577_;
goto v_reusejp_575_;
}
v_reusejp_575_:
{
return v___x_576_;
}
}
}
else
{
lean_object* v_a_580_; lean_object* v___x_582_; uint8_t v_isShared_583_; uint8_t v_isSharedCheck_587_; 
lean_dec(v_a_561_);
lean_dec(v_snd_558_);
lean_dec(v_a_540_);
lean_dec(v_goal_530_);
v_a_580_ = lean_ctor_get(v___x_562_, 0);
v_isSharedCheck_587_ = !lean_is_exclusive(v___x_562_);
if (v_isSharedCheck_587_ == 0)
{
v___x_582_ = v___x_562_;
v_isShared_583_ = v_isSharedCheck_587_;
goto v_resetjp_581_;
}
else
{
lean_inc(v_a_580_);
lean_dec(v___x_562_);
v___x_582_ = lean_box(0);
v_isShared_583_ = v_isSharedCheck_587_;
goto v_resetjp_581_;
}
v_resetjp_581_:
{
lean_object* v___x_585_; 
if (v_isShared_583_ == 0)
{
v___x_585_ = v___x_582_;
goto v_reusejp_584_;
}
else
{
lean_object* v_reuseFailAlloc_586_; 
v_reuseFailAlloc_586_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_586_, 0, v_a_580_);
v___x_585_ = v_reuseFailAlloc_586_;
goto v_reusejp_584_;
}
v_reusejp_584_:
{
return v___x_585_;
}
}
}
}
else
{
lean_object* v_a_588_; lean_object* v___x_590_; uint8_t v_isShared_591_; uint8_t v_isSharedCheck_595_; 
lean_dec(v_snd_558_);
lean_dec(v_a_540_);
lean_dec(v_goal_530_);
v_a_588_ = lean_ctor_get(v___x_560_, 0);
v_isSharedCheck_595_ = !lean_is_exclusive(v___x_560_);
if (v_isSharedCheck_595_ == 0)
{
v___x_590_ = v___x_560_;
v_isShared_591_ = v_isSharedCheck_595_;
goto v_resetjp_589_;
}
else
{
lean_inc(v_a_588_);
lean_dec(v___x_560_);
v___x_590_ = lean_box(0);
v_isShared_591_ = v_isSharedCheck_595_;
goto v_resetjp_589_;
}
v_resetjp_589_:
{
lean_object* v___x_593_; 
if (v_isShared_591_ == 0)
{
v___x_593_ = v___x_590_;
goto v_reusejp_592_;
}
else
{
lean_object* v_reuseFailAlloc_594_; 
v_reuseFailAlloc_594_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_594_, 0, v_a_588_);
v___x_593_ = v_reuseFailAlloc_594_;
goto v_reusejp_592_;
}
v_reusejp_592_:
{
return v___x_593_;
}
}
}
}
else
{
lean_object* v___x_596_; lean_object* v___x_598_; 
lean_dec(v_snd_558_);
lean_dec(v_fst_557_);
lean_dec(v_a_540_);
lean_dec(v_goal_530_);
v___x_596_ = lean_box(0);
if (v_isShared_556_ == 0)
{
lean_ctor_set(v___x_555_, 0, v___x_596_);
v___x_598_ = v___x_555_;
goto v_reusejp_597_;
}
else
{
lean_object* v_reuseFailAlloc_599_; 
v_reuseFailAlloc_599_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_599_, 0, v___x_596_);
v___x_598_ = v_reuseFailAlloc_599_;
goto v_reusejp_597_;
}
v_reusejp_597_:
{
return v___x_598_;
}
}
}
}
else
{
lean_object* v_a_601_; lean_object* v___x_603_; uint8_t v_isShared_604_; uint8_t v_isSharedCheck_608_; 
lean_dec(v_a_540_);
lean_dec(v_goal_530_);
v_a_601_ = lean_ctor_get(v___x_552_, 0);
v_isSharedCheck_608_ = !lean_is_exclusive(v___x_552_);
if (v_isSharedCheck_608_ == 0)
{
v___x_603_ = v___x_552_;
v_isShared_604_ = v_isSharedCheck_608_;
goto v_resetjp_602_;
}
else
{
lean_inc(v_a_601_);
lean_dec(v___x_552_);
v___x_603_ = lean_box(0);
v_isShared_604_ = v_isSharedCheck_608_;
goto v_resetjp_602_;
}
v_resetjp_602_:
{
lean_object* v___x_606_; 
if (v_isShared_604_ == 0)
{
v___x_606_ = v___x_603_;
goto v_reusejp_605_;
}
else
{
lean_object* v_reuseFailAlloc_607_; 
v_reuseFailAlloc_607_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_607_, 0, v_a_601_);
v___x_606_ = v_reuseFailAlloc_607_;
goto v_reusejp_605_;
}
v_reusejp_605_:
{
return v___x_606_;
}
}
}
}
else
{
lean_object* v_a_609_; lean_object* v___x_611_; uint8_t v_isShared_612_; uint8_t v_isSharedCheck_616_; 
lean_dec(v_a_540_);
lean_dec(v_goal_530_);
v_a_609_ = lean_ctor_get(v___x_546_, 0);
v_isSharedCheck_616_ = !lean_is_exclusive(v___x_546_);
if (v_isSharedCheck_616_ == 0)
{
v___x_611_ = v___x_546_;
v_isShared_612_ = v_isSharedCheck_616_;
goto v_resetjp_610_;
}
else
{
lean_inc(v_a_609_);
lean_dec(v___x_546_);
v___x_611_ = lean_box(0);
v_isShared_612_ = v_isSharedCheck_616_;
goto v_resetjp_610_;
}
v_resetjp_610_:
{
lean_object* v___x_614_; 
if (v_isShared_612_ == 0)
{
v___x_614_ = v___x_611_;
goto v_reusejp_613_;
}
else
{
lean_object* v_reuseFailAlloc_615_; 
v_reuseFailAlloc_615_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_615_, 0, v_a_609_);
v___x_614_ = v_reuseFailAlloc_615_;
goto v_reusejp_613_;
}
v_reusejp_613_:
{
return v___x_614_;
}
}
}
}
else
{
lean_object* v_a_617_; lean_object* v___x_619_; uint8_t v_isShared_620_; uint8_t v_isSharedCheck_624_; 
lean_dec_ref(v_fvarIds_531_);
lean_dec(v_goal_530_);
v_a_617_ = lean_ctor_get(v___x_539_, 0);
v_isSharedCheck_624_ = !lean_is_exclusive(v___x_539_);
if (v_isSharedCheck_624_ == 0)
{
v___x_619_ = v___x_539_;
v_isShared_620_ = v_isSharedCheck_624_;
goto v_resetjp_618_;
}
else
{
lean_inc(v_a_617_);
lean_dec(v___x_539_);
v___x_619_ = lean_box(0);
v_isShared_620_ = v_isSharedCheck_624_;
goto v_resetjp_618_;
}
v_resetjp_618_:
{
lean_object* v___x_622_; 
if (v_isShared_620_ == 0)
{
v___x_622_ = v___x_619_;
goto v_reusejp_621_;
}
else
{
lean_object* v_reuseFailAlloc_623_; 
v_reuseFailAlloc_623_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_623_, 0, v_a_617_);
v___x_622_ = v_reuseFailAlloc_623_;
goto v_reusejp_621_;
}
v_reusejp_621_:
{
return v___x_622_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_substEqs_x3f___boxed(lean_object* v_goal_625_, lean_object* v_fvarIds_626_, lean_object* v_a_627_, lean_object* v_a_628_, lean_object* v_a_629_, lean_object* v_a_630_, lean_object* v_a_631_, lean_object* v_a_632_, lean_object* v_a_633_){
_start:
{
lean_object* v_res_634_; 
v_res_634_ = lp_aesop_Aesop_BuiltinRules_substEqs_x3f(v_goal_625_, v_fvarIds_626_, v_a_627_, v_a_628_, v_a_629_, v_a_630_, v_a_631_, v_a_632_);
lean_dec(v_a_632_);
lean_dec_ref(v_a_631_);
lean_dec(v_a_630_);
lean_dec_ref(v_a_629_);
lean_dec(v_a_628_);
lean_dec(v_a_627_);
return v_res_634_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_substEqsAndIffs_x3f(lean_object* v_goal_635_, lean_object* v_fvarIds_636_, lean_object* v_a_637_, lean_object* v_a_638_, lean_object* v_a_639_, lean_object* v_a_640_, lean_object* v_a_641_, lean_object* v_a_642_){
_start:
{
lean_object* v___x_644_; 
v___x_644_ = l_Lean_Meta_saveState___redArg(v_a_640_, v_a_642_);
if (lean_obj_tag(v___x_644_) == 0)
{
lean_object* v_a_645_; lean_object* v___x_646_; 
v_a_645_ = lean_ctor_get(v___x_644_, 0);
lean_inc(v_a_645_);
lean_dec_ref_known(v___x_644_, 1);
v___x_646_ = lp_aesop_Aesop_BuiltinRules_prepareIffs(v_goal_635_, v_fvarIds_636_, v_a_637_, v_a_638_, v_a_639_, v_a_640_, v_a_641_, v_a_642_);
if (lean_obj_tag(v___x_646_) == 0)
{
lean_object* v_a_647_; lean_object* v_fst_648_; lean_object* v_snd_649_; lean_object* v___x_650_; 
v_a_647_ = lean_ctor_get(v___x_646_, 0);
lean_inc(v_a_647_);
lean_dec_ref_known(v___x_646_, 1);
v_fst_648_ = lean_ctor_get(v_a_647_, 0);
lean_inc(v_fst_648_);
v_snd_649_ = lean_ctor_get(v_a_647_, 1);
lean_inc(v_snd_649_);
lean_dec(v_a_647_);
v___x_650_ = lp_aesop_Aesop_BuiltinRules_substEqs_x3f(v_fst_648_, v_snd_649_, v_a_637_, v_a_638_, v_a_639_, v_a_640_, v_a_641_, v_a_642_);
if (lean_obj_tag(v___x_650_) == 0)
{
lean_object* v_a_651_; 
v_a_651_ = lean_ctor_get(v___x_650_, 0);
lean_inc(v_a_651_);
if (lean_obj_tag(v_a_651_) == 1)
{
lean_dec_ref_known(v_a_651_, 1);
lean_dec(v_a_645_);
return v___x_650_;
}
else
{
lean_object* v___x_652_; 
lean_dec_ref_known(v___x_650_, 1);
lean_dec(v_a_651_);
v___x_652_ = l_Lean_Meta_SavedState_restore___redArg(v_a_645_, v_a_640_, v_a_642_);
lean_dec(v_a_645_);
if (lean_obj_tag(v___x_652_) == 0)
{
lean_object* v___x_654_; uint8_t v_isShared_655_; uint8_t v_isSharedCheck_660_; 
v_isSharedCheck_660_ = !lean_is_exclusive(v___x_652_);
if (v_isSharedCheck_660_ == 0)
{
lean_object* v_unused_661_; 
v_unused_661_ = lean_ctor_get(v___x_652_, 0);
lean_dec(v_unused_661_);
v___x_654_ = v___x_652_;
v_isShared_655_ = v_isSharedCheck_660_;
goto v_resetjp_653_;
}
else
{
lean_dec(v___x_652_);
v___x_654_ = lean_box(0);
v_isShared_655_ = v_isSharedCheck_660_;
goto v_resetjp_653_;
}
v_resetjp_653_:
{
lean_object* v___x_656_; lean_object* v___x_658_; 
v___x_656_ = lean_box(0);
if (v_isShared_655_ == 0)
{
lean_ctor_set(v___x_654_, 0, v___x_656_);
v___x_658_ = v___x_654_;
goto v_reusejp_657_;
}
else
{
lean_object* v_reuseFailAlloc_659_; 
v_reuseFailAlloc_659_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_659_, 0, v___x_656_);
v___x_658_ = v_reuseFailAlloc_659_;
goto v_reusejp_657_;
}
v_reusejp_657_:
{
return v___x_658_;
}
}
}
else
{
lean_object* v_a_662_; lean_object* v___x_664_; uint8_t v_isShared_665_; uint8_t v_isSharedCheck_669_; 
v_a_662_ = lean_ctor_get(v___x_652_, 0);
v_isSharedCheck_669_ = !lean_is_exclusive(v___x_652_);
if (v_isSharedCheck_669_ == 0)
{
v___x_664_ = v___x_652_;
v_isShared_665_ = v_isSharedCheck_669_;
goto v_resetjp_663_;
}
else
{
lean_inc(v_a_662_);
lean_dec(v___x_652_);
v___x_664_ = lean_box(0);
v_isShared_665_ = v_isSharedCheck_669_;
goto v_resetjp_663_;
}
v_resetjp_663_:
{
lean_object* v___x_667_; 
if (v_isShared_665_ == 0)
{
v___x_667_ = v___x_664_;
goto v_reusejp_666_;
}
else
{
lean_object* v_reuseFailAlloc_668_; 
v_reuseFailAlloc_668_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_668_, 0, v_a_662_);
v___x_667_ = v_reuseFailAlloc_668_;
goto v_reusejp_666_;
}
v_reusejp_666_:
{
return v___x_667_;
}
}
}
}
}
else
{
lean_dec(v_a_645_);
return v___x_650_;
}
}
else
{
lean_object* v_a_670_; lean_object* v___x_672_; uint8_t v_isShared_673_; uint8_t v_isSharedCheck_677_; 
lean_dec(v_a_645_);
v_a_670_ = lean_ctor_get(v___x_646_, 0);
v_isSharedCheck_677_ = !lean_is_exclusive(v___x_646_);
if (v_isSharedCheck_677_ == 0)
{
v___x_672_ = v___x_646_;
v_isShared_673_ = v_isSharedCheck_677_;
goto v_resetjp_671_;
}
else
{
lean_inc(v_a_670_);
lean_dec(v___x_646_);
v___x_672_ = lean_box(0);
v_isShared_673_ = v_isSharedCheck_677_;
goto v_resetjp_671_;
}
v_resetjp_671_:
{
lean_object* v___x_675_; 
if (v_isShared_673_ == 0)
{
v___x_675_ = v___x_672_;
goto v_reusejp_674_;
}
else
{
lean_object* v_reuseFailAlloc_676_; 
v_reuseFailAlloc_676_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_676_, 0, v_a_670_);
v___x_675_ = v_reuseFailAlloc_676_;
goto v_reusejp_674_;
}
v_reusejp_674_:
{
return v___x_675_;
}
}
}
}
else
{
lean_object* v_a_678_; lean_object* v___x_680_; uint8_t v_isShared_681_; uint8_t v_isSharedCheck_685_; 
lean_dec(v_goal_635_);
v_a_678_ = lean_ctor_get(v___x_644_, 0);
v_isSharedCheck_685_ = !lean_is_exclusive(v___x_644_);
if (v_isSharedCheck_685_ == 0)
{
v___x_680_ = v___x_644_;
v_isShared_681_ = v_isSharedCheck_685_;
goto v_resetjp_679_;
}
else
{
lean_inc(v_a_678_);
lean_dec(v___x_644_);
v___x_680_ = lean_box(0);
v_isShared_681_ = v_isSharedCheck_685_;
goto v_resetjp_679_;
}
v_resetjp_679_:
{
lean_object* v___x_683_; 
if (v_isShared_681_ == 0)
{
v___x_683_ = v___x_680_;
goto v_reusejp_682_;
}
else
{
lean_object* v_reuseFailAlloc_684_; 
v_reuseFailAlloc_684_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_684_, 0, v_a_678_);
v___x_683_ = v_reuseFailAlloc_684_;
goto v_reusejp_682_;
}
v_reusejp_682_:
{
return v___x_683_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_substEqsAndIffs_x3f___boxed(lean_object* v_goal_686_, lean_object* v_fvarIds_687_, lean_object* v_a_688_, lean_object* v_a_689_, lean_object* v_a_690_, lean_object* v_a_691_, lean_object* v_a_692_, lean_object* v_a_693_, lean_object* v_a_694_){
_start:
{
lean_object* v_res_695_; 
v_res_695_ = lp_aesop_Aesop_BuiltinRules_substEqsAndIffs_x3f(v_goal_686_, v_fvarIds_687_, v_a_688_, v_a_689_, v_a_690_, v_a_691_, v_a_692_, v_a_693_);
lean_dec(v_a_693_);
lean_dec_ref(v_a_692_);
lean_dec(v_a_691_);
lean_dec_ref(v_a_690_);
lean_dec(v_a_689_);
lean_dec(v_a_688_);
lean_dec_ref(v_fvarIds_687_);
return v_res_695_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_subst_spec__2___redArg(lean_object* v_x_698_, lean_object* v___y_699_, lean_object* v___y_700_, lean_object* v___y_701_, lean_object* v___y_702_, lean_object* v___y_703_){
_start:
{
lean_object* v___x_705_; lean_object* v___x_706_; lean_object* v___x_707_; 
v___x_705_ = ((lean_object*)(lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_subst_spec__2___redArg___closed__0));
v___x_706_ = lean_st_mk_ref(v___x_705_);
lean_inc(v___y_703_);
lean_inc_ref(v___y_702_);
lean_inc(v___y_701_);
lean_inc_ref(v___y_700_);
lean_inc(v___y_699_);
lean_inc(v___x_706_);
v___x_707_ = lean_apply_7(v_x_698_, v___x_706_, v___y_699_, v___y_700_, v___y_701_, v___y_702_, v___y_703_, lean_box(0));
if (lean_obj_tag(v___x_707_) == 0)
{
lean_object* v_a_708_; lean_object* v___x_710_; uint8_t v_isShared_711_; uint8_t v_isSharedCheck_717_; 
v_a_708_ = lean_ctor_get(v___x_707_, 0);
v_isSharedCheck_717_ = !lean_is_exclusive(v___x_707_);
if (v_isSharedCheck_717_ == 0)
{
v___x_710_ = v___x_707_;
v_isShared_711_ = v_isSharedCheck_717_;
goto v_resetjp_709_;
}
else
{
lean_inc(v_a_708_);
lean_dec(v___x_707_);
v___x_710_ = lean_box(0);
v_isShared_711_ = v_isSharedCheck_717_;
goto v_resetjp_709_;
}
v_resetjp_709_:
{
lean_object* v___x_712_; lean_object* v___x_713_; lean_object* v___x_715_; 
v___x_712_ = lean_st_ref_get(v___x_706_);
lean_dec(v___x_706_);
v___x_713_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_713_, 0, v_a_708_);
lean_ctor_set(v___x_713_, 1, v___x_712_);
if (v_isShared_711_ == 0)
{
lean_ctor_set(v___x_710_, 0, v___x_713_);
v___x_715_ = v___x_710_;
goto v_reusejp_714_;
}
else
{
lean_object* v_reuseFailAlloc_716_; 
v_reuseFailAlloc_716_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_716_, 0, v___x_713_);
v___x_715_ = v_reuseFailAlloc_716_;
goto v_reusejp_714_;
}
v_reusejp_714_:
{
return v___x_715_;
}
}
}
else
{
lean_object* v_a_718_; lean_object* v___x_720_; uint8_t v_isShared_721_; uint8_t v_isSharedCheck_725_; 
lean_dec(v___x_706_);
v_a_718_ = lean_ctor_get(v___x_707_, 0);
v_isSharedCheck_725_ = !lean_is_exclusive(v___x_707_);
if (v_isSharedCheck_725_ == 0)
{
v___x_720_ = v___x_707_;
v_isShared_721_ = v_isSharedCheck_725_;
goto v_resetjp_719_;
}
else
{
lean_inc(v_a_718_);
lean_dec(v___x_707_);
v___x_720_ = lean_box(0);
v_isShared_721_ = v_isSharedCheck_725_;
goto v_resetjp_719_;
}
v_resetjp_719_:
{
lean_object* v___x_723_; 
if (v_isShared_721_ == 0)
{
v___x_723_ = v___x_720_;
goto v_reusejp_722_;
}
else
{
lean_object* v_reuseFailAlloc_724_; 
v_reuseFailAlloc_724_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_724_, 0, v_a_718_);
v___x_723_ = v_reuseFailAlloc_724_;
goto v_reusejp_722_;
}
v_reusejp_722_:
{
return v___x_723_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_subst_spec__2___redArg___boxed(lean_object* v_x_726_, lean_object* v___y_727_, lean_object* v___y_728_, lean_object* v___y_729_, lean_object* v___y_730_, lean_object* v___y_731_, lean_object* v___y_732_){
_start:
{
lean_object* v_res_733_; 
v_res_733_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_subst_spec__2___redArg(v_x_726_, v___y_727_, v___y_728_, v___y_729_, v___y_730_, v___y_731_);
lean_dec(v___y_731_);
lean_dec_ref(v___y_730_);
lean_dec(v___y_729_);
lean_dec_ref(v___y_728_);
lean_dec(v___y_727_);
return v_res_733_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_subst_spec__2(lean_object* v_00_u03b1_734_, lean_object* v_x_735_, lean_object* v___y_736_, lean_object* v___y_737_, lean_object* v___y_738_, lean_object* v___y_739_, lean_object* v___y_740_){
_start:
{
lean_object* v___x_742_; 
v___x_742_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_subst_spec__2___redArg(v_x_735_, v___y_736_, v___y_737_, v___y_738_, v___y_739_, v___y_740_);
return v___x_742_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_subst_spec__2___boxed(lean_object* v_00_u03b1_743_, lean_object* v_x_744_, lean_object* v___y_745_, lean_object* v___y_746_, lean_object* v___y_747_, lean_object* v___y_748_, lean_object* v___y_749_, lean_object* v___y_750_){
_start:
{
lean_object* v_res_751_; 
v_res_751_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_subst_spec__2(v_00_u03b1_743_, v_x_744_, v___y_745_, v___y_746_, v___y_747_, v___y_748_, v___y_749_);
lean_dec(v___y_749_);
lean_dec_ref(v___y_748_);
lean_dec(v___y_747_);
lean_dec_ref(v___y_746_);
lean_dec(v___y_745_);
return v_res_751_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_subst_spec__3___redArg___lam__0(lean_object* v_x_752_, lean_object* v___y_753_, lean_object* v___y_754_, lean_object* v___y_755_, lean_object* v___y_756_, lean_object* v___y_757_){
_start:
{
lean_object* v___x_759_; 
lean_inc(v___y_753_);
v___x_759_ = lean_apply_6(v_x_752_, v___y_753_, v___y_754_, v___y_755_, v___y_756_, v___y_757_, lean_box(0));
return v___x_759_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_subst_spec__3___redArg___lam__0___boxed(lean_object* v_x_760_, lean_object* v___y_761_, lean_object* v___y_762_, lean_object* v___y_763_, lean_object* v___y_764_, lean_object* v___y_765_, lean_object* v___y_766_){
_start:
{
lean_object* v_res_767_; 
v_res_767_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_subst_spec__3___redArg___lam__0(v_x_760_, v___y_761_, v___y_762_, v___y_763_, v___y_764_, v___y_765_);
lean_dec(v___y_761_);
return v_res_767_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_subst_spec__3___redArg(lean_object* v_mvarId_768_, lean_object* v_x_769_, lean_object* v___y_770_, lean_object* v___y_771_, lean_object* v___y_772_, lean_object* v___y_773_, lean_object* v___y_774_){
_start:
{
lean_object* v___f_776_; lean_object* v___x_777_; 
lean_inc(v___y_770_);
v___f_776_ = lean_alloc_closure((void*)(lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_subst_spec__3___redArg___lam__0___boxed), 7, 2);
lean_closure_set(v___f_776_, 0, v_x_769_);
lean_closure_set(v___f_776_, 1, v___y_770_);
v___x_777_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_768_, v___f_776_, v___y_771_, v___y_772_, v___y_773_, v___y_774_);
if (lean_obj_tag(v___x_777_) == 0)
{
return v___x_777_;
}
else
{
lean_object* v_a_778_; lean_object* v___x_780_; uint8_t v_isShared_781_; uint8_t v_isSharedCheck_785_; 
v_a_778_ = lean_ctor_get(v___x_777_, 0);
v_isSharedCheck_785_ = !lean_is_exclusive(v___x_777_);
if (v_isSharedCheck_785_ == 0)
{
v___x_780_ = v___x_777_;
v_isShared_781_ = v_isSharedCheck_785_;
goto v_resetjp_779_;
}
else
{
lean_inc(v_a_778_);
lean_dec(v___x_777_);
v___x_780_ = lean_box(0);
v_isShared_781_ = v_isSharedCheck_785_;
goto v_resetjp_779_;
}
v_resetjp_779_:
{
lean_object* v___x_783_; 
if (v_isShared_781_ == 0)
{
v___x_783_ = v___x_780_;
goto v_reusejp_782_;
}
else
{
lean_object* v_reuseFailAlloc_784_; 
v_reuseFailAlloc_784_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_784_, 0, v_a_778_);
v___x_783_ = v_reuseFailAlloc_784_;
goto v_reusejp_782_;
}
v_reusejp_782_:
{
return v___x_783_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_subst_spec__3___redArg___boxed(lean_object* v_mvarId_786_, lean_object* v_x_787_, lean_object* v___y_788_, lean_object* v___y_789_, lean_object* v___y_790_, lean_object* v___y_791_, lean_object* v___y_792_, lean_object* v___y_793_){
_start:
{
lean_object* v_res_794_; 
v_res_794_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_subst_spec__3___redArg(v_mvarId_786_, v_x_787_, v___y_788_, v___y_789_, v___y_790_, v___y_791_, v___y_792_);
lean_dec(v___y_792_);
lean_dec_ref(v___y_791_);
lean_dec(v___y_790_);
lean_dec_ref(v___y_789_);
lean_dec(v___y_788_);
return v_res_794_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_subst_spec__3(lean_object* v_00_u03b1_795_, lean_object* v_mvarId_796_, lean_object* v_x_797_, lean_object* v___y_798_, lean_object* v___y_799_, lean_object* v___y_800_, lean_object* v___y_801_, lean_object* v___y_802_){
_start:
{
lean_object* v___x_804_; 
v___x_804_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_subst_spec__3___redArg(v_mvarId_796_, v_x_797_, v___y_798_, v___y_799_, v___y_800_, v___y_801_, v___y_802_);
return v___x_804_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_subst_spec__3___boxed(lean_object* v_00_u03b1_805_, lean_object* v_mvarId_806_, lean_object* v_x_807_, lean_object* v___y_808_, lean_object* v___y_809_, lean_object* v___y_810_, lean_object* v___y_811_, lean_object* v___y_812_, lean_object* v___y_813_){
_start:
{
lean_object* v_res_814_; 
v_res_814_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_subst_spec__3(v_00_u03b1_805_, v_mvarId_806_, v_x_807_, v___y_808_, v___y_809_, v___y_810_, v___y_811_, v___y_812_);
lean_dec(v___y_812_);
lean_dec_ref(v___y_811_);
lean_dec(v___y_810_);
lean_dec_ref(v___y_809_);
lean_dec(v___y_808_);
return v_res_814_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_BuiltinRules_subst_spec__0_spec__0(lean_object* v_msgData_815_, lean_object* v___y_816_, lean_object* v___y_817_, lean_object* v___y_818_, lean_object* v___y_819_){
_start:
{
lean_object* v___x_821_; lean_object* v_env_822_; lean_object* v___x_823_; lean_object* v_mctx_824_; lean_object* v_lctx_825_; lean_object* v_options_826_; lean_object* v___x_827_; lean_object* v___x_828_; lean_object* v___x_829_; 
v___x_821_ = lean_st_ref_get(v___y_819_);
v_env_822_ = lean_ctor_get(v___x_821_, 0);
lean_inc_ref(v_env_822_);
lean_dec(v___x_821_);
v___x_823_ = lean_st_ref_get(v___y_817_);
v_mctx_824_ = lean_ctor_get(v___x_823_, 0);
lean_inc_ref(v_mctx_824_);
lean_dec(v___x_823_);
v_lctx_825_ = lean_ctor_get(v___y_816_, 2);
v_options_826_ = lean_ctor_get(v___y_818_, 2);
lean_inc_ref(v_options_826_);
lean_inc_ref(v_lctx_825_);
v___x_827_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_827_, 0, v_env_822_);
lean_ctor_set(v___x_827_, 1, v_mctx_824_);
lean_ctor_set(v___x_827_, 2, v_lctx_825_);
lean_ctor_set(v___x_827_, 3, v_options_826_);
v___x_828_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_828_, 0, v___x_827_);
lean_ctor_set(v___x_828_, 1, v_msgData_815_);
v___x_829_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_829_, 0, v___x_828_);
return v___x_829_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_BuiltinRules_subst_spec__0_spec__0___boxed(lean_object* v_msgData_830_, lean_object* v___y_831_, lean_object* v___y_832_, lean_object* v___y_833_, lean_object* v___y_834_, lean_object* v___y_835_){
_start:
{
lean_object* v_res_836_; 
v_res_836_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_BuiltinRules_subst_spec__0_spec__0(v_msgData_830_, v___y_831_, v___y_832_, v___y_833_, v___y_834_);
lean_dec(v___y_834_);
lean_dec_ref(v___y_833_);
lean_dec(v___y_832_);
lean_dec_ref(v___y_831_);
return v_res_836_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_subst_spec__0___redArg(lean_object* v_msg_837_, lean_object* v___y_838_, lean_object* v___y_839_, lean_object* v___y_840_, lean_object* v___y_841_){
_start:
{
lean_object* v_ref_843_; lean_object* v___x_844_; lean_object* v_a_845_; lean_object* v___x_847_; uint8_t v_isShared_848_; uint8_t v_isSharedCheck_853_; 
v_ref_843_ = lean_ctor_get(v___y_840_, 5);
v___x_844_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_BuiltinRules_subst_spec__0_spec__0(v_msg_837_, v___y_838_, v___y_839_, v___y_840_, v___y_841_);
v_a_845_ = lean_ctor_get(v___x_844_, 0);
v_isSharedCheck_853_ = !lean_is_exclusive(v___x_844_);
if (v_isSharedCheck_853_ == 0)
{
v___x_847_ = v___x_844_;
v_isShared_848_ = v_isSharedCheck_853_;
goto v_resetjp_846_;
}
else
{
lean_inc(v_a_845_);
lean_dec(v___x_844_);
v___x_847_ = lean_box(0);
v_isShared_848_ = v_isSharedCheck_853_;
goto v_resetjp_846_;
}
v_resetjp_846_:
{
lean_object* v___x_849_; lean_object* v___x_851_; 
lean_inc(v_ref_843_);
v___x_849_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_849_, 0, v_ref_843_);
lean_ctor_set(v___x_849_, 1, v_a_845_);
if (v_isShared_848_ == 0)
{
lean_ctor_set_tag(v___x_847_, 1);
lean_ctor_set(v___x_847_, 0, v___x_849_);
v___x_851_ = v___x_847_;
goto v_reusejp_850_;
}
else
{
lean_object* v_reuseFailAlloc_852_; 
v_reuseFailAlloc_852_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_852_, 0, v___x_849_);
v___x_851_ = v_reuseFailAlloc_852_;
goto v_reusejp_850_;
}
v_reusejp_850_:
{
return v___x_851_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_subst_spec__0___redArg___boxed(lean_object* v_msg_854_, lean_object* v___y_855_, lean_object* v___y_856_, lean_object* v___y_857_, lean_object* v___y_858_, lean_object* v___y_859_){
_start:
{
lean_object* v_res_860_; 
v_res_860_ = lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_subst_spec__0___redArg(v_msg_854_, v___y_855_, v___y_856_, v___y_857_, v___y_858_);
lean_dec(v___y_858_);
lean_dec_ref(v___y_857_);
lean_dec(v___y_856_);
lean_dec_ref(v___y_855_);
return v_res_860_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_subst_spec__1___closed__1(void){
_start:
{
lean_object* v___x_862_; lean_object* v___x_863_; 
v___x_862_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_subst_spec__1___closed__0));
v___x_863_ = l_Lean_stringToMessageData(v___x_862_);
return v___x_863_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_subst_spec__1(size_t v_sz_864_, size_t v_i_865_, lean_object* v_bs_866_, lean_object* v___y_867_, lean_object* v___y_868_, lean_object* v___y_869_, lean_object* v___y_870_, lean_object* v___y_871_){
_start:
{
uint8_t v___x_873_; 
v___x_873_ = lean_usize_dec_lt(v_i_865_, v_sz_864_);
if (v___x_873_ == 0)
{
lean_object* v___x_874_; 
v___x_874_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_874_, 0, v_bs_866_);
return v___x_874_;
}
else
{
lean_object* v_v_875_; lean_object* v___x_876_; lean_object* v_bs_x27_877_; lean_object* v_a_879_; 
v_v_875_ = lean_array_uget(v_bs_866_, v_i_865_);
v___x_876_ = lean_unsigned_to_nat(0u);
v_bs_x27_877_ = lean_array_uset(v_bs_866_, v_i_865_, v___x_876_);
if (lean_obj_tag(v_v_875_) == 2)
{
lean_object* v_ldecl_884_; lean_object* v___x_885_; 
v_ldecl_884_ = lean_ctor_get(v_v_875_, 0);
lean_inc_ref(v_ldecl_884_);
lean_dec_ref_known(v_v_875_, 1);
v___x_885_ = l_Lean_LocalDecl_fvarId(v_ldecl_884_);
lean_dec_ref(v_ldecl_884_);
v_a_879_ = v___x_885_;
goto v___jp_878_;
}
else
{
lean_object* v___x_886_; lean_object* v___x_887_; 
lean_dec(v_v_875_);
v___x_886_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_subst_spec__1___closed__1, &lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_subst_spec__1___closed__1_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_subst_spec__1___closed__1);
v___x_887_ = lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_subst_spec__0___redArg(v___x_886_, v___y_868_, v___y_869_, v___y_870_, v___y_871_);
if (lean_obj_tag(v___x_887_) == 0)
{
lean_object* v_a_888_; 
v_a_888_ = lean_ctor_get(v___x_887_, 0);
lean_inc(v_a_888_);
lean_dec_ref_known(v___x_887_, 1);
v_a_879_ = v_a_888_;
goto v___jp_878_;
}
else
{
lean_object* v_a_889_; lean_object* v___x_891_; uint8_t v_isShared_892_; uint8_t v_isSharedCheck_896_; 
lean_dec_ref(v_bs_x27_877_);
v_a_889_ = lean_ctor_get(v___x_887_, 0);
v_isSharedCheck_896_ = !lean_is_exclusive(v___x_887_);
if (v_isSharedCheck_896_ == 0)
{
v___x_891_ = v___x_887_;
v_isShared_892_ = v_isSharedCheck_896_;
goto v_resetjp_890_;
}
else
{
lean_inc(v_a_889_);
lean_dec(v___x_887_);
v___x_891_ = lean_box(0);
v_isShared_892_ = v_isSharedCheck_896_;
goto v_resetjp_890_;
}
v_resetjp_890_:
{
lean_object* v___x_894_; 
if (v_isShared_892_ == 0)
{
v___x_894_ = v___x_891_;
goto v_reusejp_893_;
}
else
{
lean_object* v_reuseFailAlloc_895_; 
v_reuseFailAlloc_895_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_895_, 0, v_a_889_);
v___x_894_ = v_reuseFailAlloc_895_;
goto v_reusejp_893_;
}
v_reusejp_893_:
{
return v___x_894_;
}
}
}
}
v___jp_878_:
{
size_t v___x_880_; size_t v___x_881_; lean_object* v___x_882_; 
v___x_880_ = ((size_t)1ULL);
v___x_881_ = lean_usize_add(v_i_865_, v___x_880_);
v___x_882_ = lean_array_uset(v_bs_x27_877_, v_i_865_, v_a_879_);
v_i_865_ = v___x_881_;
v_bs_866_ = v___x_882_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_subst_spec__1___boxed(lean_object* v_sz_897_, lean_object* v_i_898_, lean_object* v_bs_899_, lean_object* v___y_900_, lean_object* v___y_901_, lean_object* v___y_902_, lean_object* v___y_903_, lean_object* v___y_904_, lean_object* v___y_905_){
_start:
{
size_t v_sz_boxed_906_; size_t v_i_boxed_907_; lean_object* v_res_908_; 
v_sz_boxed_906_ = lean_unbox_usize(v_sz_897_);
lean_dec(v_sz_897_);
v_i_boxed_907_ = lean_unbox_usize(v_i_898_);
lean_dec(v_i_898_);
v_res_908_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_subst_spec__1(v_sz_boxed_906_, v_i_boxed_907_, v_bs_899_, v___y_900_, v___y_901_, v___y_902_, v___y_903_, v___y_904_);
lean_dec(v___y_904_);
lean_dec_ref(v___y_903_);
lean_dec(v___y_902_);
lean_dec_ref(v___y_901_);
lean_dec(v___y_900_);
return v_res_908_;
}
}
static lean_object* _init_lp_aesop_Aesop_BuiltinRules_subst___lam__0___closed__1(void){
_start:
{
lean_object* v___x_910_; lean_object* v___x_911_; 
v___x_910_ = ((lean_object*)(lp_aesop_Aesop_BuiltinRules_subst___lam__0___closed__0));
v___x_911_ = l_Lean_stringToMessageData(v___x_910_);
return v___x_911_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_subst___lam__0(size_t v_sz_912_, size_t v___x_913_, lean_object* v_indexMatchLocations_914_, lean_object* v_goal_915_, lean_object* v___y_916_, lean_object* v___y_917_, lean_object* v___y_918_, lean_object* v___y_919_, lean_object* v___y_920_){
_start:
{
lean_object* v___x_922_; 
v___x_922_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_subst_spec__1(v_sz_912_, v___x_913_, v_indexMatchLocations_914_, v___y_916_, v___y_917_, v___y_918_, v___y_919_, v___y_920_);
if (lean_obj_tag(v___x_922_) == 0)
{
lean_object* v_a_923_; lean_object* v___x_924_; lean_object* v___x_925_; 
v_a_923_ = lean_ctor_get(v___x_922_, 0);
lean_inc(v_a_923_);
lean_dec_ref_known(v___x_922_, 1);
lean_inc(v_goal_915_);
v___x_924_ = lean_alloc_closure((void*)(lp_aesop_Aesop_BuiltinRules_substEqsAndIffs_x3f___boxed), 9, 2);
lean_closure_set(v___x_924_, 0, v_goal_915_);
lean_closure_set(v___x_924_, 1, v_a_923_);
v___x_925_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_subst_spec__2___redArg(v___x_924_, v___y_916_, v___y_917_, v___y_918_, v___y_919_, v___y_920_);
if (lean_obj_tag(v___x_925_) == 0)
{
lean_object* v_a_926_; lean_object* v_fst_927_; 
v_a_926_ = lean_ctor_get(v___x_925_, 0);
lean_inc(v_a_926_);
lean_dec_ref_known(v___x_925_, 1);
v_fst_927_ = lean_ctor_get(v_a_926_, 0);
lean_inc(v_fst_927_);
if (lean_obj_tag(v_fst_927_) == 1)
{
lean_object* v_snd_928_; lean_object* v___x_930_; uint8_t v_isShared_931_; uint8_t v_isSharedCheck_965_; 
v_snd_928_ = lean_ctor_get(v_a_926_, 1);
v_isSharedCheck_965_ = !lean_is_exclusive(v_a_926_);
if (v_isSharedCheck_965_ == 0)
{
lean_object* v_unused_966_; 
v_unused_966_ = lean_ctor_get(v_a_926_, 0);
lean_dec(v_unused_966_);
v___x_930_ = v_a_926_;
v_isShared_931_ = v_isSharedCheck_965_;
goto v_resetjp_929_;
}
else
{
lean_inc(v_snd_928_);
lean_dec(v_a_926_);
v___x_930_ = lean_box(0);
v_isShared_931_ = v_isSharedCheck_965_;
goto v_resetjp_929_;
}
v_resetjp_929_:
{
lean_object* v_val_932_; lean_object* v___x_934_; uint8_t v_isShared_935_; uint8_t v_isSharedCheck_964_; 
v_val_932_ = lean_ctor_get(v_fst_927_, 0);
v_isSharedCheck_964_ = !lean_is_exclusive(v_fst_927_);
if (v_isSharedCheck_964_ == 0)
{
v___x_934_ = v_fst_927_;
v_isShared_935_ = v_isSharedCheck_964_;
goto v_resetjp_933_;
}
else
{
lean_inc(v_val_932_);
lean_dec(v_fst_927_);
v___x_934_ = lean_box(0);
v_isShared_935_ = v_isSharedCheck_964_;
goto v_resetjp_933_;
}
v_resetjp_933_:
{
lean_object* v___x_936_; 
v___x_936_ = lp_aesop_Aesop_mvarIdToSubgoal(v_goal_915_, v_val_932_, v___y_916_, v___y_917_, v___y_918_, v___y_919_, v___y_920_);
if (lean_obj_tag(v___x_936_) == 0)
{
lean_object* v_a_937_; lean_object* v___x_939_; uint8_t v_isShared_940_; uint8_t v_isSharedCheck_955_; 
v_a_937_ = lean_ctor_get(v___x_936_, 0);
v_isSharedCheck_955_ = !lean_is_exclusive(v___x_936_);
if (v_isSharedCheck_955_ == 0)
{
v___x_939_ = v___x_936_;
v_isShared_940_ = v_isSharedCheck_955_;
goto v_resetjp_938_;
}
else
{
lean_inc(v_a_937_);
lean_dec(v___x_936_);
v___x_939_ = lean_box(0);
v_isShared_940_ = v_isSharedCheck_955_;
goto v_resetjp_938_;
}
v_resetjp_938_:
{
lean_object* v___x_941_; lean_object* v___x_942_; lean_object* v___x_943_; lean_object* v___x_945_; 
v___x_941_ = lean_unsigned_to_nat(1u);
v___x_942_ = lean_mk_empty_array_with_capacity(v___x_941_);
v___x_943_ = lean_array_push(v___x_942_, v_a_937_);
if (v_isShared_935_ == 0)
{
lean_ctor_set(v___x_934_, 0, v_snd_928_);
v___x_945_ = v___x_934_;
goto v_reusejp_944_;
}
else
{
lean_object* v_reuseFailAlloc_954_; 
v_reuseFailAlloc_954_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_954_, 0, v_snd_928_);
v___x_945_ = v_reuseFailAlloc_954_;
goto v_reusejp_944_;
}
v_reusejp_944_:
{
lean_object* v___x_946_; lean_object* v___x_948_; 
v___x_946_ = lean_box(0);
if (v_isShared_931_ == 0)
{
lean_ctor_set(v___x_930_, 1, v___x_946_);
lean_ctor_set(v___x_930_, 0, v___x_945_);
v___x_948_ = v___x_930_;
goto v_reusejp_947_;
}
else
{
lean_object* v_reuseFailAlloc_953_; 
v_reuseFailAlloc_953_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_953_, 0, v___x_945_);
lean_ctor_set(v_reuseFailAlloc_953_, 1, v___x_946_);
v___x_948_ = v_reuseFailAlloc_953_;
goto v_reusejp_947_;
}
v_reusejp_947_:
{
lean_object* v___x_949_; lean_object* v___x_951_; 
v___x_949_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_949_, 0, v___x_943_);
lean_ctor_set(v___x_949_, 1, v___x_948_);
if (v_isShared_940_ == 0)
{
lean_ctor_set(v___x_939_, 0, v___x_949_);
v___x_951_ = v___x_939_;
goto v_reusejp_950_;
}
else
{
lean_object* v_reuseFailAlloc_952_; 
v_reuseFailAlloc_952_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_952_, 0, v___x_949_);
v___x_951_ = v_reuseFailAlloc_952_;
goto v_reusejp_950_;
}
v_reusejp_950_:
{
return v___x_951_;
}
}
}
}
}
else
{
lean_object* v_a_956_; lean_object* v___x_958_; uint8_t v_isShared_959_; uint8_t v_isSharedCheck_963_; 
lean_del_object(v___x_934_);
lean_del_object(v___x_930_);
lean_dec(v_snd_928_);
v_a_956_ = lean_ctor_get(v___x_936_, 0);
v_isSharedCheck_963_ = !lean_is_exclusive(v___x_936_);
if (v_isSharedCheck_963_ == 0)
{
v___x_958_ = v___x_936_;
v_isShared_959_ = v_isSharedCheck_963_;
goto v_resetjp_957_;
}
else
{
lean_inc(v_a_956_);
lean_dec(v___x_936_);
v___x_958_ = lean_box(0);
v_isShared_959_ = v_isSharedCheck_963_;
goto v_resetjp_957_;
}
v_resetjp_957_:
{
lean_object* v___x_961_; 
if (v_isShared_959_ == 0)
{
v___x_961_ = v___x_958_;
goto v_reusejp_960_;
}
else
{
lean_object* v_reuseFailAlloc_962_; 
v_reuseFailAlloc_962_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_962_, 0, v_a_956_);
v___x_961_ = v_reuseFailAlloc_962_;
goto v_reusejp_960_;
}
v_reusejp_960_:
{
return v___x_961_;
}
}
}
}
}
}
else
{
lean_object* v___x_967_; lean_object* v___x_968_; 
lean_dec(v_fst_927_);
lean_dec(v_a_926_);
lean_dec(v_goal_915_);
v___x_967_ = lean_obj_once(&lp_aesop_Aesop_BuiltinRules_subst___lam__0___closed__1, &lp_aesop_Aesop_BuiltinRules_subst___lam__0___closed__1_once, _init_lp_aesop_Aesop_BuiltinRules_subst___lam__0___closed__1);
v___x_968_ = lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_subst_spec__0___redArg(v___x_967_, v___y_917_, v___y_918_, v___y_919_, v___y_920_);
return v___x_968_;
}
}
else
{
lean_object* v_a_969_; lean_object* v___x_971_; uint8_t v_isShared_972_; uint8_t v_isSharedCheck_976_; 
lean_dec(v_goal_915_);
v_a_969_ = lean_ctor_get(v___x_925_, 0);
v_isSharedCheck_976_ = !lean_is_exclusive(v___x_925_);
if (v_isSharedCheck_976_ == 0)
{
v___x_971_ = v___x_925_;
v_isShared_972_ = v_isSharedCheck_976_;
goto v_resetjp_970_;
}
else
{
lean_inc(v_a_969_);
lean_dec(v___x_925_);
v___x_971_ = lean_box(0);
v_isShared_972_ = v_isSharedCheck_976_;
goto v_resetjp_970_;
}
v_resetjp_970_:
{
lean_object* v___x_974_; 
if (v_isShared_972_ == 0)
{
v___x_974_ = v___x_971_;
goto v_reusejp_973_;
}
else
{
lean_object* v_reuseFailAlloc_975_; 
v_reuseFailAlloc_975_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_975_, 0, v_a_969_);
v___x_974_ = v_reuseFailAlloc_975_;
goto v_reusejp_973_;
}
v_reusejp_973_:
{
return v___x_974_;
}
}
}
}
else
{
lean_object* v_a_977_; lean_object* v___x_979_; uint8_t v_isShared_980_; uint8_t v_isSharedCheck_984_; 
lean_dec(v_goal_915_);
v_a_977_ = lean_ctor_get(v___x_922_, 0);
v_isSharedCheck_984_ = !lean_is_exclusive(v___x_922_);
if (v_isSharedCheck_984_ == 0)
{
v___x_979_ = v___x_922_;
v_isShared_980_ = v_isSharedCheck_984_;
goto v_resetjp_978_;
}
else
{
lean_inc(v_a_977_);
lean_dec(v___x_922_);
v___x_979_ = lean_box(0);
v_isShared_980_ = v_isSharedCheck_984_;
goto v_resetjp_978_;
}
v_resetjp_978_:
{
lean_object* v___x_982_; 
if (v_isShared_980_ == 0)
{
v___x_982_ = v___x_979_;
goto v_reusejp_981_;
}
else
{
lean_object* v_reuseFailAlloc_983_; 
v_reuseFailAlloc_983_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_983_, 0, v_a_977_);
v___x_982_ = v_reuseFailAlloc_983_;
goto v_reusejp_981_;
}
v_reusejp_981_:
{
return v___x_982_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_subst___lam__0___boxed(lean_object* v_sz_985_, lean_object* v___x_986_, lean_object* v_indexMatchLocations_987_, lean_object* v_goal_988_, lean_object* v___y_989_, lean_object* v___y_990_, lean_object* v___y_991_, lean_object* v___y_992_, lean_object* v___y_993_, lean_object* v___y_994_){
_start:
{
size_t v_sz_boxed_995_; size_t v___x_4339__boxed_996_; lean_object* v_res_997_; 
v_sz_boxed_995_ = lean_unbox_usize(v_sz_985_);
lean_dec(v_sz_985_);
v___x_4339__boxed_996_ = lean_unbox_usize(v___x_986_);
lean_dec(v___x_986_);
v_res_997_ = lp_aesop_Aesop_BuiltinRules_subst___lam__0(v_sz_boxed_995_, v___x_4339__boxed_996_, v_indexMatchLocations_987_, v_goal_988_, v___y_989_, v___y_990_, v___y_991_, v___y_992_, v___y_993_);
lean_dec(v___y_993_);
lean_dec_ref(v___y_992_);
lean_dec(v___y_991_);
lean_dec_ref(v___y_990_);
lean_dec(v___y_989_);
return v_res_997_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_subst(lean_object* v_a_998_, lean_object* v_a_999_, lean_object* v_a_1000_, lean_object* v_a_1001_, lean_object* v_a_1002_, lean_object* v_a_1003_){
_start:
{
lean_object* v_goal_1005_; lean_object* v_indexMatchLocations_1006_; size_t v_sz_1007_; lean_object* v___x_1008_; lean_object* v___x_1009_; lean_object* v___f_1010_; lean_object* v___x_1011_; 
v_goal_1005_ = lean_ctor_get(v_a_998_, 0);
lean_inc_n(v_goal_1005_, 2);
v_indexMatchLocations_1006_ = lean_ctor_get(v_a_998_, 2);
lean_inc_ref(v_indexMatchLocations_1006_);
lean_dec_ref(v_a_998_);
v_sz_1007_ = lean_array_size(v_indexMatchLocations_1006_);
v___x_1008_ = lean_box_usize(v_sz_1007_);
v___x_1009_ = ((lean_object*)(lp_aesop_Aesop_BuiltinRules_substEqs_x3f___boxed__const__1));
v___f_1010_ = lean_alloc_closure((void*)(lp_aesop_Aesop_BuiltinRules_subst___lam__0___boxed), 10, 4);
lean_closure_set(v___f_1010_, 0, v___x_1008_);
lean_closure_set(v___f_1010_, 1, v___x_1009_);
lean_closure_set(v___f_1010_, 2, v_indexMatchLocations_1006_);
lean_closure_set(v___f_1010_, 3, v_goal_1005_);
v___x_1011_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_BuiltinRules_subst_spec__3___redArg(v_goal_1005_, v___f_1010_, v_a_999_, v_a_1000_, v_a_1001_, v_a_1002_, v_a_1003_);
if (lean_obj_tag(v___x_1011_) == 0)
{
lean_object* v_a_1012_; lean_object* v_snd_1013_; lean_object* v_fst_1014_; lean_object* v_fst_1015_; lean_object* v_snd_1016_; lean_object* v___x_1017_; 
v_a_1012_ = lean_ctor_get(v___x_1011_, 0);
lean_inc(v_a_1012_);
lean_dec_ref_known(v___x_1011_, 1);
v_snd_1013_ = lean_ctor_get(v_a_1012_, 1);
lean_inc(v_snd_1013_);
v_fst_1014_ = lean_ctor_get(v_a_1012_, 0);
lean_inc(v_fst_1014_);
lean_dec(v_a_1012_);
v_fst_1015_ = lean_ctor_get(v_snd_1013_, 0);
lean_inc(v_fst_1015_);
v_snd_1016_ = lean_ctor_get(v_snd_1013_, 1);
lean_inc(v_snd_1016_);
lean_dec(v_snd_1013_);
v___x_1017_ = l_Lean_Meta_saveState___redArg(v_a_1001_, v_a_1003_);
if (lean_obj_tag(v___x_1017_) == 0)
{
lean_object* v_a_1018_; lean_object* v___x_1020_; uint8_t v_isShared_1021_; uint8_t v_isSharedCheck_1029_; 
v_a_1018_ = lean_ctor_get(v___x_1017_, 0);
v_isSharedCheck_1029_ = !lean_is_exclusive(v___x_1017_);
if (v_isSharedCheck_1029_ == 0)
{
v___x_1020_ = v___x_1017_;
v_isShared_1021_ = v_isSharedCheck_1029_;
goto v_resetjp_1019_;
}
else
{
lean_inc(v_a_1018_);
lean_dec(v___x_1017_);
v___x_1020_ = lean_box(0);
v_isShared_1021_ = v_isSharedCheck_1029_;
goto v_resetjp_1019_;
}
v_resetjp_1019_:
{
lean_object* v___x_1022_; lean_object* v___x_1023_; lean_object* v___x_1024_; lean_object* v___x_1025_; lean_object* v___x_1027_; 
v___x_1022_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1022_, 0, v_fst_1014_);
lean_ctor_set(v___x_1022_, 1, v_a_1018_);
lean_ctor_set(v___x_1022_, 2, v_fst_1015_);
lean_ctor_set(v___x_1022_, 3, v_snd_1016_);
v___x_1023_ = lean_unsigned_to_nat(1u);
v___x_1024_ = lean_mk_empty_array_with_capacity(v___x_1023_);
v___x_1025_ = lean_array_push(v___x_1024_, v___x_1022_);
if (v_isShared_1021_ == 0)
{
lean_ctor_set(v___x_1020_, 0, v___x_1025_);
v___x_1027_ = v___x_1020_;
goto v_reusejp_1026_;
}
else
{
lean_object* v_reuseFailAlloc_1028_; 
v_reuseFailAlloc_1028_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1028_, 0, v___x_1025_);
v___x_1027_ = v_reuseFailAlloc_1028_;
goto v_reusejp_1026_;
}
v_reusejp_1026_:
{
return v___x_1027_;
}
}
}
else
{
lean_object* v_a_1030_; lean_object* v___x_1032_; uint8_t v_isShared_1033_; uint8_t v_isSharedCheck_1037_; 
lean_dec(v_snd_1016_);
lean_dec(v_fst_1015_);
lean_dec(v_fst_1014_);
v_a_1030_ = lean_ctor_get(v___x_1017_, 0);
v_isSharedCheck_1037_ = !lean_is_exclusive(v___x_1017_);
if (v_isSharedCheck_1037_ == 0)
{
v___x_1032_ = v___x_1017_;
v_isShared_1033_ = v_isSharedCheck_1037_;
goto v_resetjp_1031_;
}
else
{
lean_inc(v_a_1030_);
lean_dec(v___x_1017_);
v___x_1032_ = lean_box(0);
v_isShared_1033_ = v_isSharedCheck_1037_;
goto v_resetjp_1031_;
}
v_resetjp_1031_:
{
lean_object* v___x_1035_; 
if (v_isShared_1033_ == 0)
{
v___x_1035_ = v___x_1032_;
goto v_reusejp_1034_;
}
else
{
lean_object* v_reuseFailAlloc_1036_; 
v_reuseFailAlloc_1036_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1036_, 0, v_a_1030_);
v___x_1035_ = v_reuseFailAlloc_1036_;
goto v_reusejp_1034_;
}
v_reusejp_1034_:
{
return v___x_1035_;
}
}
}
}
else
{
lean_object* v_a_1038_; lean_object* v___x_1040_; uint8_t v_isShared_1041_; uint8_t v_isSharedCheck_1045_; 
v_a_1038_ = lean_ctor_get(v___x_1011_, 0);
v_isSharedCheck_1045_ = !lean_is_exclusive(v___x_1011_);
if (v_isSharedCheck_1045_ == 0)
{
v___x_1040_ = v___x_1011_;
v_isShared_1041_ = v_isSharedCheck_1045_;
goto v_resetjp_1039_;
}
else
{
lean_inc(v_a_1038_);
lean_dec(v___x_1011_);
v___x_1040_ = lean_box(0);
v_isShared_1041_ = v_isSharedCheck_1045_;
goto v_resetjp_1039_;
}
v_resetjp_1039_:
{
lean_object* v___x_1043_; 
if (v_isShared_1041_ == 0)
{
v___x_1043_ = v___x_1040_;
goto v_reusejp_1042_;
}
else
{
lean_object* v_reuseFailAlloc_1044_; 
v_reuseFailAlloc_1044_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1044_, 0, v_a_1038_);
v___x_1043_ = v_reuseFailAlloc_1044_;
goto v_reusejp_1042_;
}
v_reusejp_1042_:
{
return v___x_1043_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_subst___boxed(lean_object* v_a_1046_, lean_object* v_a_1047_, lean_object* v_a_1048_, lean_object* v_a_1049_, lean_object* v_a_1050_, lean_object* v_a_1051_, lean_object* v_a_1052_){
_start:
{
lean_object* v_res_1053_; 
v_res_1053_ = lp_aesop_Aesop_BuiltinRules_subst(v_a_1046_, v_a_1047_, v_a_1048_, v_a_1049_, v_a_1050_, v_a_1051_);
lean_dec(v_a_1051_);
lean_dec_ref(v_a_1050_);
lean_dec(v_a_1049_);
lean_dec_ref(v_a_1048_);
lean_dec(v_a_1047_);
return v_res_1053_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_subst_spec__0(lean_object* v_00_u03b1_1054_, lean_object* v_msg_1055_, lean_object* v___y_1056_, lean_object* v___y_1057_, lean_object* v___y_1058_, lean_object* v___y_1059_, lean_object* v___y_1060_){
_start:
{
lean_object* v___x_1062_; 
v___x_1062_ = lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_subst_spec__0___redArg(v_msg_1055_, v___y_1057_, v___y_1058_, v___y_1059_, v___y_1060_);
return v___x_1062_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_subst_spec__0___boxed(lean_object* v_00_u03b1_1063_, lean_object* v_msg_1064_, lean_object* v___y_1065_, lean_object* v___y_1066_, lean_object* v___y_1067_, lean_object* v___y_1068_, lean_object* v___y_1069_, lean_object* v___y_1070_){
_start:
{
lean_object* v_res_1071_; 
v_res_1071_ = lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_subst_spec__0(v_00_u03b1_1063_, v_msg_1064_, v___y_1065_, v___y_1066_, v___y_1067_, v___y_1068_, v___y_1069_);
lean_dec(v___y_1069_);
lean_dec_ref(v___y_1068_);
lean_dec(v___y_1067_);
lean_dec_ref(v___y_1066_);
lean_dec(v___y_1065_);
return v_res_1071_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Frontend_Attribute(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_BuiltinRules_Subst(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Frontend_Attribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_RuleTac_Forward_Basic(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_BuiltinRules_Subst(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleTac_Forward_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Frontend_Attribute(uint8_t builtin);
lean_object* initialize_aesop_Aesop_RuleTac_Forward_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_BuiltinRules_Subst(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Frontend_Attribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleTac_Forward_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_BuiltinRules_Subst(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_BuiltinRules_Subst(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_BuiltinRules_Subst(builtin);
}
#ifdef __cplusplus
}
#endif
