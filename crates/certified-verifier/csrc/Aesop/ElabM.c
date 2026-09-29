// Lean compiler output
// Module: Aesop.ElabM
// Imports: public import Init public meta import Init public import Lean.Elab.Term.TermElabM
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
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadEIO(lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVarAt(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
lean_object* l_StateRefT_x27_instMonad___redArg(lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_instMonadTermElabM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_instMonadTermElabM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonad___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabM_Context_forAdditionalRules(lean_object*);
static lean_once_cell_t lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__0;
static lean_once_cell_t lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__1;
static lean_once_cell_t lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__2;
static lean_once_cell_t lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__3;
static lean_once_cell_t lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__4;
static const lean_array_object lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__5 = (const lean_object*)&lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__5_value;
static const lean_string_object lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "True"};
static const lean_object* lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__6 = (const lean_object*)&lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__6_value;
static const lean_ctor_object lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__6_value),LEAN_SCALAR_PTR_LITERAL(78, 21, 103, 131, 118, 13, 187, 164)}};
static const lean_object* lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__7 = (const lean_object*)&lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__7_value;
static lean_once_cell_t lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__8;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabM_Context_forErasing(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabM_Context_forGlobalErasing(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabM_Context_forGlobalErasing___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_instMonadElabM___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instMonadElabM___closed__0;
static lean_once_cell_t lp_aesop_Aesop_instMonadElabM___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instMonadElabM___closed__1;
static const lean_closure_object lp_aesop_Aesop_instMonadElabM___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instMonadElabM___closed__2 = (const lean_object*)&lp_aesop_Aesop_instMonadElabM___closed__2_value;
static const lean_closure_object lp_aesop_Aesop_instMonadElabM___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instMonadElabM___closed__3 = (const lean_object*)&lp_aesop_Aesop_instMonadElabM___closed__3_value;
static const lean_closure_object lp_aesop_Aesop_instMonadElabM___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instMonadElabM___closed__4 = (const lean_object*)&lp_aesop_Aesop_instMonadElabM___closed__4_value;
static const lean_closure_object lp_aesop_Aesop_instMonadElabM___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instMonadElabM___closed__5 = (const lean_object*)&lp_aesop_Aesop_instMonadElabM___closed__5_value;
static const lean_closure_object lp_aesop_Aesop_instMonadElabM___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Elab_Term_instMonadTermElabM___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instMonadElabM___closed__6 = (const lean_object*)&lp_aesop_Aesop_instMonadElabM___closed__6_value;
static const lean_closure_object lp_aesop_Aesop_instMonadElabM___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Elab_Term_instMonadTermElabM___lam__1___boxed, .m_arity = 11, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instMonadElabM___closed__7 = (const lean_object*)&lp_aesop_Aesop_instMonadElabM___closed__7_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadElabM;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabM_run___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabM_run___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabM_run(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabM_run___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_shouldParsePriorities___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_shouldParsePriorities___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_shouldParsePriorities(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_shouldParsePriorities___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getGoal___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getGoal___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getGoal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getGoal___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabM_Context_forAdditionalRules(lean_object* v_goal_1_){
_start:
{
uint8_t v___x_2_; lean_object* v___x_3_; 
v___x_2_ = 1;
v___x_3_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_3_, 0, v_goal_1_);
lean_ctor_set_uint8(v___x_3_, sizeof(void*)*1, v___x_2_);
return v___x_3_;
}
}
static lean_object* _init_lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__0(void){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_4_;
}
}
static lean_object* _init_lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__1(void){
_start:
{
lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_5_ = lean_obj_once(&lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__0, &lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__0_once, _init_lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__0);
v___x_6_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_6_, 0, v___x_5_);
return v___x_6_;
}
}
static lean_object* _init_lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__2(void){
_start:
{
lean_object* v___x_7_; lean_object* v___x_8_; lean_object* v___x_9_; 
v___x_7_ = lean_unsigned_to_nat(32u);
v___x_8_ = lean_mk_empty_array_with_capacity(v___x_7_);
v___x_9_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_9_, 0, v___x_8_);
return v___x_9_;
}
}
static lean_object* _init_lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__3(void){
_start:
{
size_t v___x_10_; lean_object* v___x_11_; lean_object* v___x_12_; lean_object* v___x_13_; lean_object* v___x_14_; lean_object* v___x_15_; 
v___x_10_ = ((size_t)5ULL);
v___x_11_ = lean_unsigned_to_nat(0u);
v___x_12_ = lean_unsigned_to_nat(32u);
v___x_13_ = lean_mk_empty_array_with_capacity(v___x_12_);
v___x_14_ = lean_obj_once(&lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__2, &lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__2_once, _init_lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__2);
v___x_15_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_15_, 0, v___x_14_);
lean_ctor_set(v___x_15_, 1, v___x_13_);
lean_ctor_set(v___x_15_, 2, v___x_11_);
lean_ctor_set(v___x_15_, 3, v___x_11_);
lean_ctor_set_usize(v___x_15_, 4, v___x_10_);
return v___x_15_;
}
}
static lean_object* _init_lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__4(void){
_start:
{
lean_object* v___x_16_; lean_object* v___x_17_; lean_object* v___x_18_; lean_object* v___x_19_; 
v___x_16_ = lean_box(1);
v___x_17_ = lean_obj_once(&lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__3, &lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__3_once, _init_lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__3);
v___x_18_ = lean_obj_once(&lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__1, &lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__1_once, _init_lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__1);
v___x_19_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_19_, 0, v___x_18_);
lean_ctor_set(v___x_19_, 1, v___x_17_);
lean_ctor_set(v___x_19_, 2, v___x_16_);
return v___x_19_;
}
}
static lean_object* _init_lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__8(void){
_start:
{
lean_object* v___x_25_; lean_object* v___x_26_; lean_object* v___x_27_; 
v___x_25_ = lean_box(0);
v___x_26_ = ((lean_object*)(lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__7));
v___x_27_ = l_Lean_Expr_const___override(v___x_26_, v___x_25_);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules(lean_object* v_a_28_, lean_object* v_a_29_, lean_object* v_a_30_, lean_object* v_a_31_){
_start:
{
lean_object* v___x_33_; lean_object* v___x_34_; lean_object* v___x_35_; lean_object* v___x_36_; uint8_t v___x_37_; lean_object* v___x_38_; lean_object* v___x_39_; 
v___x_33_ = lean_unsigned_to_nat(0u);
v___x_34_ = lean_obj_once(&lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__4, &lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__4_once, _init_lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__4);
v___x_35_ = ((lean_object*)(lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__5));
v___x_36_ = lean_obj_once(&lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__8, &lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__8_once, _init_lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__8);
v___x_37_ = 0;
v___x_38_ = lean_box(0);
v___x_39_ = l_Lean_Meta_mkFreshExprMVarAt(v___x_34_, v___x_35_, v___x_36_, v___x_37_, v___x_38_, v___x_33_, v_a_28_, v_a_29_, v_a_30_, v_a_31_);
if (lean_obj_tag(v___x_39_) == 0)
{
lean_object* v_a_40_; lean_object* v___x_42_; uint8_t v_isShared_43_; uint8_t v_isSharedCheck_49_; 
v_a_40_ = lean_ctor_get(v___x_39_, 0);
v_isSharedCheck_49_ = !lean_is_exclusive(v___x_39_);
if (v_isSharedCheck_49_ == 0)
{
v___x_42_ = v___x_39_;
v_isShared_43_ = v_isSharedCheck_49_;
goto v_resetjp_41_;
}
else
{
lean_inc(v_a_40_);
lean_dec(v___x_39_);
v___x_42_ = lean_box(0);
v_isShared_43_ = v_isSharedCheck_49_;
goto v_resetjp_41_;
}
v_resetjp_41_:
{
lean_object* v___x_44_; lean_object* v___x_45_; lean_object* v___x_47_; 
v___x_44_ = l_Lean_Expr_mvarId_x21(v_a_40_);
lean_dec(v_a_40_);
v___x_45_ = lp_aesop_Aesop_ElabM_Context_forAdditionalRules(v___x_44_);
if (v_isShared_43_ == 0)
{
lean_ctor_set(v___x_42_, 0, v___x_45_);
v___x_47_ = v___x_42_;
goto v_reusejp_46_;
}
else
{
lean_object* v_reuseFailAlloc_48_; 
v_reuseFailAlloc_48_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_48_, 0, v___x_45_);
v___x_47_ = v_reuseFailAlloc_48_;
goto v_reusejp_46_;
}
v_reusejp_46_:
{
return v___x_47_;
}
}
}
else
{
lean_object* v_a_50_; lean_object* v___x_52_; uint8_t v_isShared_53_; uint8_t v_isSharedCheck_57_; 
v_a_50_ = lean_ctor_get(v___x_39_, 0);
v_isSharedCheck_57_ = !lean_is_exclusive(v___x_39_);
if (v_isSharedCheck_57_ == 0)
{
v___x_52_ = v___x_39_;
v_isShared_53_ = v_isSharedCheck_57_;
goto v_resetjp_51_;
}
else
{
lean_inc(v_a_50_);
lean_dec(v___x_39_);
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___boxed(lean_object* v_a_58_, lean_object* v_a_59_, lean_object* v_a_60_, lean_object* v_a_61_, lean_object* v_a_62_){
_start:
{
lean_object* v_res_63_; 
v_res_63_ = lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules(v_a_58_, v_a_59_, v_a_60_, v_a_61_);
lean_dec(v_a_61_);
lean_dec_ref(v_a_60_);
lean_dec(v_a_59_);
lean_dec_ref(v_a_58_);
return v_res_63_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabM_Context_forErasing(lean_object* v_goal_64_){
_start:
{
uint8_t v___x_65_; lean_object* v___x_66_; 
v___x_65_ = 0;
v___x_66_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_66_, 0, v_goal_64_);
lean_ctor_set_uint8(v___x_66_, sizeof(void*)*1, v___x_65_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabM_Context_forGlobalErasing(lean_object* v_a_67_, lean_object* v_a_68_, lean_object* v_a_69_, lean_object* v_a_70_){
_start:
{
lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; uint8_t v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; 
v___x_72_ = lean_unsigned_to_nat(32u);
v___x_73_ = lean_mk_empty_array_with_capacity(v___x_72_);
lean_dec_ref(v___x_73_);
v___x_74_ = lean_unsigned_to_nat(0u);
v___x_75_ = lean_obj_once(&lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__4, &lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__4_once, _init_lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__4);
v___x_76_ = ((lean_object*)(lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__5));
v___x_77_ = lean_obj_once(&lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__8, &lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__8_once, _init_lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules___closed__8);
v___x_78_ = 0;
v___x_79_ = lean_box(0);
v___x_80_ = l_Lean_Meta_mkFreshExprMVarAt(v___x_75_, v___x_76_, v___x_77_, v___x_78_, v___x_79_, v___x_74_, v_a_67_, v_a_68_, v_a_69_, v_a_70_);
if (lean_obj_tag(v___x_80_) == 0)
{
lean_object* v_a_81_; lean_object* v___x_83_; uint8_t v_isShared_84_; uint8_t v_isSharedCheck_90_; 
v_a_81_ = lean_ctor_get(v___x_80_, 0);
v_isSharedCheck_90_ = !lean_is_exclusive(v___x_80_);
if (v_isSharedCheck_90_ == 0)
{
v___x_83_ = v___x_80_;
v_isShared_84_ = v_isSharedCheck_90_;
goto v_resetjp_82_;
}
else
{
lean_inc(v_a_81_);
lean_dec(v___x_80_);
v___x_83_ = lean_box(0);
v_isShared_84_ = v_isSharedCheck_90_;
goto v_resetjp_82_;
}
v_resetjp_82_:
{
lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_88_; 
v___x_85_ = l_Lean_Expr_mvarId_x21(v_a_81_);
lean_dec(v_a_81_);
v___x_86_ = lp_aesop_Aesop_ElabM_Context_forErasing(v___x_85_);
if (v_isShared_84_ == 0)
{
lean_ctor_set(v___x_83_, 0, v___x_86_);
v___x_88_ = v___x_83_;
goto v_reusejp_87_;
}
else
{
lean_object* v_reuseFailAlloc_89_; 
v_reuseFailAlloc_89_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_89_, 0, v___x_86_);
v___x_88_ = v_reuseFailAlloc_89_;
goto v_reusejp_87_;
}
v_reusejp_87_:
{
return v___x_88_;
}
}
}
else
{
lean_object* v_a_91_; lean_object* v___x_93_; uint8_t v_isShared_94_; uint8_t v_isSharedCheck_98_; 
v_a_91_ = lean_ctor_get(v___x_80_, 0);
v_isSharedCheck_98_ = !lean_is_exclusive(v___x_80_);
if (v_isSharedCheck_98_ == 0)
{
v___x_93_ = v___x_80_;
v_isShared_94_ = v_isSharedCheck_98_;
goto v_resetjp_92_;
}
else
{
lean_inc(v_a_91_);
lean_dec(v___x_80_);
v___x_93_ = lean_box(0);
v_isShared_94_ = v_isSharedCheck_98_;
goto v_resetjp_92_;
}
v_resetjp_92_:
{
lean_object* v___x_96_; 
if (v_isShared_94_ == 0)
{
v___x_96_ = v___x_93_;
goto v_reusejp_95_;
}
else
{
lean_object* v_reuseFailAlloc_97_; 
v_reuseFailAlloc_97_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_97_, 0, v_a_91_);
v___x_96_ = v_reuseFailAlloc_97_;
goto v_reusejp_95_;
}
v_reusejp_95_:
{
return v___x_96_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabM_Context_forGlobalErasing___boxed(lean_object* v_a_99_, lean_object* v_a_100_, lean_object* v_a_101_, lean_object* v_a_102_, lean_object* v_a_103_){
_start:
{
lean_object* v_res_104_; 
v_res_104_ = lp_aesop_Aesop_ElabM_Context_forGlobalErasing(v_a_99_, v_a_100_, v_a_101_, v_a_102_);
lean_dec(v_a_102_);
lean_dec_ref(v_a_101_);
lean_dec(v_a_100_);
lean_dec_ref(v_a_99_);
return v_res_104_;
}
}
static lean_object* _init_lp_aesop_Aesop_instMonadElabM___closed__0(void){
_start:
{
lean_object* v___x_105_; 
v___x_105_ = l_instMonadEIO(lean_box(0));
return v___x_105_;
}
}
static lean_object* _init_lp_aesop_Aesop_instMonadElabM___closed__1(void){
_start:
{
lean_object* v___x_106_; lean_object* v___x_107_; 
v___x_106_ = lean_obj_once(&lp_aesop_Aesop_instMonadElabM___closed__0, &lp_aesop_Aesop_instMonadElabM___closed__0_once, _init_lp_aesop_Aesop_instMonadElabM___closed__0);
v___x_107_ = l_StateRefT_x27_instMonad___redArg(v___x_106_);
return v___x_107_;
}
}
static lean_object* _init_lp_aesop_Aesop_instMonadElabM(void){
_start:
{
lean_object* v___x_114_; lean_object* v_toApplicative_115_; lean_object* v_toFunctor_116_; lean_object* v_toSeq_117_; lean_object* v_toSeqLeft_118_; lean_object* v_toSeqRight_119_; lean_object* v___f_120_; lean_object* v___f_121_; lean_object* v___f_122_; lean_object* v___f_123_; lean_object* v___x_124_; lean_object* v___f_125_; lean_object* v___f_126_; lean_object* v___f_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v_toApplicative_131_; lean_object* v___x_133_; uint8_t v_isShared_134_; uint8_t v_isSharedCheck_189_; 
v___x_114_ = lean_obj_once(&lp_aesop_Aesop_instMonadElabM___closed__1, &lp_aesop_Aesop_instMonadElabM___closed__1_once, _init_lp_aesop_Aesop_instMonadElabM___closed__1);
v_toApplicative_115_ = lean_ctor_get(v___x_114_, 0);
v_toFunctor_116_ = lean_ctor_get(v_toApplicative_115_, 0);
v_toSeq_117_ = lean_ctor_get(v_toApplicative_115_, 2);
v_toSeqLeft_118_ = lean_ctor_get(v_toApplicative_115_, 3);
v_toSeqRight_119_ = lean_ctor_get(v_toApplicative_115_, 4);
v___f_120_ = ((lean_object*)(lp_aesop_Aesop_instMonadElabM___closed__2));
v___f_121_ = ((lean_object*)(lp_aesop_Aesop_instMonadElabM___closed__3));
lean_inc_ref_n(v_toFunctor_116_, 2);
v___f_122_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_122_, 0, v_toFunctor_116_);
v___f_123_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_123_, 0, v_toFunctor_116_);
v___x_124_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_124_, 0, v___f_122_);
lean_ctor_set(v___x_124_, 1, v___f_123_);
lean_inc(v_toSeqRight_119_);
v___f_125_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_125_, 0, v_toSeqRight_119_);
lean_inc(v_toSeqLeft_118_);
v___f_126_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_126_, 0, v_toSeqLeft_118_);
lean_inc(v_toSeq_117_);
v___f_127_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_127_, 0, v_toSeq_117_);
v___x_128_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_128_, 0, v___x_124_);
lean_ctor_set(v___x_128_, 1, v___f_120_);
lean_ctor_set(v___x_128_, 2, v___f_127_);
lean_ctor_set(v___x_128_, 3, v___f_126_);
lean_ctor_set(v___x_128_, 4, v___f_125_);
v___x_129_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_129_, 0, v___x_128_);
lean_ctor_set(v___x_129_, 1, v___f_121_);
v___x_130_ = l_StateRefT_x27_instMonad___redArg(v___x_129_);
v_toApplicative_131_ = lean_ctor_get(v___x_130_, 0);
v_isSharedCheck_189_ = !lean_is_exclusive(v___x_130_);
if (v_isSharedCheck_189_ == 0)
{
lean_object* v_unused_190_; 
v_unused_190_ = lean_ctor_get(v___x_130_, 1);
lean_dec(v_unused_190_);
v___x_133_ = v___x_130_;
v_isShared_134_ = v_isSharedCheck_189_;
goto v_resetjp_132_;
}
else
{
lean_inc(v_toApplicative_131_);
lean_dec(v___x_130_);
v___x_133_ = lean_box(0);
v_isShared_134_ = v_isSharedCheck_189_;
goto v_resetjp_132_;
}
v_resetjp_132_:
{
lean_object* v_toFunctor_135_; lean_object* v_toSeq_136_; lean_object* v_toSeqLeft_137_; lean_object* v_toSeqRight_138_; lean_object* v___x_140_; uint8_t v_isShared_141_; uint8_t v_isSharedCheck_187_; 
v_toFunctor_135_ = lean_ctor_get(v_toApplicative_131_, 0);
v_toSeq_136_ = lean_ctor_get(v_toApplicative_131_, 2);
v_toSeqLeft_137_ = lean_ctor_get(v_toApplicative_131_, 3);
v_toSeqRight_138_ = lean_ctor_get(v_toApplicative_131_, 4);
v_isSharedCheck_187_ = !lean_is_exclusive(v_toApplicative_131_);
if (v_isSharedCheck_187_ == 0)
{
lean_object* v_unused_188_; 
v_unused_188_ = lean_ctor_get(v_toApplicative_131_, 1);
lean_dec(v_unused_188_);
v___x_140_ = v_toApplicative_131_;
v_isShared_141_ = v_isSharedCheck_187_;
goto v_resetjp_139_;
}
else
{
lean_inc(v_toSeqRight_138_);
lean_inc(v_toSeqLeft_137_);
lean_inc(v_toSeq_136_);
lean_inc(v_toFunctor_135_);
lean_dec(v_toApplicative_131_);
v___x_140_ = lean_box(0);
v_isShared_141_ = v_isSharedCheck_187_;
goto v_resetjp_139_;
}
v_resetjp_139_:
{
lean_object* v___f_142_; lean_object* v___f_143_; lean_object* v___f_144_; lean_object* v___f_145_; lean_object* v___x_146_; lean_object* v___f_147_; lean_object* v___f_148_; lean_object* v___f_149_; lean_object* v___x_151_; 
v___f_142_ = ((lean_object*)(lp_aesop_Aesop_instMonadElabM___closed__4));
v___f_143_ = ((lean_object*)(lp_aesop_Aesop_instMonadElabM___closed__5));
lean_inc_ref(v_toFunctor_135_);
v___f_144_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_144_, 0, v_toFunctor_135_);
v___f_145_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_145_, 0, v_toFunctor_135_);
v___x_146_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_146_, 0, v___f_144_);
lean_ctor_set(v___x_146_, 1, v___f_145_);
v___f_147_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_147_, 0, v_toSeqRight_138_);
v___f_148_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_148_, 0, v_toSeqLeft_137_);
v___f_149_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_149_, 0, v_toSeq_136_);
if (v_isShared_141_ == 0)
{
lean_ctor_set(v___x_140_, 4, v___f_147_);
lean_ctor_set(v___x_140_, 3, v___f_148_);
lean_ctor_set(v___x_140_, 2, v___f_149_);
lean_ctor_set(v___x_140_, 1, v___f_142_);
lean_ctor_set(v___x_140_, 0, v___x_146_);
v___x_151_ = v___x_140_;
goto v_reusejp_150_;
}
else
{
lean_object* v_reuseFailAlloc_186_; 
v_reuseFailAlloc_186_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_186_, 0, v___x_146_);
lean_ctor_set(v_reuseFailAlloc_186_, 1, v___f_142_);
lean_ctor_set(v_reuseFailAlloc_186_, 2, v___f_149_);
lean_ctor_set(v_reuseFailAlloc_186_, 3, v___f_148_);
lean_ctor_set(v_reuseFailAlloc_186_, 4, v___f_147_);
v___x_151_ = v_reuseFailAlloc_186_;
goto v_reusejp_150_;
}
v_reusejp_150_:
{
lean_object* v___x_153_; 
if (v_isShared_134_ == 0)
{
lean_ctor_set(v___x_133_, 1, v___f_143_);
lean_ctor_set(v___x_133_, 0, v___x_151_);
v___x_153_ = v___x_133_;
goto v_reusejp_152_;
}
else
{
lean_object* v_reuseFailAlloc_185_; 
v_reuseFailAlloc_185_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_185_, 0, v___x_151_);
lean_ctor_set(v_reuseFailAlloc_185_, 1, v___f_143_);
v___x_153_ = v_reuseFailAlloc_185_;
goto v_reusejp_152_;
}
v_reusejp_152_:
{
lean_object* v___x_154_; lean_object* v_toApplicative_155_; lean_object* v___x_157_; uint8_t v_isShared_158_; uint8_t v_isSharedCheck_183_; 
v___x_154_ = l_StateRefT_x27_instMonad___redArg(v___x_153_);
v_toApplicative_155_ = lean_ctor_get(v___x_154_, 0);
v_isSharedCheck_183_ = !lean_is_exclusive(v___x_154_);
if (v_isSharedCheck_183_ == 0)
{
lean_object* v_unused_184_; 
v_unused_184_ = lean_ctor_get(v___x_154_, 1);
lean_dec(v_unused_184_);
v___x_157_ = v___x_154_;
v_isShared_158_ = v_isSharedCheck_183_;
goto v_resetjp_156_;
}
else
{
lean_inc(v_toApplicative_155_);
lean_dec(v___x_154_);
v___x_157_ = lean_box(0);
v_isShared_158_ = v_isSharedCheck_183_;
goto v_resetjp_156_;
}
v_resetjp_156_:
{
lean_object* v_toFunctor_159_; lean_object* v_toSeq_160_; lean_object* v_toSeqLeft_161_; lean_object* v_toSeqRight_162_; lean_object* v___x_164_; uint8_t v_isShared_165_; uint8_t v_isSharedCheck_181_; 
v_toFunctor_159_ = lean_ctor_get(v_toApplicative_155_, 0);
v_toSeq_160_ = lean_ctor_get(v_toApplicative_155_, 2);
v_toSeqLeft_161_ = lean_ctor_get(v_toApplicative_155_, 3);
v_toSeqRight_162_ = lean_ctor_get(v_toApplicative_155_, 4);
v_isSharedCheck_181_ = !lean_is_exclusive(v_toApplicative_155_);
if (v_isSharedCheck_181_ == 0)
{
lean_object* v_unused_182_; 
v_unused_182_ = lean_ctor_get(v_toApplicative_155_, 1);
lean_dec(v_unused_182_);
v___x_164_ = v_toApplicative_155_;
v_isShared_165_ = v_isSharedCheck_181_;
goto v_resetjp_163_;
}
else
{
lean_inc(v_toSeqRight_162_);
lean_inc(v_toSeqLeft_161_);
lean_inc(v_toSeq_160_);
lean_inc(v_toFunctor_159_);
lean_dec(v_toApplicative_155_);
v___x_164_ = lean_box(0);
v_isShared_165_ = v_isSharedCheck_181_;
goto v_resetjp_163_;
}
v_resetjp_163_:
{
lean_object* v___f_166_; lean_object* v___f_167_; lean_object* v___f_168_; lean_object* v___f_169_; lean_object* v___x_170_; lean_object* v___f_171_; lean_object* v___f_172_; lean_object* v___f_173_; lean_object* v___x_175_; 
v___f_166_ = ((lean_object*)(lp_aesop_Aesop_instMonadElabM___closed__6));
v___f_167_ = ((lean_object*)(lp_aesop_Aesop_instMonadElabM___closed__7));
lean_inc_ref(v_toFunctor_159_);
v___f_168_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_168_, 0, v_toFunctor_159_);
v___f_169_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_169_, 0, v_toFunctor_159_);
v___x_170_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_170_, 0, v___f_168_);
lean_ctor_set(v___x_170_, 1, v___f_169_);
v___f_171_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_171_, 0, v_toSeqRight_162_);
v___f_172_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_172_, 0, v_toSeqLeft_161_);
v___f_173_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_173_, 0, v_toSeq_160_);
if (v_isShared_165_ == 0)
{
lean_ctor_set(v___x_164_, 4, v___f_171_);
lean_ctor_set(v___x_164_, 3, v___f_172_);
lean_ctor_set(v___x_164_, 2, v___f_173_);
lean_ctor_set(v___x_164_, 1, v___f_166_);
lean_ctor_set(v___x_164_, 0, v___x_170_);
v___x_175_ = v___x_164_;
goto v_reusejp_174_;
}
else
{
lean_object* v_reuseFailAlloc_180_; 
v_reuseFailAlloc_180_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_180_, 0, v___x_170_);
lean_ctor_set(v_reuseFailAlloc_180_, 1, v___f_166_);
lean_ctor_set(v_reuseFailAlloc_180_, 2, v___f_173_);
lean_ctor_set(v_reuseFailAlloc_180_, 3, v___f_172_);
lean_ctor_set(v_reuseFailAlloc_180_, 4, v___f_171_);
v___x_175_ = v_reuseFailAlloc_180_;
goto v_reusejp_174_;
}
v_reusejp_174_:
{
lean_object* v___x_177_; 
if (v_isShared_158_ == 0)
{
lean_ctor_set(v___x_157_, 1, v___f_167_);
lean_ctor_set(v___x_157_, 0, v___x_175_);
v___x_177_ = v___x_157_;
goto v_reusejp_176_;
}
else
{
lean_object* v_reuseFailAlloc_179_; 
v_reuseFailAlloc_179_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_179_, 0, v___x_175_);
lean_ctor_set(v_reuseFailAlloc_179_, 1, v___f_167_);
v___x_177_ = v_reuseFailAlloc_179_;
goto v_reusejp_176_;
}
v_reusejp_176_:
{
lean_object* v___x_178_; 
v___x_178_ = l_ReaderT_instMonad___redArg(v___x_177_);
return v___x_178_;
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabM_run___redArg(lean_object* v_ctx_191_, lean_object* v_x_192_, lean_object* v_a_193_, lean_object* v_a_194_, lean_object* v_a_195_, lean_object* v_a_196_, lean_object* v_a_197_, lean_object* v_a_198_){
_start:
{
lean_object* v___x_200_; 
lean_inc(v_a_198_);
lean_inc_ref(v_a_197_);
lean_inc(v_a_196_);
lean_inc_ref(v_a_195_);
lean_inc(v_a_194_);
lean_inc_ref(v_a_193_);
v___x_200_ = lean_apply_8(v_x_192_, v_ctx_191_, v_a_193_, v_a_194_, v_a_195_, v_a_196_, v_a_197_, v_a_198_, lean_box(0));
return v___x_200_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabM_run___redArg___boxed(lean_object* v_ctx_201_, lean_object* v_x_202_, lean_object* v_a_203_, lean_object* v_a_204_, lean_object* v_a_205_, lean_object* v_a_206_, lean_object* v_a_207_, lean_object* v_a_208_, lean_object* v_a_209_){
_start:
{
lean_object* v_res_210_; 
v_res_210_ = lp_aesop_Aesop_ElabM_run___redArg(v_ctx_201_, v_x_202_, v_a_203_, v_a_204_, v_a_205_, v_a_206_, v_a_207_, v_a_208_);
lean_dec(v_a_208_);
lean_dec_ref(v_a_207_);
lean_dec(v_a_206_);
lean_dec_ref(v_a_205_);
lean_dec(v_a_204_);
lean_dec_ref(v_a_203_);
return v_res_210_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabM_run(lean_object* v_00_u03b1_211_, lean_object* v_ctx_212_, lean_object* v_x_213_, lean_object* v_a_214_, lean_object* v_a_215_, lean_object* v_a_216_, lean_object* v_a_217_, lean_object* v_a_218_, lean_object* v_a_219_){
_start:
{
lean_object* v___x_221_; 
lean_inc(v_a_219_);
lean_inc_ref(v_a_218_);
lean_inc(v_a_217_);
lean_inc_ref(v_a_216_);
lean_inc(v_a_215_);
lean_inc_ref(v_a_214_);
v___x_221_ = lean_apply_8(v_x_213_, v_ctx_212_, v_a_214_, v_a_215_, v_a_216_, v_a_217_, v_a_218_, v_a_219_, lean_box(0));
return v___x_221_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabM_run___boxed(lean_object* v_00_u03b1_222_, lean_object* v_ctx_223_, lean_object* v_x_224_, lean_object* v_a_225_, lean_object* v_a_226_, lean_object* v_a_227_, lean_object* v_a_228_, lean_object* v_a_229_, lean_object* v_a_230_, lean_object* v_a_231_){
_start:
{
lean_object* v_res_232_; 
v_res_232_ = lp_aesop_Aesop_ElabM_run(v_00_u03b1_222_, v_ctx_223_, v_x_224_, v_a_225_, v_a_226_, v_a_227_, v_a_228_, v_a_229_, v_a_230_);
lean_dec(v_a_230_);
lean_dec_ref(v_a_229_);
lean_dec(v_a_228_);
lean_dec_ref(v_a_227_);
lean_dec(v_a_226_);
lean_dec_ref(v_a_225_);
return v_res_232_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_shouldParsePriorities___redArg(lean_object* v_a_233_){
_start:
{
uint8_t v_parsePriorities_235_; lean_object* v___x_236_; lean_object* v___x_237_; 
v_parsePriorities_235_ = lean_ctor_get_uint8(v_a_233_, sizeof(void*)*1);
v___x_236_ = lean_box(v_parsePriorities_235_);
v___x_237_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_237_, 0, v___x_236_);
return v___x_237_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_shouldParsePriorities___redArg___boxed(lean_object* v_a_238_, lean_object* v_a_239_){
_start:
{
lean_object* v_res_240_; 
v_res_240_ = lp_aesop_Aesop_shouldParsePriorities___redArg(v_a_238_);
lean_dec_ref(v_a_238_);
return v_res_240_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_shouldParsePriorities(lean_object* v_a_241_, lean_object* v_a_242_, lean_object* v_a_243_, lean_object* v_a_244_, lean_object* v_a_245_, lean_object* v_a_246_, lean_object* v_a_247_){
_start:
{
lean_object* v___x_249_; 
v___x_249_ = lp_aesop_Aesop_shouldParsePriorities___redArg(v_a_241_);
return v___x_249_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_shouldParsePriorities___boxed(lean_object* v_a_250_, lean_object* v_a_251_, lean_object* v_a_252_, lean_object* v_a_253_, lean_object* v_a_254_, lean_object* v_a_255_, lean_object* v_a_256_, lean_object* v_a_257_){
_start:
{
lean_object* v_res_258_; 
v_res_258_ = lp_aesop_Aesop_shouldParsePriorities(v_a_250_, v_a_251_, v_a_252_, v_a_253_, v_a_254_, v_a_255_, v_a_256_);
lean_dec(v_a_256_);
lean_dec_ref(v_a_255_);
lean_dec(v_a_254_);
lean_dec_ref(v_a_253_);
lean_dec(v_a_252_);
lean_dec_ref(v_a_251_);
lean_dec_ref(v_a_250_);
return v_res_258_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getGoal___redArg(lean_object* v_a_259_){
_start:
{
lean_object* v_goal_261_; lean_object* v___x_262_; 
v_goal_261_ = lean_ctor_get(v_a_259_, 0);
lean_inc(v_goal_261_);
v___x_262_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_262_, 0, v_goal_261_);
return v___x_262_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getGoal___redArg___boxed(lean_object* v_a_263_, lean_object* v_a_264_){
_start:
{
lean_object* v_res_265_; 
v_res_265_ = lp_aesop_Aesop_getGoal___redArg(v_a_263_);
lean_dec_ref(v_a_263_);
return v_res_265_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getGoal(lean_object* v_a_266_, lean_object* v_a_267_, lean_object* v_a_268_, lean_object* v_a_269_, lean_object* v_a_270_, lean_object* v_a_271_, lean_object* v_a_272_){
_start:
{
lean_object* v___x_274_; 
v___x_274_ = lp_aesop_Aesop_getGoal___redArg(v_a_266_);
return v___x_274_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getGoal___boxed(lean_object* v_a_275_, lean_object* v_a_276_, lean_object* v_a_277_, lean_object* v_a_278_, lean_object* v_a_279_, lean_object* v_a_280_, lean_object* v_a_281_, lean_object* v_a_282_){
_start:
{
lean_object* v_res_283_; 
v_res_283_ = lp_aesop_Aesop_getGoal(v_a_275_, v_a_276_, v_a_277_, v_a_278_, v_a_279_, v_a_280_, v_a_281_);
lean_dec(v_a_281_);
lean_dec_ref(v_a_280_);
lean_dec(v_a_279_);
lean_dec_ref(v_a_278_);
lean_dec(v_a_277_);
lean_dec_ref(v_a_276_);
lean_dec_ref(v_a_275_);
return v_res_283_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Term_TermElabM(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_ElabM(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Term_TermElabM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_instMonadElabM = _init_lp_aesop_Aesop_instMonadElabM();
lean_mark_persistent(lp_aesop_Aesop_instMonadElabM);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_ElabM(uint8_t builtin) {
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
lean_object* initialize_Lean_Elab_Term_TermElabM(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_ElabM(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Term_TermElabM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_ElabM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_ElabM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_ElabM(builtin);
}
#ifdef __cplusplus
}
#endif
