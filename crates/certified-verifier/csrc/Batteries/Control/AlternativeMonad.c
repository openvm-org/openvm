// Lean compiler output
// Module: Batteries.Control.AlternativeMonad
// Imports: public import Init public meta import Init public import Batteries.Control.Lemmas public import Batteries.Control.OptionT import all Init.Control.Option import all Init.Control.State import all Init.Control.Reader import all Init.Control.StateRef public import Lean.Meta.Basic
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
lean_object* l_Lean_Meta_instMonadMetaM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadOption___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instAlternativeOfMonad___redArg(lean_object*, lean_object*);
lean_object* l_ReaderT_instMonad___redArg(lean_object*);
lean_object* l_instAlternativeOption___lam__1___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_StateRefT_x27_instAlternativeOfMonad___redArg(lean_object*, lean_object*);
lean_object* l_ReaderT_bind___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadOption___lam__0(lean_object*, lean_object*);
lean_object* l_Option_bind(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instAlternativeOption___lam__0(lean_object*);
lean_object* l_instMonadOption___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadOption___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instFunctorOption___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Option_map(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_OptionT_instAlternative___redArg(lean_object*);
lean_object* l_OptionT_bind(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_instAlternative___redArg(lean_object*, lean_object*);
lean_object* l_StateT_bind(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_instAlternativeMetaM;
LEAN_EXPORT lean_object* lp_batteries_AlternativeMonad_toMonad___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_AlternativeMonad_toMonad(lean_object*, lean_object*);
static const lean_closure_object lp_batteries_Option_instAlternativeMonad___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instAlternativeOption___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Option_instAlternativeMonad___closed__0 = (const lean_object*)&lp_batteries_Option_instAlternativeMonad___closed__0_value;
static const lean_closure_object lp_batteries_Option_instAlternativeMonad___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instAlternativeOption___lam__1___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Option_instAlternativeMonad___closed__1 = (const lean_object*)&lp_batteries_Option_instAlternativeMonad___closed__1_value;
static const lean_closure_object lp_batteries_Option_instAlternativeMonad___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadOption___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Option_instAlternativeMonad___closed__2 = (const lean_object*)&lp_batteries_Option_instAlternativeMonad___closed__2_value;
static const lean_closure_object lp_batteries_Option_instAlternativeMonad___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadOption___lam__1, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Option_instAlternativeMonad___closed__3 = (const lean_object*)&lp_batteries_Option_instAlternativeMonad___closed__3_value;
static const lean_closure_object lp_batteries_Option_instAlternativeMonad___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadOption___lam__2___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Option_instAlternativeMonad___closed__4 = (const lean_object*)&lp_batteries_Option_instAlternativeMonad___closed__4_value;
static const lean_closure_object lp_batteries_Option_instAlternativeMonad___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadOption___lam__3___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Option_instAlternativeMonad___closed__5 = (const lean_object*)&lp_batteries_Option_instAlternativeMonad___closed__5_value;
static const lean_closure_object lp_batteries_Option_instAlternativeMonad___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instFunctorOption___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Option_instAlternativeMonad___closed__6 = (const lean_object*)&lp_batteries_Option_instAlternativeMonad___closed__6_value;
static const lean_closure_object lp_batteries_Option_instAlternativeMonad___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Option_map, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Option_instAlternativeMonad___closed__7 = (const lean_object*)&lp_batteries_Option_instAlternativeMonad___closed__7_value;
static const lean_ctor_object lp_batteries_Option_instAlternativeMonad___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Option_instAlternativeMonad___closed__7_value),((lean_object*)&lp_batteries_Option_instAlternativeMonad___closed__6_value)}};
static const lean_object* lp_batteries_Option_instAlternativeMonad___closed__8 = (const lean_object*)&lp_batteries_Option_instAlternativeMonad___closed__8_value;
static const lean_ctor_object lp_batteries_Option_instAlternativeMonad___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Option_instAlternativeMonad___closed__8_value),((lean_object*)&lp_batteries_Option_instAlternativeMonad___closed__2_value),((lean_object*)&lp_batteries_Option_instAlternativeMonad___closed__3_value),((lean_object*)&lp_batteries_Option_instAlternativeMonad___closed__4_value),((lean_object*)&lp_batteries_Option_instAlternativeMonad___closed__5_value)}};
static const lean_object* lp_batteries_Option_instAlternativeMonad___closed__9 = (const lean_object*)&lp_batteries_Option_instAlternativeMonad___closed__9_value;
static const lean_ctor_object lp_batteries_Option_instAlternativeMonad___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Option_instAlternativeMonad___closed__9_value),((lean_object*)&lp_batteries_Option_instAlternativeMonad___closed__0_value),((lean_object*)&lp_batteries_Option_instAlternativeMonad___closed__1_value)}};
static const lean_object* lp_batteries_Option_instAlternativeMonad___closed__10 = (const lean_object*)&lp_batteries_Option_instAlternativeMonad___closed__10_value;
static const lean_closure_object lp_batteries_Option_instAlternativeMonad___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Option_bind, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Option_instAlternativeMonad___closed__11 = (const lean_object*)&lp_batteries_Option_instAlternativeMonad___closed__11_value;
static const lean_ctor_object lp_batteries_Option_instAlternativeMonad___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Option_instAlternativeMonad___closed__10_value),((lean_object*)&lp_batteries_Option_instAlternativeMonad___closed__11_value)}};
static const lean_object* lp_batteries_Option_instAlternativeMonad___closed__12 = (const lean_object*)&lp_batteries_Option_instAlternativeMonad___closed__12_value;
LEAN_EXPORT const lean_object* lp_batteries_Option_instAlternativeMonad = (const lean_object*)&lp_batteries_Option_instAlternativeMonad___closed__12_value;
LEAN_EXPORT lean_object* lp_batteries_OptionT_instAlternativeMonadOfMonad___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_OptionT_instAlternativeMonadOfMonad(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_StateT_instAlternativeMonad___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_StateT_instAlternativeMonad(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_ReaderT_instAlternativeMonad___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_ReaderT_instAlternativeMonad(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_StateRefT_x27_instAlternativeMonad___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_StateRefT_x27_instAlternativeMonad(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries_instAlternativeMonadMetaM___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_instAlternativeMonadMetaM___closed__0 = (const lean_object*)&lp_batteries_instAlternativeMonadMetaM___closed__0_value;
static lean_once_cell_t lp_batteries_instAlternativeMonadMetaM___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_instAlternativeMonadMetaM___closed__1;
LEAN_EXPORT lean_object* lp_batteries_instAlternativeMonadMetaM;
LEAN_EXPORT lean_object* lp_batteries_AlternativeMonad_toMonad___redArg(lean_object* v_self_1_){
_start:
{
lean_object* v_toAlternative_2_; lean_object* v_toBind_3_; lean_object* v___x_5_; uint8_t v_isShared_6_; uint8_t v_isSharedCheck_11_; 
v_toAlternative_2_ = lean_ctor_get(v_self_1_, 0);
v_toBind_3_ = lean_ctor_get(v_self_1_, 1);
v_isSharedCheck_11_ = !lean_is_exclusive(v_self_1_);
if (v_isSharedCheck_11_ == 0)
{
v___x_5_ = v_self_1_;
v_isShared_6_ = v_isSharedCheck_11_;
goto v_resetjp_4_;
}
else
{
lean_inc(v_toBind_3_);
lean_inc(v_toAlternative_2_);
lean_dec(v_self_1_);
v___x_5_ = lean_box(0);
v_isShared_6_ = v_isSharedCheck_11_;
goto v_resetjp_4_;
}
v_resetjp_4_:
{
lean_object* v_toApplicative_7_; lean_object* v___x_9_; 
v_toApplicative_7_ = lean_ctor_get(v_toAlternative_2_, 0);
lean_inc_ref(v_toApplicative_7_);
lean_dec_ref(v_toAlternative_2_);
if (v_isShared_6_ == 0)
{
lean_ctor_set(v___x_5_, 0, v_toApplicative_7_);
v___x_9_ = v___x_5_;
goto v_reusejp_8_;
}
else
{
lean_object* v_reuseFailAlloc_10_; 
v_reuseFailAlloc_10_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_10_, 0, v_toApplicative_7_);
lean_ctor_set(v_reuseFailAlloc_10_, 1, v_toBind_3_);
v___x_9_ = v_reuseFailAlloc_10_;
goto v_reusejp_8_;
}
v_reusejp_8_:
{
return v___x_9_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_AlternativeMonad_toMonad(lean_object* v_m_12_, lean_object* v_self_13_){
_start:
{
lean_object* v___x_14_; 
v___x_14_ = lp_batteries_AlternativeMonad_toMonad___redArg(v_self_13_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_batteries_OptionT_instAlternativeMonadOfMonad___redArg(lean_object* v_inst_41_){
_start:
{
lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___x_44_; 
lean_inc_ref(v_inst_41_);
v___x_42_ = l_OptionT_instAlternative___redArg(v_inst_41_);
v___x_43_ = lean_alloc_closure((void*)(l_OptionT_bind), 6, 2);
lean_closure_set(v___x_43_, 0, lean_box(0));
lean_closure_set(v___x_43_, 1, v_inst_41_);
v___x_44_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_44_, 0, v___x_42_);
lean_ctor_set(v___x_44_, 1, v___x_43_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_batteries_OptionT_instAlternativeMonadOfMonad(lean_object* v_m_45_, lean_object* v_inst_46_){
_start:
{
lean_object* v___x_47_; 
v___x_47_ = lp_batteries_OptionT_instAlternativeMonadOfMonad___redArg(v_inst_46_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_batteries_StateT_instAlternativeMonad___redArg(lean_object* v_inst_48_){
_start:
{
lean_object* v___x_49_; lean_object* v_toAlternative_50_; lean_object* v___x_52_; uint8_t v_isShared_53_; uint8_t v_isSharedCheck_59_; 
lean_inc_ref(v_inst_48_);
v___x_49_ = lp_batteries_AlternativeMonad_toMonad___redArg(v_inst_48_);
v_toAlternative_50_ = lean_ctor_get(v_inst_48_, 0);
v_isSharedCheck_59_ = !lean_is_exclusive(v_inst_48_);
if (v_isSharedCheck_59_ == 0)
{
lean_object* v_unused_60_; 
v_unused_60_ = lean_ctor_get(v_inst_48_, 1);
lean_dec(v_unused_60_);
v___x_52_ = v_inst_48_;
v_isShared_53_ = v_isSharedCheck_59_;
goto v_resetjp_51_;
}
else
{
lean_inc(v_toAlternative_50_);
lean_dec(v_inst_48_);
v___x_52_ = lean_box(0);
v_isShared_53_ = v_isSharedCheck_59_;
goto v_resetjp_51_;
}
v_resetjp_51_:
{
lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_57_; 
lean_inc_ref(v___x_49_);
v___x_54_ = l_StateT_instAlternative___redArg(v___x_49_, v_toAlternative_50_);
v___x_55_ = lean_alloc_closure((void*)(l_StateT_bind), 8, 3);
lean_closure_set(v___x_55_, 0, lean_box(0));
lean_closure_set(v___x_55_, 1, lean_box(0));
lean_closure_set(v___x_55_, 2, v___x_49_);
if (v_isShared_53_ == 0)
{
lean_ctor_set(v___x_52_, 1, v___x_55_);
lean_ctor_set(v___x_52_, 0, v___x_54_);
v___x_57_ = v___x_52_;
goto v_reusejp_56_;
}
else
{
lean_object* v_reuseFailAlloc_58_; 
v_reuseFailAlloc_58_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_58_, 0, v___x_54_);
lean_ctor_set(v_reuseFailAlloc_58_, 1, v___x_55_);
v___x_57_ = v_reuseFailAlloc_58_;
goto v_reusejp_56_;
}
v_reusejp_56_:
{
return v___x_57_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_StateT_instAlternativeMonad(lean_object* v_00_u03c3_61_, lean_object* v_m_62_, lean_object* v_inst_63_){
_start:
{
lean_object* v___x_64_; 
v___x_64_ = lp_batteries_StateT_instAlternativeMonad___redArg(v_inst_63_);
return v___x_64_;
}
}
LEAN_EXPORT lean_object* lp_batteries_ReaderT_instAlternativeMonad___redArg(lean_object* v_inst_65_){
_start:
{
lean_object* v_toAlternative_66_; lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v_toBind_70_; lean_object* v___x_72_; uint8_t v_isShared_73_; uint8_t v_isSharedCheck_77_; 
v_toAlternative_66_ = lean_ctor_get(v_inst_65_, 0);
lean_inc_ref(v_toAlternative_66_);
v___x_67_ = lp_batteries_AlternativeMonad_toMonad___redArg(v_inst_65_);
lean_inc_ref(v___x_67_);
v___x_68_ = l_ReaderT_instAlternativeOfMonad___redArg(v_toAlternative_66_, v___x_67_);
v___x_69_ = l_ReaderT_instMonad___redArg(v___x_67_);
v_toBind_70_ = lean_ctor_get(v___x_69_, 1);
v_isSharedCheck_77_ = !lean_is_exclusive(v___x_69_);
if (v_isSharedCheck_77_ == 0)
{
lean_object* v_unused_78_; 
v_unused_78_ = lean_ctor_get(v___x_69_, 0);
lean_dec(v_unused_78_);
v___x_72_ = v___x_69_;
v_isShared_73_ = v_isSharedCheck_77_;
goto v_resetjp_71_;
}
else
{
lean_inc(v_toBind_70_);
lean_dec(v___x_69_);
v___x_72_ = lean_box(0);
v_isShared_73_ = v_isSharedCheck_77_;
goto v_resetjp_71_;
}
v_resetjp_71_:
{
lean_object* v___x_75_; 
if (v_isShared_73_ == 0)
{
lean_ctor_set(v___x_72_, 0, v___x_68_);
v___x_75_ = v___x_72_;
goto v_reusejp_74_;
}
else
{
lean_object* v_reuseFailAlloc_76_; 
v_reuseFailAlloc_76_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_76_, 0, v___x_68_);
lean_ctor_set(v_reuseFailAlloc_76_, 1, v_toBind_70_);
v___x_75_ = v_reuseFailAlloc_76_;
goto v_reusejp_74_;
}
v_reusejp_74_:
{
return v___x_75_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_ReaderT_instAlternativeMonad(lean_object* v_m_79_, lean_object* v_00_u03c1_80_, lean_object* v_inst_81_){
_start:
{
lean_object* v___x_82_; 
v___x_82_ = lp_batteries_ReaderT_instAlternativeMonad___redArg(v_inst_81_);
return v___x_82_;
}
}
LEAN_EXPORT lean_object* lp_batteries_StateRefT_x27_instAlternativeMonad___redArg(lean_object* v_inst_83_){
_start:
{
lean_object* v_toAlternative_84_; lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; 
v_toAlternative_84_ = lean_ctor_get(v_inst_83_, 0);
lean_inc_ref(v_toAlternative_84_);
v___x_85_ = lp_batteries_AlternativeMonad_toMonad___redArg(v_inst_83_);
lean_inc_ref(v___x_85_);
v___x_86_ = l_StateRefT_x27_instAlternativeOfMonad___redArg(v_toAlternative_84_, v___x_85_);
v___x_87_ = lean_alloc_closure((void*)(l_ReaderT_bind___boxed), 8, 3);
lean_closure_set(v___x_87_, 0, lean_box(0));
lean_closure_set(v___x_87_, 1, lean_box(0));
lean_closure_set(v___x_87_, 2, v___x_85_);
v___x_88_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_88_, 0, v___x_86_);
lean_ctor_set(v___x_88_, 1, v___x_87_);
return v___x_88_;
}
}
LEAN_EXPORT lean_object* lp_batteries_StateRefT_x27_instAlternativeMonad(lean_object* v_m_89_, lean_object* v_00_u03c9_90_, lean_object* v_00_u03c3_91_, lean_object* v_inst_92_){
_start:
{
lean_object* v___x_93_; 
v___x_93_ = lp_batteries_StateRefT_x27_instAlternativeMonad___redArg(v_inst_92_);
return v___x_93_;
}
}
static lean_object* _init_lp_batteries_instAlternativeMonadMetaM___closed__1(void){
_start:
{
lean_object* v___f_95_; lean_object* v___x_96_; lean_object* v___x_97_; 
v___f_95_ = ((lean_object*)(lp_batteries_instAlternativeMonadMetaM___closed__0));
v___x_96_ = l_Lean_Meta_instAlternativeMetaM;
v___x_97_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_97_, 0, v___x_96_);
lean_ctor_set(v___x_97_, 1, v___f_95_);
return v___x_97_;
}
}
static lean_object* _init_lp_batteries_instAlternativeMonadMetaM(void){
_start:
{
lean_object* v___x_98_; 
v___x_98_ = lean_obj_once(&lp_batteries_instAlternativeMonadMetaM___closed__1, &lp_batteries_instAlternativeMonadMetaM___closed__1_once, _init_lp_batteries_instAlternativeMonadMetaM___closed__1);
return v___x_98_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Control_Lemmas(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Control_OptionT(uint8_t builtin);
lean_object* runtime_initialize_Init_Control_Option(uint8_t builtin);
lean_object* runtime_initialize_Init_Control_State(uint8_t builtin);
lean_object* runtime_initialize_Init_Control_Reader(uint8_t builtin);
lean_object* runtime_initialize_Init_Control_StateRef(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Control_AlternativeMonad(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Control_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Control_OptionT(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Init_Control_Option(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Init_Control_State(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Init_Control_Reader(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Init_Control_StateRef(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_batteries_instAlternativeMonadMetaM = _init_lp_batteries_instAlternativeMonadMetaM();
lean_mark_persistent(lp_batteries_instAlternativeMonadMetaM);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Control_AlternativeMonad(uint8_t builtin) {
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
lean_object* initialize_batteries_Batteries_Control_Lemmas(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Control_OptionT(uint8_t builtin);
lean_object* initialize_Init_Control_Option(uint8_t builtin);
lean_object* initialize_Init_Control_State(uint8_t builtin);
lean_object* initialize_Init_Control_Reader(uint8_t builtin);
lean_object* initialize_Init_Control_StateRef(uint8_t builtin);
lean_object* initialize_Lean_Meta_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Control_AlternativeMonad(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Control_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Control_OptionT(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init_Control_Option(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init_Control_State(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init_Control_Reader(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init_Control_StateRef(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Control_AlternativeMonad(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Control_AlternativeMonad(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Control_AlternativeMonad(builtin);
}
#ifdef __cplusplus
}
#endif
