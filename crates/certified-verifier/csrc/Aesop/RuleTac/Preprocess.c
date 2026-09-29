// Lean compiler output
// Module: Aesop.RuleTac.Preprocess
// Imports: public import Init public meta import Init public import Aesop.RuleTac.Basic public import Aesop.Script.SpecificTactics
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
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lp_aesop_Aesop_renameInaccessibleFVarsS___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_preprocess_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_preprocess_spec__0___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_preprocess_spec__0___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_preprocess_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_preprocess_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_preprocess_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_preprocess_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_RuleTac_preprocess___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleTac_preprocess___closed__0;
static lean_once_cell_t lp_aesop_Aesop_RuleTac_preprocess___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleTac_preprocess___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_preprocess(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_preprocess___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_preprocess_spec__0___redArg(lean_object* v_x_3_, lean_object* v___y_4_, lean_object* v___y_5_, lean_object* v___y_6_, lean_object* v___y_7_, lean_object* v___y_8_){
_start:
{
lean_object* v___x_10_; lean_object* v___x_11_; lean_object* v___x_12_; 
v___x_10_ = ((lean_object*)(lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_preprocess_spec__0___redArg___closed__0));
v___x_11_ = lean_st_mk_ref(v___x_10_);
lean_inc(v___y_8_);
lean_inc_ref(v___y_7_);
lean_inc(v___y_6_);
lean_inc_ref(v___y_5_);
lean_inc(v___y_4_);
lean_inc(v___x_11_);
v___x_12_ = lean_apply_7(v_x_3_, v___x_11_, v___y_4_, v___y_5_, v___y_6_, v___y_7_, v___y_8_, lean_box(0));
if (lean_obj_tag(v___x_12_) == 0)
{
lean_object* v_a_13_; lean_object* v___x_15_; uint8_t v_isShared_16_; uint8_t v_isSharedCheck_22_; 
v_a_13_ = lean_ctor_get(v___x_12_, 0);
v_isSharedCheck_22_ = !lean_is_exclusive(v___x_12_);
if (v_isSharedCheck_22_ == 0)
{
v___x_15_ = v___x_12_;
v_isShared_16_ = v_isSharedCheck_22_;
goto v_resetjp_14_;
}
else
{
lean_inc(v_a_13_);
lean_dec(v___x_12_);
v___x_15_ = lean_box(0);
v_isShared_16_ = v_isSharedCheck_22_;
goto v_resetjp_14_;
}
v_resetjp_14_:
{
lean_object* v___x_17_; lean_object* v___x_18_; lean_object* v___x_20_; 
v___x_17_ = lean_st_ref_get(v___x_11_);
lean_dec(v___x_11_);
v___x_18_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_18_, 0, v_a_13_);
lean_ctor_set(v___x_18_, 1, v___x_17_);
if (v_isShared_16_ == 0)
{
lean_ctor_set(v___x_15_, 0, v___x_18_);
v___x_20_ = v___x_15_;
goto v_reusejp_19_;
}
else
{
lean_object* v_reuseFailAlloc_21_; 
v_reuseFailAlloc_21_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_21_, 0, v___x_18_);
v___x_20_ = v_reuseFailAlloc_21_;
goto v_reusejp_19_;
}
v_reusejp_19_:
{
return v___x_20_;
}
}
}
else
{
lean_object* v_a_23_; lean_object* v___x_25_; uint8_t v_isShared_26_; uint8_t v_isSharedCheck_30_; 
lean_dec(v___x_11_);
v_a_23_ = lean_ctor_get(v___x_12_, 0);
v_isSharedCheck_30_ = !lean_is_exclusive(v___x_12_);
if (v_isSharedCheck_30_ == 0)
{
v___x_25_ = v___x_12_;
v_isShared_26_ = v_isSharedCheck_30_;
goto v_resetjp_24_;
}
else
{
lean_inc(v_a_23_);
lean_dec(v___x_12_);
v___x_25_ = lean_box(0);
v_isShared_26_ = v_isSharedCheck_30_;
goto v_resetjp_24_;
}
v_resetjp_24_:
{
lean_object* v___x_28_; 
if (v_isShared_26_ == 0)
{
v___x_28_ = v___x_25_;
goto v_reusejp_27_;
}
else
{
lean_object* v_reuseFailAlloc_29_; 
v_reuseFailAlloc_29_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_29_, 0, v_a_23_);
v___x_28_ = v_reuseFailAlloc_29_;
goto v_reusejp_27_;
}
v_reusejp_27_:
{
return v___x_28_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_preprocess_spec__0___redArg___boxed(lean_object* v_x_31_, lean_object* v___y_32_, lean_object* v___y_33_, lean_object* v___y_34_, lean_object* v___y_35_, lean_object* v___y_36_, lean_object* v___y_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_preprocess_spec__0___redArg(v_x_31_, v___y_32_, v___y_33_, v___y_34_, v___y_35_, v___y_36_);
lean_dec(v___y_36_);
lean_dec_ref(v___y_35_);
lean_dec(v___y_34_);
lean_dec_ref(v___y_33_);
lean_dec(v___y_32_);
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_preprocess_spec__0(lean_object* v_00_u03b1_39_, lean_object* v_x_40_, lean_object* v___y_41_, lean_object* v___y_42_, lean_object* v___y_43_, lean_object* v___y_44_, lean_object* v___y_45_){
_start:
{
lean_object* v___x_47_; 
v___x_47_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_preprocess_spec__0___redArg(v_x_40_, v___y_41_, v___y_42_, v___y_43_, v___y_44_, v___y_45_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_preprocess_spec__0___boxed(lean_object* v_00_u03b1_48_, lean_object* v_x_49_, lean_object* v___y_50_, lean_object* v___y_51_, lean_object* v___y_52_, lean_object* v___y_53_, lean_object* v___y_54_, lean_object* v___y_55_){
_start:
{
lean_object* v_res_56_; 
v_res_56_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_preprocess_spec__0(v_00_u03b1_48_, v_x_49_, v___y_50_, v___y_51_, v___y_52_, v___y_53_, v___y_54_);
lean_dec(v___y_54_);
lean_dec_ref(v___y_53_);
lean_dec(v___y_52_);
lean_dec_ref(v___y_51_);
lean_dec(v___y_50_);
return v_res_56_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleTac_preprocess___closed__0(void){
_start:
{
lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; 
v___x_57_ = lean_box(0);
v___x_58_ = lean_unsigned_to_nat(16u);
v___x_59_ = lean_mk_array(v___x_58_, v___x_57_);
return v___x_59_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleTac_preprocess___closed__1(void){
_start:
{
lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; 
v___x_60_ = lean_obj_once(&lp_aesop_Aesop_RuleTac_preprocess___closed__0, &lp_aesop_Aesop_RuleTac_preprocess___closed__0_once, _init_lp_aesop_Aesop_RuleTac_preprocess___closed__0);
v___x_61_ = lean_unsigned_to_nat(0u);
v___x_62_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_62_, 0, v___x_61_);
lean_ctor_set(v___x_62_, 1, v___x_60_);
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_preprocess(lean_object* v_a_63_, lean_object* v_a_64_, lean_object* v_a_65_, lean_object* v_a_66_, lean_object* v_a_67_, lean_object* v_a_68_){
_start:
{
lean_object* v_goal_70_; lean_object* v___x_71_; lean_object* v___x_72_; 
v_goal_70_ = lean_ctor_get(v_a_63_, 0);
lean_inc_n(v_goal_70_, 2);
lean_dec_ref(v_a_63_);
v___x_71_ = lean_alloc_closure((void*)(lp_aesop_Aesop_renameInaccessibleFVarsS___boxed), 8, 1);
lean_closure_set(v___x_71_, 0, v_goal_70_);
v___x_72_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_preprocess_spec__0___redArg(v___x_71_, v_a_64_, v_a_65_, v_a_66_, v_a_67_, v_a_68_);
if (lean_obj_tag(v___x_72_) == 0)
{
lean_object* v_a_73_; lean_object* v_fst_74_; lean_object* v_snd_75_; lean_object* v_fst_76_; lean_object* v___x_77_; uint8_t v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; 
v_a_73_ = lean_ctor_get(v___x_72_, 0);
lean_inc(v_a_73_);
lean_dec_ref_known(v___x_72_, 1);
v_fst_74_ = lean_ctor_get(v_a_73_, 0);
lean_inc(v_fst_74_);
v_snd_75_ = lean_ctor_get(v_a_73_, 1);
lean_inc(v_snd_75_);
lean_dec(v_a_73_);
v_fst_76_ = lean_ctor_get(v_fst_74_, 0);
lean_inc(v_fst_76_);
lean_dec(v_fst_74_);
v___x_77_ = lean_obj_once(&lp_aesop_Aesop_RuleTac_preprocess___closed__1, &lp_aesop_Aesop_RuleTac_preprocess___closed__1_once, _init_lp_aesop_Aesop_RuleTac_preprocess___closed__1);
v___x_78_ = 0;
v___x_79_ = lean_alloc_ctor(0, 4, 1);
lean_ctor_set(v___x_79_, 0, v_goal_70_);
lean_ctor_set(v___x_79_, 1, v_fst_76_);
lean_ctor_set(v___x_79_, 2, v___x_77_);
lean_ctor_set(v___x_79_, 3, v___x_77_);
lean_ctor_set_uint8(v___x_79_, sizeof(void*)*4, v___x_78_);
v___x_80_ = l_Lean_Meta_saveState___redArg(v_a_66_, v_a_68_);
if (lean_obj_tag(v___x_80_) == 0)
{
lean_object* v_a_81_; lean_object* v___x_83_; uint8_t v_isShared_84_; uint8_t v_isSharedCheck_95_; 
v_a_81_ = lean_ctor_get(v___x_80_, 0);
v_isSharedCheck_95_ = !lean_is_exclusive(v___x_80_);
if (v_isSharedCheck_95_ == 0)
{
v___x_83_ = v___x_80_;
v_isShared_84_ = v_isSharedCheck_95_;
goto v_resetjp_82_;
}
else
{
lean_inc(v_a_81_);
lean_dec(v___x_80_);
v___x_83_ = lean_box(0);
v_isShared_84_ = v_isSharedCheck_95_;
goto v_resetjp_82_;
}
v_resetjp_82_:
{
lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_93_; 
v___x_85_ = lean_unsigned_to_nat(1u);
v___x_86_ = lean_mk_empty_array_with_capacity(v___x_85_);
lean_inc_ref(v___x_86_);
v___x_87_ = lean_array_push(v___x_86_, v___x_79_);
v___x_88_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_88_, 0, v_snd_75_);
v___x_89_ = lean_box(0);
v___x_90_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_90_, 0, v___x_87_);
lean_ctor_set(v___x_90_, 1, v_a_81_);
lean_ctor_set(v___x_90_, 2, v___x_88_);
lean_ctor_set(v___x_90_, 3, v___x_89_);
v___x_91_ = lean_array_push(v___x_86_, v___x_90_);
if (v_isShared_84_ == 0)
{
lean_ctor_set(v___x_83_, 0, v___x_91_);
v___x_93_ = v___x_83_;
goto v_reusejp_92_;
}
else
{
lean_object* v_reuseFailAlloc_94_; 
v_reuseFailAlloc_94_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_94_, 0, v___x_91_);
v___x_93_ = v_reuseFailAlloc_94_;
goto v_reusejp_92_;
}
v_reusejp_92_:
{
return v___x_93_;
}
}
}
else
{
lean_object* v_a_96_; lean_object* v___x_98_; uint8_t v_isShared_99_; uint8_t v_isSharedCheck_103_; 
lean_dec_ref_known(v___x_79_, 4);
lean_dec(v_snd_75_);
v_a_96_ = lean_ctor_get(v___x_80_, 0);
v_isSharedCheck_103_ = !lean_is_exclusive(v___x_80_);
if (v_isSharedCheck_103_ == 0)
{
v___x_98_ = v___x_80_;
v_isShared_99_ = v_isSharedCheck_103_;
goto v_resetjp_97_;
}
else
{
lean_inc(v_a_96_);
lean_dec(v___x_80_);
v___x_98_ = lean_box(0);
v_isShared_99_ = v_isSharedCheck_103_;
goto v_resetjp_97_;
}
v_resetjp_97_:
{
lean_object* v___x_101_; 
if (v_isShared_99_ == 0)
{
v___x_101_ = v___x_98_;
goto v_reusejp_100_;
}
else
{
lean_object* v_reuseFailAlloc_102_; 
v_reuseFailAlloc_102_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_102_, 0, v_a_96_);
v___x_101_ = v_reuseFailAlloc_102_;
goto v_reusejp_100_;
}
v_reusejp_100_:
{
return v___x_101_;
}
}
}
}
else
{
lean_object* v_a_104_; lean_object* v___x_106_; uint8_t v_isShared_107_; uint8_t v_isSharedCheck_111_; 
lean_dec(v_goal_70_);
v_a_104_ = lean_ctor_get(v___x_72_, 0);
v_isSharedCheck_111_ = !lean_is_exclusive(v___x_72_);
if (v_isSharedCheck_111_ == 0)
{
v___x_106_ = v___x_72_;
v_isShared_107_ = v_isSharedCheck_111_;
goto v_resetjp_105_;
}
else
{
lean_inc(v_a_104_);
lean_dec(v___x_72_);
v___x_106_ = lean_box(0);
v_isShared_107_ = v_isSharedCheck_111_;
goto v_resetjp_105_;
}
v_resetjp_105_:
{
lean_object* v___x_109_; 
if (v_isShared_107_ == 0)
{
v___x_109_ = v___x_106_;
goto v_reusejp_108_;
}
else
{
lean_object* v_reuseFailAlloc_110_; 
v_reuseFailAlloc_110_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_110_, 0, v_a_104_);
v___x_109_ = v_reuseFailAlloc_110_;
goto v_reusejp_108_;
}
v_reusejp_108_:
{
return v___x_109_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_preprocess___boxed(lean_object* v_a_112_, lean_object* v_a_113_, lean_object* v_a_114_, lean_object* v_a_115_, lean_object* v_a_116_, lean_object* v_a_117_, lean_object* v_a_118_){
_start:
{
lean_object* v_res_119_; 
v_res_119_ = lp_aesop_Aesop_RuleTac_preprocess(v_a_112_, v_a_113_, v_a_114_, v_a_115_, v_a_116_, v_a_117_);
lean_dec(v_a_117_);
lean_dec_ref(v_a_116_);
lean_dec(v_a_115_);
lean_dec_ref(v_a_114_);
lean_dec(v_a_113_);
return v_res_119_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_RuleTac_Basic(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Script_SpecificTactics(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_RuleTac_Preprocess(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleTac_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Script_SpecificTactics(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_RuleTac_Preprocess(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_RuleTac_Basic(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Script_SpecificTactics(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_RuleTac_Preprocess(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleTac_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Script_SpecificTactics(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleTac_Preprocess(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_RuleTac_Preprocess(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_RuleTac_Preprocess(builtin);
}
#ifdef __cplusplus
}
#endif
