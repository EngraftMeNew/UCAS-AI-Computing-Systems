// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Model implementation (design independent parts)

#include "Vmatrix_vector_mult_4x4x16__pch.h"
#include "verilated_vcd_c.h"

//============================================================
// Constructors

Vmatrix_vector_mult_4x4x16::Vmatrix_vector_mult_4x4x16(VerilatedContext* _vcontextp__, const char* _vcname__)
    : VerilatedModel{*_vcontextp__}
    , vlSymsp{new Vmatrix_vector_mult_4x4x16__Syms(contextp(), _vcname__, this)}
    , clk{vlSymsp->TOP.clk}
    , reset{vlSymsp->TOP.reset}
    , enable{vlSymsp->TOP.enable}
    , activations{vlSymsp->TOP.activations}
    , weights{vlSymsp->TOP.weights}
    , results{vlSymsp->TOP.results}
    , rootp{&(vlSymsp->TOP)}
{
    // Register model with the context
    contextp()->addModel(this);
}

Vmatrix_vector_mult_4x4x16::Vmatrix_vector_mult_4x4x16(const char* _vcname__)
    : Vmatrix_vector_mult_4x4x16(Verilated::threadContextp(), _vcname__)
{
}

//============================================================
// Destructor

Vmatrix_vector_mult_4x4x16::~Vmatrix_vector_mult_4x4x16() {
    delete vlSymsp;
}

//============================================================
// Evaluation function

#ifdef VL_DEBUG
void Vmatrix_vector_mult_4x4x16___024root___eval_debug_assertions(Vmatrix_vector_mult_4x4x16___024root* vlSelf);
#endif  // VL_DEBUG
void Vmatrix_vector_mult_4x4x16___024root___eval_static(Vmatrix_vector_mult_4x4x16___024root* vlSelf);
void Vmatrix_vector_mult_4x4x16___024root___eval_initial(Vmatrix_vector_mult_4x4x16___024root* vlSelf);
void Vmatrix_vector_mult_4x4x16___024root___eval_settle(Vmatrix_vector_mult_4x4x16___024root* vlSelf);
void Vmatrix_vector_mult_4x4x16___024root___eval(Vmatrix_vector_mult_4x4x16___024root* vlSelf);

void Vmatrix_vector_mult_4x4x16::eval_step() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate Vmatrix_vector_mult_4x4x16::eval_step\n"); );
#ifdef VL_DEBUG
    // Debug assertions
    Vmatrix_vector_mult_4x4x16___024root___eval_debug_assertions(&(vlSymsp->TOP));
#endif  // VL_DEBUG
    vlSymsp->__Vm_activity = true;
    vlSymsp->__Vm_deleter.deleteAll();
    if (VL_UNLIKELY(!vlSymsp->__Vm_didInit)) {
        vlSymsp->__Vm_didInit = true;
        VL_DEBUG_IF(VL_DBG_MSGF("+ Initial\n"););
        Vmatrix_vector_mult_4x4x16___024root___eval_static(&(vlSymsp->TOP));
        Vmatrix_vector_mult_4x4x16___024root___eval_initial(&(vlSymsp->TOP));
        Vmatrix_vector_mult_4x4x16___024root___eval_settle(&(vlSymsp->TOP));
    }
    VL_DEBUG_IF(VL_DBG_MSGF("+ Eval\n"););
    Vmatrix_vector_mult_4x4x16___024root___eval(&(vlSymsp->TOP));
    // Evaluate cleanup
    Verilated::endOfEval(vlSymsp->__Vm_evalMsgQp);
}

//============================================================
// Events and timing
bool Vmatrix_vector_mult_4x4x16::eventsPending() { return false; }

uint64_t Vmatrix_vector_mult_4x4x16::nextTimeSlot() {
    VL_FATAL_MT(__FILE__, __LINE__, "", "%Error: No delays in the design");
    return 0;
}

//============================================================
// Utilities

const char* Vmatrix_vector_mult_4x4x16::name() const {
    return vlSymsp->name();
}

//============================================================
// Invoke final blocks

void Vmatrix_vector_mult_4x4x16___024root___eval_final(Vmatrix_vector_mult_4x4x16___024root* vlSelf);

VL_ATTR_COLD void Vmatrix_vector_mult_4x4x16::final() {
    Vmatrix_vector_mult_4x4x16___024root___eval_final(&(vlSymsp->TOP));
}

//============================================================
// Implementations of abstract methods from VerilatedModel

const char* Vmatrix_vector_mult_4x4x16::hierName() const { return vlSymsp->name(); }
const char* Vmatrix_vector_mult_4x4x16::modelName() const { return "Vmatrix_vector_mult_4x4x16"; }
unsigned Vmatrix_vector_mult_4x4x16::threads() const { return 1; }
void Vmatrix_vector_mult_4x4x16::prepareClone() const { contextp()->prepareClone(); }
void Vmatrix_vector_mult_4x4x16::atClone() const {
    contextp()->threadPoolpOnClone();
}
std::unique_ptr<VerilatedTraceConfig> Vmatrix_vector_mult_4x4x16::traceConfig() const {
    return std::unique_ptr<VerilatedTraceConfig>{new VerilatedTraceConfig{false, false, false}};
};

//============================================================
// Trace configuration

void Vmatrix_vector_mult_4x4x16___024root__trace_decl_types(VerilatedVcd* tracep);

void Vmatrix_vector_mult_4x4x16___024root__trace_init_top(Vmatrix_vector_mult_4x4x16___024root* vlSelf, VerilatedVcd* tracep);

VL_ATTR_COLD static void trace_init(void* voidSelf, VerilatedVcd* tracep, uint32_t code) {
    // Callback from tracep->open()
    Vmatrix_vector_mult_4x4x16___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vmatrix_vector_mult_4x4x16___024root*>(voidSelf);
    Vmatrix_vector_mult_4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    if (!vlSymsp->_vm_contextp__->calcUnusedSigs()) {
        VL_FATAL_MT(__FILE__, __LINE__, __FILE__,
            "Turning on wave traces requires Verilated::traceEverOn(true) call before time 0.");
    }
    vlSymsp->__Vm_baseCode = code;
    tracep->pushPrefix(std::string{vlSymsp->name()}, VerilatedTracePrefixType::SCOPE_MODULE);
    Vmatrix_vector_mult_4x4x16___024root__trace_decl_types(tracep);
    Vmatrix_vector_mult_4x4x16___024root__trace_init_top(vlSelf, tracep);
    tracep->popPrefix();
}

VL_ATTR_COLD void Vmatrix_vector_mult_4x4x16___024root__trace_register(Vmatrix_vector_mult_4x4x16___024root* vlSelf, VerilatedVcd* tracep);

VL_ATTR_COLD void Vmatrix_vector_mult_4x4x16::trace(VerilatedVcdC* tfp, int levels, int options) {
    if (tfp->isOpen()) {
        vl_fatal(__FILE__, __LINE__, __FILE__,"'Vmatrix_vector_mult_4x4x16::trace()' shall not be called after 'VerilatedVcdC::open()'.");
    }
    if (false && levels && options) {}  // Prevent unused
    tfp->spTrace()->addModel(this);
    tfp->spTrace()->addInitCb(&trace_init, &(vlSymsp->TOP));
    Vmatrix_vector_mult_4x4x16___024root__trace_register(&(vlSymsp->TOP), tfp->spTrace());
}
