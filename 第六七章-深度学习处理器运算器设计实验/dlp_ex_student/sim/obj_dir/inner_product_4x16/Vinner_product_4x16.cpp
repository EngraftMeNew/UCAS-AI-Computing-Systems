// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Model implementation (design independent parts)

#include "Vinner_product_4x16__pch.h"
#include "verilated_vcd_c.h"

//============================================================
// Constructors

Vinner_product_4x16::Vinner_product_4x16(VerilatedContext* _vcontextp__, const char* _vcname__)
    : VerilatedModel{*_vcontextp__}
    , vlSymsp{new Vinner_product_4x16__Syms(contextp(), _vcname__, this)}
    , clk{vlSymsp->TOP.clk}
    , reset{vlSymsp->TOP.reset}
    , enable{vlSymsp->TOP.enable}
    , result{vlSymsp->TOP.result}
    , activations{vlSymsp->TOP.activations}
    , weights{vlSymsp->TOP.weights}
    , rootp{&(vlSymsp->TOP)}
{
    // Register model with the context
    contextp()->addModel(this);
}

Vinner_product_4x16::Vinner_product_4x16(const char* _vcname__)
    : Vinner_product_4x16(Verilated::threadContextp(), _vcname__)
{
}

//============================================================
// Destructor

Vinner_product_4x16::~Vinner_product_4x16() {
    delete vlSymsp;
}

//============================================================
// Evaluation function

#ifdef VL_DEBUG
void Vinner_product_4x16___024root___eval_debug_assertions(Vinner_product_4x16___024root* vlSelf);
#endif  // VL_DEBUG
void Vinner_product_4x16___024root___eval_static(Vinner_product_4x16___024root* vlSelf);
void Vinner_product_4x16___024root___eval_initial(Vinner_product_4x16___024root* vlSelf);
void Vinner_product_4x16___024root___eval_settle(Vinner_product_4x16___024root* vlSelf);
void Vinner_product_4x16___024root___eval(Vinner_product_4x16___024root* vlSelf);

void Vinner_product_4x16::eval_step() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate Vinner_product_4x16::eval_step\n"); );
#ifdef VL_DEBUG
    // Debug assertions
    Vinner_product_4x16___024root___eval_debug_assertions(&(vlSymsp->TOP));
#endif  // VL_DEBUG
    vlSymsp->__Vm_activity = true;
    vlSymsp->__Vm_deleter.deleteAll();
    if (VL_UNLIKELY(!vlSymsp->__Vm_didInit)) {
        vlSymsp->__Vm_didInit = true;
        VL_DEBUG_IF(VL_DBG_MSGF("+ Initial\n"););
        Vinner_product_4x16___024root___eval_static(&(vlSymsp->TOP));
        Vinner_product_4x16___024root___eval_initial(&(vlSymsp->TOP));
        Vinner_product_4x16___024root___eval_settle(&(vlSymsp->TOP));
    }
    VL_DEBUG_IF(VL_DBG_MSGF("+ Eval\n"););
    Vinner_product_4x16___024root___eval(&(vlSymsp->TOP));
    // Evaluate cleanup
    Verilated::endOfEval(vlSymsp->__Vm_evalMsgQp);
}

//============================================================
// Events and timing
bool Vinner_product_4x16::eventsPending() { return false; }

uint64_t Vinner_product_4x16::nextTimeSlot() {
    VL_FATAL_MT(__FILE__, __LINE__, "", "%Error: No delays in the design");
    return 0;
}

//============================================================
// Utilities

const char* Vinner_product_4x16::name() const {
    return vlSymsp->name();
}

//============================================================
// Invoke final blocks

void Vinner_product_4x16___024root___eval_final(Vinner_product_4x16___024root* vlSelf);

VL_ATTR_COLD void Vinner_product_4x16::final() {
    Vinner_product_4x16___024root___eval_final(&(vlSymsp->TOP));
}

//============================================================
// Implementations of abstract methods from VerilatedModel

const char* Vinner_product_4x16::hierName() const { return vlSymsp->name(); }
const char* Vinner_product_4x16::modelName() const { return "Vinner_product_4x16"; }
unsigned Vinner_product_4x16::threads() const { return 1; }
void Vinner_product_4x16::prepareClone() const { contextp()->prepareClone(); }
void Vinner_product_4x16::atClone() const {
    contextp()->threadPoolpOnClone();
}
std::unique_ptr<VerilatedTraceConfig> Vinner_product_4x16::traceConfig() const {
    return std::unique_ptr<VerilatedTraceConfig>{new VerilatedTraceConfig{false, false, false}};
};

//============================================================
// Trace configuration

void Vinner_product_4x16___024root__trace_decl_types(VerilatedVcd* tracep);

void Vinner_product_4x16___024root__trace_init_top(Vinner_product_4x16___024root* vlSelf, VerilatedVcd* tracep);

VL_ATTR_COLD static void trace_init(void* voidSelf, VerilatedVcd* tracep, uint32_t code) {
    // Callback from tracep->open()
    Vinner_product_4x16___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vinner_product_4x16___024root*>(voidSelf);
    Vinner_product_4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    if (!vlSymsp->_vm_contextp__->calcUnusedSigs()) {
        VL_FATAL_MT(__FILE__, __LINE__, __FILE__,
            "Turning on wave traces requires Verilated::traceEverOn(true) call before time 0.");
    }
    vlSymsp->__Vm_baseCode = code;
    tracep->pushPrefix(std::string{vlSymsp->name()}, VerilatedTracePrefixType::SCOPE_MODULE);
    Vinner_product_4x16___024root__trace_decl_types(tracep);
    Vinner_product_4x16___024root__trace_init_top(vlSelf, tracep);
    tracep->popPrefix();
}

VL_ATTR_COLD void Vinner_product_4x16___024root__trace_register(Vinner_product_4x16___024root* vlSelf, VerilatedVcd* tracep);

VL_ATTR_COLD void Vinner_product_4x16::trace(VerilatedVcdC* tfp, int levels, int options) {
    if (tfp->isOpen()) {
        vl_fatal(__FILE__, __LINE__, __FILE__,"'Vinner_product_4x16::trace()' shall not be called after 'VerilatedVcdC::open()'.");
    }
    if (false && levels && options) {}  // Prevent unused
    tfp->spTrace()->addModel(this);
    tfp->spTrace()->addInitCb(&trace_init, &(vlSymsp->TOP));
    Vinner_product_4x16___024root__trace_register(&(vlSymsp->TOP), tfp->spTrace());
}
