#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vinner_product_4x16.h"

using namespace std;

vluint64_t sim_time = 0;

// 读取16位二进制数据
bool load_16bit_data(const string& path, vector<int16_t>& data) {
    ifstream f(path);
    if (!f) { cerr << "ERROR: 无法打开文件: " << path << endl; return false; }
    
    string line;
    data.clear();
    while (getline(f, line)) {
        if (line.empty() || line[0] == '/') continue;
        size_t comment_pos = line.find("//");
        if (comment_pos != string::npos) line = line.substr(0, comment_pos);
        
        stringstream ss(line);
        string bin;
        if (ss >> bin && bin.size() == 16) {
            try {
                int value = stoi(bin, nullptr, 2);
                data.push_back(static_cast<int16_t>(value));
            } catch (...) {
                cerr << "WARN: 无效数据: " << bin << endl;
            }
        }
    }
    cout << "INFO: " << path << " 加载 " << data.size() << " 个数据" << endl;
    return true;
}

// 读取32位二进制结果
bool load_32bit_result(const string& path, int32_t& result) {
    ifstream f(path);
    if (!f) { cerr << "ERROR: 无法打开结果文件: " << path << endl; return false; }
    
    string line;
    if (getline(f, line)) {
        size_t comment_pos = line.find("//");
        if (comment_pos != string::npos) line = line.substr(0, comment_pos);
        
        stringstream ss(line);
        string bin;
        if (ss >> bin && bin.size() == 32) {
            try {
                long long value = stoll(bin, nullptr, 2);
                result = static_cast<int32_t>(value);
                cout << "INFO: 预期结果加载成功（十进制: " << result << "）" << endl;
                return true;
            } catch (...) {
                cerr << "ERROR: 无效结果数据: " << bin << endl;
            }
        }
    }
    cerr << "ERROR: 未找到有效结果数据" << endl;
    return false;
}

int main(int argc, char**argv) {
    Verilated::commandArgs(argc, argv);
    Verilated::traceEverOn(true);

    cout << "\n=== 内积运算器测试（单组数据）===" << endl;

    // 加载数据
    vector<int16_t> neuron, weight;
    int32_t expected_result;
    
    if (!load_16bit_data("/mnt/c/CODE/dlp_ex_student/data/mv/neuron", neuron) ||
        !load_16bit_data("/mnt/c/CODE/dlp_ex_student/data/mv/weight", weight) ||
        !load_32bit_result("/mnt/c/CODE/dlp_ex_student/data/mv/result", expected_result)) {
        return 1;
    }

    // 验证数据量（至少4）
    if (neuron.size() < 4 || weight.size() < 4) {
        cerr << "ERROR: 数据量必须至少4个（当前neuron=" << neuron.size()
             << ", weight=" << weight.size() << "）" << endl;
        return 1;
    }

    // 打印加载的数据（调试用）
    cout << "\n=== 输入数据 ===" << endl;
    cout << "神经元数据（十进制）: ";
    for (int i = 0; i < 4; i++) cout << (int)neuron[i] << " ";
    cout << "\n权重数据（十进制）:   ";
    for (int i = 0; i < 4; i++) cout << (int)weight[i] << " ";
    cout << "\n预期结果（十进制）:   " << expected_result << endl;

    // 初始化DUT和波形
    Vinner_product_4x16* dut = new Vinner_product_4x16;
    VerilatedVcdC* wave = new VerilatedVcdC;
    dut->trace(wave, 99);
    wave->open("wave_inner.vcd");

    // 初始化信号
    dut->clk = 0;
    dut->reset = 1;  // 初始复位
    dut->enable = 0;
    for (int i = 0; i < 4; i++) {
        dut->activations[i] = 0;
        dut->weights[i] = 0;
    }

    // 仿真参数
    const int CLK_PERIOD = 2;  // 完整时钟周期
    int32_t actual_result = 0;
    bool result_captured = false;

    // 仿真循环 - 延长时间确保结果稳定输出
    while (sim_time < 60) {
        // 时钟翻转（每个时间单位翻转一次）
        if (sim_time % (CLK_PERIOD / 2) == 0) {
            dut->clk = !dut->clk;
            // 打印时钟上升沿，帮助调试时序
            if (dut->clk == 1) {
                cout << "INFO: 时钟上升沿 @ 时间 " << sim_time << endl;
            }
        }

        // 第5个时间单位释放复位
        if (sim_time == 5) {
            dut->reset = 0;
            cout << "INFO: 释放复位 @ 时间 " << sim_time << endl;
        }

        // 在时钟下降沿加载数据（确保数据在上升沿前稳定）
        if (sim_time == 9 && dut->clk == 0) {
            for (int i = 0; i < 4; i++) {
                dut->activations[i] = neuron[i];
                dut->weights[i] = weight[i];
            }
            cout << "INFO: 加载数据 @ 时间 " << sim_time << endl;
        }

        // 在数据加载后的下降沿使能计算
        if (sim_time == 11 && dut->clk == 0) {
            dut->enable = 1;
            cout << "INFO: 使能计算 @ 时间 " << sim_time << endl;
        }

        // 保持使能至少一个完整周期
        if (sim_time == 15 && dut->clk == 0) {
            dut->enable = 0;
            cout << "INFO: 关闭使能 @ 时间 " << sim_time << endl;
        }

        // 关键修改：在使能关闭后至少等待2个时钟周期再捕获结果
        // 匹配模块的寄存器延迟特性
        if (dut->clk == 1 && dut->reset == 0 && !result_captured && sim_time > 25) {
            actual_result = dut->result;
            result_captured = true;
            cout << "INFO: 在时间 " << sim_time << " 捕获结果" << endl;
        }

        dut->eval();
        wave->dump(sim_time);
        sim_time++;
    }

    // 清理资源
    wave->close();
    delete dut;
    delete wave;

    // 结果验证
    cout << "\n=== 结果验证 ===" << endl;
    cout << "实际结果（十进制）: " << actual_result << endl;
    cout << "预期结果（十进制）: " << expected_result << endl;
    cout << "验证结果: " << (actual_result == expected_result ? "通过" : "失败") << endl;

    return (actual_result == expected_result) ? 0 : 1;
}
