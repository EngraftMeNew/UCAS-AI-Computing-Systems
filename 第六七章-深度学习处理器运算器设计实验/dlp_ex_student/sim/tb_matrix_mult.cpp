#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cstdint>
#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vmatrix_mult_4x4x4x16.h"

using namespace std;

vluint64_t sim_time = 0;

// 加载矩阵维度配置
bool load_config(const string& path, int& rowsA, int& colsA, int& colsB) {
    ifstream f(path);
    if (!f) { cerr << "ERROR: 无法打开config文件: " << path << endl; return false; }
    string line;
    if (!getline(f, line)) { cerr << "ERROR: config文件为空" << endl; return false; }

    stringstream ss(line);
    string a, b, c;
    if (!(ss >> a >> b >> c)) {
        cerr << "ERROR: config文件格式错误" << endl;
        return false;
    }

    try {
        rowsA = stoi(a, nullptr, 16);
        colsA = stoi(b, nullptr, 16);
        colsB = stoi(c, nullptr, 16);
        cout << "INFO: 矩阵维度（16进制转换为10进制）: "
             << "A(" << rowsA << "x" << colsA << "), "
             << "B(" << colsA << "x" << colsB << "), "
             << "结果(" << rowsA << "x" << colsB << ")" << endl;
        return true;
    } catch (...) {
        cerr << "ERROR: 解析config失败" << endl;
        return false;
    }
}

// 加载16位矩阵数据
bool load_16bit_matrix(const string& path, vector<int16_t>& data) {
    ifstream f(path);
    if (!f) { cerr << "ERROR: 无法打开文件: " << path << endl; return false; }

    string line;
    data.clear();
    while (getline(f, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string bin;
        while (ss >> bin) {
            if (bin.size() != 16) {
                cerr << "ERROR: 16位数据长度错误: " << bin.size() << "位" << endl;
                return false;
            }
            try {
                uint16_t bits = stoi(bin, nullptr, 2);
                data.push_back(static_cast<int16_t>(bits));
            } catch (...) {
                cerr << "ERROR: 无效16位数据: " << bin << endl;
                return false;
            }
        }
    }

    cout << "INFO: " << path << " 加载 " << data.size() << " 个数据" << endl;
    return true;
}

// 加载32位结果数据
bool load_32bit_results(const string& path, vector<int32_t>& results) {
    ifstream f(path);
    if (!f) { cerr << "ERROR: 无法打开结果文件: " << path << endl; return false; }

    string line;
    results.clear();
    while (getline(f, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string bin;
        while (ss >> bin) {
            if (bin.size() != 32) {
                cerr << "ERROR: 32位结果长度错误: " << bin.size() << "位" << endl;
                return false;
            }
            try {
                uint32_t bits = stoul(bin, nullptr, 2);
                results.push_back(static_cast<int32_t>(bits));
            } catch (...) {
                cerr << "ERROR: 无效32位结果: " << bin << endl;
                return false;
            }
        }
    }

    cout << "INFO: 结果文件加载 " << results.size() << " 个结果" << endl;
    return true;
}

int main(int argc, char**argv) {
    Verilated::commandArgs(argc, argv);
    Verilated::traceEverOn(true);

    cout << "\n=== 矩阵乘矩阵测试 ===" << endl;

    int rowsA, colsA, colsB;
    vector<int16_t> activations, weights;
    vector<int32_t> expected_results, actual_results;

    // 加载配置和数据
    if (!load_config("/mnt/c/CODE/dlp_ex_student/data/mm/config", rowsA, colsA, colsB) ||
        !load_16bit_matrix("/mnt/c/CODE/dlp_ex_student/data/mm/neuron", activations) ||
        !load_16bit_matrix("/mnt/c/CODE/dlp_ex_student/data/mm/weight", weights) ||
        !load_32bit_results("/mnt/c/CODE/dlp_ex_student/data/mm/result", expected_results)) {
        return 1;
    }

    // 验证数据长度
    if (activations.size() != (size_t)(rowsA * colsA) ||
        weights.size() != (size_t)(colsA * colsB) ||
        expected_results.size() != (size_t)(rowsA * colsB)) {
        cerr << "ERROR: 数据量不匹配（激活值=" << activations.size() << ", 权重=" << weights.size()
             << ", 预期结果=" << expected_results.size() << "）" << endl;
        return 1;
    }

    // 打印输入数据
    cout << "\n=== 输入数据 ===" << endl;
    cout << "激活值矩阵（activations，" << rowsA << "x" << colsA << "）:" << endl;
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsA; j++) {
            cout << (int)activations[i * colsA + j] << "\t";
        }
        cout << endl;
    }

    cout << "\n权重矩阵（weights，" << colsA << "x" << colsB << "）:" << endl;
    for (int i = 0; i < colsA; i++) {
        for (int j = 0; j < colsB; j++) {
            cout << (int)weights[i * colsB + j] << "\t";
        }
        cout << endl;
    }

    // 初始化DUT
    Vmatrix_mult_4x4x4x16* dut = new Vmatrix_mult_4x4x4x16;
    VerilatedVcdC* wave = new VerilatedVcdC;
    dut->trace(wave, 99);
    wave->open("wave_matrix_mult.vcd");

    // 初始化信号
    dut->clk = 0;
    dut->reset = 1;
    dut->enable = 0;

    // 清零输入矩阵
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            dut->activations[i][j] = 0;
            dut->weights[i][j] = 0;
        }
    }

    // 仿真参数
    const int CLK_PERIOD = 2;
    bool result_captured = false;

    // 仿真主循环
    while (sim_time < 120) {
        // 时钟翻转
        if (sim_time % (CLK_PERIOD / 2) == 0) {
            dut->clk = !dut->clk;
        }

        // 释放复位
        if (sim_time == 5) {
            dut->reset = 0;
        }

        // 加载矩阵数据
        if (sim_time == 9 && dut->clk == 0) {
            for (int i = 0; i < rowsA; i++) {
                for (int j = 0; j < colsA; j++) {
                    dut->activations[i][j] = activations[i * colsA + j];
                }
            }
            for (int i = 0; i < colsA; i++) {
                for (int j = 0; j < colsB; j++) {
                    dut->weights[i][j] = weights[i * colsB + j];
                }
            }
        }

        // 启动计算
        if (sim_time == 11 && dut->clk == 0) {
            dut->enable = 1;
        }

        // 关闭使能
        if (sim_time == 19 && dut->clk == 0) {
            dut->enable = 0;
        }

        // 捕获结果
        if (dut->clk == 1 && dut->reset == 0 && !result_captured && sim_time > 60) {
            for (int i = 0; i < rowsA; i++) {
                for (int j = 0; j < colsB; j++) {
                    actual_results.push_back(dut->results[i][j]);
                }
            }
            result_captured = true;
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
    bool all_pass = true;
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            int idx = i * colsB + j;
            bool pass = (actual_results[idx] == expected_results[idx]);

            cout << "结果[" << i << "][" << j << "]: 实际=" << actual_results[idx]
                 << ", 预期=" << expected_results[idx]
                 << " [" << (pass ? "通过" : "失败") << "]" << endl;

            if (!pass) all_pass = false;
        }
    }

    cout << "\n验证结果: " << (all_pass ? "全部通过" : "存在失败") << endl;
    return all_pass ? 0 : 1;
}
