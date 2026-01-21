三个运算器分别编译、运行测试、查看仿真波形
注：data文件夹中config和实验教材维度一致（4），其他数据需要自行更改并验证结果

1.编译命令：
make inner_product_4x16
make matrix_vector_mult_4x4x16
make matrix_mult_4x4x4x16
作用：使用Verilator将Verilog代码编译成C++可执行文件

2.运行测试命令
./obj_dir/inner_product_4x16/Vinner_product_4x16          # 测试内积运算器
./obj_dir/matrix_vector_mult_4x4x16/Vmatrix_vector_mult_4x4x16  # 测试矩阵乘向量
./obj_dir/matrix_mult_4x4x4x16/Vmatrix_mult_4x4x4x16      # 测试矩阵乘法
作用：运行编译好的测试程序，验证硬件模块功能是否正确，对比结果

3.查看波形命令
gtkwave wave_inner.vcd &             # 查看内积运算器波形
gtkwave wave_matrix_vector.vcd &     # 查看矩阵乘向量波形
gtkwave wave_matrix_mult.vcd &       # 查看矩阵乘法波形

