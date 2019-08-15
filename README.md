# DDPG_CNN_Pendulum
I used the CNN + DDPG realizing inverted pendulum control
python3.5
tensorflow + GPU
gym环境


其中CNN_1与CNN_2是根据全连接进行改造的
CNN_1中是在第二个卷积层的输出中加入Actor网络的输出Policy
CNN_2中是在地一个全连接的输出中加入Actor网络的输出Policy

两个版本最后运行的结果都不太理想，卷积网路所使用的图像是使用pygame进行绘制的。

对于全连接网络来说，网络的输入有（角度的cos值，角度的sin值，角速度）
而卷积神经网络，我只使用了角度信息进行绘制，角速度信息没有使用到。

继续修改plot代码，进行训练。

添加了角速度的信息，以摆杆的支点为圆心画圆，当速度越大，半径越大，且具有方向信息，顺时针与逆时针使用颜色进行区分。

纪念第一次使用git😊
