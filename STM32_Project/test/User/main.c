#include "stm32f10x.h"
#include "usart.h"
#include "sys_tick.h"
#include "fashion_star_uart_servo.h"
#include <stdlib.h> // 用于 abs()

// --- 串口定义 ---
Usart_DataTypeDef* SERVO_UART = &usart1;  // 控制舵机的串口
Usart_DataTypeDef* OPENMV_UART = &usart2; // 接收PC数据的串口

// --- 舵机ID定义 ---
#define PAN_SERVO_ID  0  // 水平舵机 (Pan)
#define TILT_SERVO_ID 1  // 垂直舵机 (Tilt)

// ======================= 可调参数 (在这里调整云台性能) =======================

// 1. 增量转换比例 (非常重要)
//    这个值决定了PC发来的增量指令 (如 138, 比中心128大了10) 转换成舵机角度的幅度
//    值越大，云台对指令的反应越剧烈、越快；值越小，云台运动越平滑、越慢
#define INCREMENT_SCALAR        0.08f 

// 2. 舵机角度限位 (安全保护，必须根据你的云台结构设置)
#define PAN_ANGLE_MIN           -135.0f  // 水平最小角度
#define PAN_ANGLE_MAX           135.0f   // 水平最大角度
#define TILT_ANGLE_MIN          -135.0f  // 垂直最小角度 (请根据你的云台结构修改)
#define TILT_ANGLE_MAX          135.0f   // 垂直最大角度 (请根据你的云台结构修改)

// 3. 舵机中心位置
#define PAN_CENTER_ANGLE        0.0f
#define TILT_CENTER_ANGLE       0.0f
// ===========================================================================


// --- 全局变量，用于保存和累积舵机当前的目标角度 ---
float g_pan_angle = PAN_CENTER_ANGLE;
float g_tilt_angle = TILT_CENTER_ANGLE;


int main(void)
{
    // --- 系统初始化 ---
    SysTick_Init();
    Usart_Init();

    // --- 舵机归位到中间位置 ---
    FSUS_SetServoAngle(SERVO_UART, PAN_SERVO_ID, g_pan_angle, 1000, 0);
    FSUS_SetServoAngle(SERVO_UART, TILT_SERVO_ID, g_tilt_angle, 1000, 0);
    SysTick_DelayMs(1500); // 等待舵机归位完成

    // --- 主循环 ---
    while (1)
    {
	
    }
}
