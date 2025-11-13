#include "stm32f10x.h"
#include "usart.h"
#include "sys_tick.h"
#include "fashion_star_uart_servo.h"

Usart_DataTypeDef* SERVO_UART = &usart1; // 控制舵机的串口
Usart_DataTypeDef* OPENMV_UART = &usart2;
// 2. 定义舵机ID
#define PAN_SERVO_ID  0  // 水平舵机
#define TILT_SERVO_ID 1  // 垂直舵机

float pan_angle = 0.f;
float tilt_angle = 0.f;


int main(void)
{
    // --- 系统初始化 ---
    SysTick_Init(); // 初始化滴答定时器
    Usart_Init();   // 初始化串口
	
	FSUS_DampingMode(SERVO_UART, PAN_SERVO_ID, 500);
	FSUS_DampingMode(SERVO_UART, tilt_angle, 500);
	
    // --- 舵机归位到中间位置 (0度) ---
    FSUS_SetServoAngle(SERVO_UART, PAN_SERVO_ID, pan_angle, 1000, 0);
    FSUS_SetServoAngle(SERVO_UART, TILT_SERVO_ID, tilt_angle, 1000, 0);
    SysTick_DelayMs(1500);

    // --- 主循环 ---
    while (1)
    {
        // 检查串口接收缓冲区的数据是否足够一个包 (4字节)
        if (RingBuffer_GetByteUsed(OPENMV_UART->recvBuf) >= 4)
        {
            // 检查包头和包尾是否正确
            // RingBuffer_GetValueByIndex() 函数可以在不取出数据的情况下查看
            if (RingBuffer_GetValueByIndex(OPENMV_UART->recvBuf, 0) == 0xFF &&
                RingBuffer_GetValueByIndex(OPENMV_UART->recvBuf, 3) == 0xFE)
            {
                // 包头包尾都正确，这是一个有效的数据包
                uint8_t header, pan_raw, tilt_raw, footer;

                // 从缓冲区中按顺序取出4个字节
                header = RingBuffer_Pop(OPENMV_UART->recvBuf); // 丢弃 0xFF
                pan_raw = RingBuffer_Pop(OPENMV_UART->recvBuf); // 水平舵机原始值
                tilt_raw = RingBuffer_Pop(OPENMV_UART->recvBuf); // 垂直舵机原始值
                footer = RingBuffer_Pop(OPENMV_UART->recvBuf); // 丢弃 0xFE

                pan_angle = 128 - (float)pan_raw;
                tilt_angle = 128 - (float)tilt_raw;
				
				if (pan_angle > 135) pan_angle = 135;
				else if(pan_angle < -135) pan_angle = -135;
				if(tilt_angle > 135) tilt_angle = 135;
				else if(tilt_angle < -135) tilt_angle = -135;
				
                // 发送指令控制两个舵机
                FSUS_SetServoAngle(SERVO_UART, PAN_SERVO_ID, pan_angle, 20, 0);
                FSUS_SetServoAngle(SERVO_UART, TILT_SERVO_ID, tilt_angle, 20, 0);
            }
            else
            {
                // 包头或包尾不正确，说明数据乱了
                // 丢弃缓冲区里的第一个字节，然后继续寻找
                RingBuffer_Pop(OPENMV_UART->recvBuf);
            }
        }
    }
}
