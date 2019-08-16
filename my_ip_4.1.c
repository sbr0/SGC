/******************************************************************************
*
* Copyright (C) 2009 - 2014 Xilinx, Inc.  All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* Use of the Software is limited solely to applications:
* (a) running on a Xilinx device, or
* (b) that interact with a Xilinx device through a bus or interconnect.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
* XILINX  BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF
* OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*
* Except as contained in this notice, the name of the Xilinx shall not be used
* in advertising or otherwise to promote the sale, use or other dealings in
* this Software without prior written authorization from Xilinx.
*
******************************************************************************/

/*
 * helloworld.c: simple test application
 *
 * This application configures UART 16550 to baud rate 9600.
 * PS7 UART (Zynq) is not initialized by this application, since
 * bootrom/bsp configures it to baud rate 115200
 *
 * ------------------------------------------------
 * | UART TYPE   BAUD RATE                        |
 * ------------------------------------------------
 *   uartns550   9600
 *   uartlite    Configurable only in HW design
 *   ps7_uart    115200 (configured by bootrom/bsp)
 */

#include <stdio.h>
#include "platform.h"
#include "xil_printf.h"

#define P_M 32
#define P_N 7

#define MY_IP_BASE_ADDR 0xA0000000
#define A_VAL_BASE_ADDR 0x40
#define A_IDX_BASE_ADDR 0x60
#define B_VAL_BASE_ADDR 0x20
#define A_VAL_READY_REG 0x00
#define A_IDX_READY_REG 0x01
#define B_VAL_READY_REG 0x02
#define TEN_RESULTS_READY_REG 0x03 // 0x00000001 if ready, else 0x00000000
#define RESULT_DATA 0x04
#define SANITY_CHECK_REG 0x7F //Should return 0xABCDEF00


int main()
{
    init_platform();

    volatile unsigned int * my_ip = (volatile unsigned int *) MY_IP_BASE_ADDR;

    print("Hello World\n\r");
    printf("Sanity check register (should be 0xABCDEF00) = 0x%08x \n", my_ip[SANITY_CHECK_REG]);

    if (my_ip[A_VAL_READY_REG] & 0x00000001) {
    	my_ip[A_VAL_BASE_ADDR] = 0x00000000;
    }
    if (my_ip[A_IDX_READY_REG] & 0x00000001) {
      	my_ip[A_IDX_BASE_ADDR] = 0x00000003;
    }
    if (my_ip[B_VAL_READY_REG] & 0x00000001) {
       	my_ip[B_VAL_BASE_ADDR] = 0x41100000;
    }
    if (my_ip[A_VAL_READY_REG] & 0x00000001) {
        my_ip[A_VAL_BASE_ADDR] = 0x40100000;
	}
	if (my_ip[A_IDX_READY_REG] & 0x00000001) {
		my_ip[A_IDX_BASE_ADDR] = 0x00000000;
	}
	if (my_ip[B_VAL_READY_REG] & 0x00000001) {
		my_ip[B_VAL_BASE_ADDR] = 0x41200000;
	}
	if (my_ip[A_VAL_READY_REG] & 0x00000001) {
		my_ip[A_VAL_BASE_ADDR] = 0x40300000;
	}
	if (my_ip[A_IDX_READY_REG] & 0x00000001) {
		my_ip[A_IDX_BASE_ADDR] = 0x00000001;
	}
	if (my_ip[B_VAL_READY_REG] & 0x00000001) {
		my_ip[B_VAL_BASE_ADDR] = 0x41300000;
	}
	if (my_ip[A_VAL_READY_REG] & 0x00000001) {
		my_ip[A_VAL_BASE_ADDR] = 0x40500000;
	}
	if (my_ip[A_IDX_READY_REG] & 0x00000001) {
		my_ip[A_IDX_BASE_ADDR] = 0x00000002;
	}
	if (my_ip[B_VAL_READY_REG] & 0x00000001) {
		my_ip[B_VAL_BASE_ADDR] = 0x41400000;
	}
	if (my_ip[A_VAL_READY_REG] & 0x00000001) {
		my_ip[A_VAL_BASE_ADDR] = 0x00000001;
	}
	if (my_ip[A_IDX_READY_REG] & 0x00000001) {
		my_ip[A_IDX_BASE_ADDR] = 0x00000001;
	}
	if (my_ip[A_VAL_READY_REG] & 0x00000001) {
		my_ip[A_VAL_BASE_ADDR] = 0x40900000;
	}
	if (my_ip[A_IDX_READY_REG] & 0x00000001) {
		my_ip[A_IDX_BASE_ADDR] = 0x00000001;
	}
	if (my_ip[A_VAL_READY_REG] & 0x00000001) {
		my_ip[A_VAL_BASE_ADDR] = 0x00000002;
	}
	if (my_ip[A_IDX_READY_REG] & 0x00000001) {
		my_ip[A_IDX_BASE_ADDR] = 0x00000003;
	}
	if (my_ip[A_VAL_READY_REG] & 0x00000001) {
		my_ip[A_VAL_BASE_ADDR] = 0x40900000;
	}
	if (my_ip[A_IDX_READY_REG] & 0x00000001) {
		my_ip[A_IDX_BASE_ADDR] = 0x00000000;
	}
	if (my_ip[A_VAL_READY_REG] & 0x00000001) {
		my_ip[A_VAL_BASE_ADDR] = 0x40a00000;
	}
	if (my_ip[A_IDX_READY_REG] & 0x00000001) {
		my_ip[A_IDX_BASE_ADDR] = 0x00000001;
	}
	if (my_ip[A_VAL_READY_REG] & 0x00000001) {
		my_ip[A_VAL_BASE_ADDR] = 0x40b00000;
	}
	if (my_ip[A_IDX_READY_REG] & 0x00000001) {
		my_ip[A_IDX_BASE_ADDR] = 0x00000002;
	}
	if (my_ip[A_VAL_READY_REG] & 0x00000001) {
		my_ip[A_VAL_BASE_ADDR] = 0x00000003;
	}
	if (my_ip[A_IDX_READY_REG] & 0x00000001) {
		my_ip[A_IDX_BASE_ADDR] = 0x00000001;
	}
	if (my_ip[A_VAL_READY_REG] & 0x00000001) {
		my_ip[A_VAL_BASE_ADDR] = 0x40c00000;
	}
	if (my_ip[A_IDX_READY_REG] & 0x00000001) {
		my_ip[A_IDX_BASE_ADDR] = 0x00000001;
	}

	for (int i = 0 ; i < 10000; i++) {
		print(" ");
	}
	printf("\n");
    printf("value of sanity check register (should be 0xABCDEF00) = 0x%08x\n", my_ip[SANITY_CHECK_REG]);

	for (int i = 0; i < 5; i++) {
		printf("Result %d is : %f\n", i, my_ip[RESULT_DATA]);
	}

    cleanup_platform();
    return 0;
}
//
