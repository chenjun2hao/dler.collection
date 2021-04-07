#include<opencv2/opencv.hpp>
#include<arm_neon.h>
#include <chrono>
#include <stdlib.h>

using namespace cv;
using namespace std;

void rgb_deinterleave_neon(uint8_t *r, uint8_t *g, uint8_t *b, uint8_t *rgb, int len_color) {
    /*
     * Take the elements of "rgb" and store the individual colors "r", "g", and "b"
     */
    int num8x16 = len_color / 16;
    uint8x16x3_t intlv_rgb;
    for (int i=0; i < num8x16; i++) {
        intlv_rgb = vld3q_u8(rgb+3*16*i);
        vst1q_u8(r+16*i, intlv_rgb.val[0]);
        vst1q_u8(g+16*i, intlv_rgb.val[1]);
        vst1q_u8(b+16*i, intlv_rgb.val[2]);
    }
}


void neon_rgb_split(Mat src){
	int size = src.rows * src.cols;
	uint8_t *r, *g, *b;
	uint8_t *rgb = src.data;
	r = (uint8_t*)malloc(sizeof(uint8_t) * size);
	g = (uint8_t*)malloc(sizeof(uint8_t) * size);
	b = (uint8_t*)malloc(sizeof(uint8_t) * size);

	auto start = chrono::system_clock::now();
	rgb_deinterleave_neon(r, g, b, rgb, size);
	auto end   = chrono::system_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "neon cost time is:" << double(duration.count()) << " us" << endl;

	// Mat b_color(src.rows, src.cols, CV_8UC3);
	// Mat g_color(src.rows, src.cols, CV_8UC3);
	// Mat r_color(src.rows, src.cols, CV_8UC3);
	// size *=3;
	// for(int i=0; i<size; i+=3)
	// {
	// 	b_color.data[i] = b[i/3];
	// 	b_color.data[i+1] = 0;
	// 	b_color.data[i+2] = 0;
		
	// 	g_color.data[i] = 0;
	// 	g_color.data[i+1] = g[i/3];
	// 	g_color.data[i+2] = 0;
		
	// 	r_color.data[i] = 0;
	// 	r_color.data[i+1] = 0;
	// 	r_color.data[i+2] = r[i/3];
	// }
	// imwrite("b_neon.jpg", b_color);
	// imwrite("g_neon.jpg", g_color);
	// imwrite("r_neon.jpg", r_color);
	free(r);
	free(g);
	free(b);
}

void rgb_assembly_neon(uint8_t *r, uint8_t *g, uint8_t *b, uint8_t *rgb, int len_color) {

    asm volatile(
        "1:                                             \n"		// goto 标志位
		"prfm   pldl1keep, [%3, 384]    				\n"
        "ld3    {v0.16b, v1.16b, v2.16b}, [%3], #48     \n"
        "subs   %w4, %w4, #16           \n"
        "st1    {v0.16b}, [%2], #16     \n"     // b
        "st1    {v1.16b}, [%1], #16     \n"     // g
        "st1    {v2.16b}, [%0], #16     \n"     // r
        "b.gt   1b                      \n"
        : "+r"(r),
          "+r"(g),
          "+r"(b),
          "+r"(rgb),
          "+r"(len_color)
        :
        : "v0", "v1", "v2"          // use 3 个128bit寄存器
    );
}


void assembly_rgb_split(Mat src){
	int size = src.rows * src.cols;
	uint8_t *r, *g, *b;
	uint8_t *rgb = src.data;
	r = (uint8_t*)malloc(sizeof(uint8_t) * size);
	g = (uint8_t*)malloc(sizeof(uint8_t) * size);
	b = (uint8_t*)malloc(sizeof(uint8_t) * size);

	auto start = chrono::system_clock::now();
	rgb_assembly_neon(r, g, b, rgb, size);
	auto end   = chrono::system_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "assembly cost time is:" << double(duration.count()) << " us" << endl;

	// Mat b_color(src.rows, src.cols, CV_8UC3);
	// Mat g_color(src.rows, src.cols, CV_8UC3);
	// Mat r_color(src.rows, src.cols, CV_8UC3);
	// size *=3;
	// for(int i=0; i<size; i+=3)
	// {
	// 	b_color.data[i] = b[i/3];
	// 	b_color.data[i+1] = 0;
	// 	b_color.data[i+2] = 0;
		
	// 	g_color.data[i] = 0;
	// 	g_color.data[i+1] = g[i/3];
	// 	g_color.data[i+2] = 0;
		
	// 	r_color.data[i] = 0;
	// 	r_color.data[i+1] = 0;
	// 	r_color.data[i+2] = r[i/3];
	// }
	// imwrite("b_asm.jpg", b_color);
	// imwrite("g_asm.jpg", g_color);
	// imwrite("r_asm.jpg", r_color);
	free(r);
	free(g);
	free(b);
}


void opencv_rgb_split(Mat src)
{
	int size = src.rows * src.cols * 3;
	Mat b(src.rows, src.cols, CV_8UC1);
	Mat g(src.rows, src.cols, CV_8UC1);
	Mat r(src.rows, src.cols, CV_8UC1);
	
	auto start = chrono::system_clock::now();
	Mat out[] = {b, g, r};
	split(src, out);
	auto end   = chrono::system_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "opencv cost time is:" << double(duration.count()) << " us" << endl;
	
	// Mat b_color(src.rows, src.cols, CV_8UC3);
	// Mat g_color(src.rows, src.cols, CV_8UC3);
	// Mat r_color(src.rows, src.cols, CV_8UC3);
	// for(int i=0; i<size; i+=3)
	// {
	// 	b_color.data[i] = b.data[i/3];
	// 	b_color.data[i+1] = 0;
	// 	b_color.data[i+2] = 0;
		
	// 	g_color.data[i] = 0;
	// 	g_color.data[i+1] = g.data[i/3];
	// 	g_color.data[i+2] = 0;
		
	// 	r_color.data[i] = 0;
	// 	r_color.data[i+1] = 0;
	// 	r_color.data[i+2] = r.data[i/3];
	// }
	// imwrite("b.jpg", b_color);
	// imwrite("g.jpg", g_color);
	// imwrite("r.jpg", r_color);

}


void own_split_kernel(uint8_t* r, uint8_t* g, uint8_t* b, uint8_t* rgb, int lencolor){
	for(int i=0; i<lencolor; ++i){
		b[i] = rgb[3*i];
		g[i] = rgb[3*i + 1];
		r[i] = rgb[3*i + 2];
	}
}

void own_rgb_split(Mat src){
	int size = src.rows * src.cols;
	uint8_t *rgb = src.data;
	uint8_t* r = (uint8_t*)malloc(sizeof(uint8_t) * size);
	uint8_t* g = (uint8_t*)malloc(sizeof(uint8_t) * size);
	uint8_t* b = (uint8_t*)malloc(sizeof(uint8_t) * size);

	auto start = chrono::system_clock::now();
	own_split_kernel(r, g, b, rgb, size);
	auto end   = chrono::system_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "own cost time is:" << double(duration.count()) << " us" << endl;

	// Mat b_color(src.rows, src.cols, CV_8UC3);
	// Mat g_color(src.rows, src.cols, CV_8UC3);
	// Mat r_color(src.rows, src.cols, CV_8UC3);
	// size *= 3;
	// for(int i=0; i<size; i+=3)
	// {
	// 	b_color.data[i] = b[i/3];
	// 	b_color.data[i+1] = 0;
	// 	b_color.data[i+2] = 0;
		
	// 	g_color.data[i] = 0;
	// 	g_color.data[i+1] = g[i/3];
	// 	g_color.data[i+2] = 0;
		
	// 	r_color.data[i] = 0;
	// 	r_color.data[i+1] = 0;
	// 	r_color.data[i+2] = r[i/3];
	// }
	// imwrite("b_own.jpg", b_color);
	// imwrite("g_own.jpg", g_color);
	// imwrite("r_own.jpg", r_color);
	free(r);
	free(g);
	free(b);
}


int main(int argc, char** argv){
	Mat img = cv::imread("../person.png");

	neon_rgb_split(img);

	assembly_rgb_split(img);

	opencv_rgb_split(img);

	own_rgb_split(img);

    return 0;
}
