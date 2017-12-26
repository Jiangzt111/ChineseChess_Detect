// GetCircle.cpp : �������̨Ӧ�ó������ڵ㡣  
//Բ�μ�����demo  
//�������Ű���������״��ͼƬ���������е�Բ��  
#include "cv.h"  
#include "highgui.h"  
#include <math.h>  
#include <string.h>  
#include <iostream>  
#include "windows.h"  //��������ʱ���Ӧ��ͷ�ļ�  



#include "opencv2/nonfree/nonfree.hpp"    
#include "opencv2/nonfree/features2d.hpp"    
#include "opencv2/legacy/legacy.hpp"  
#include <opencv2/opencv.hpp>  
using namespace std;
using namespace cv;


//SIFT�������  
SiftFeatureDetector detector;        //�����ص������  
vector<KeyPoint> keypoint01, keypoint02;//���������������������  
double x[9], y[10]; //��¼��������
int pic_num = 0;      //��¼��ǰ�����ڼ���ͼ
char hanzi1[33][3] = {"0", "ʿ","ʿ", "��", "��", "��", "��", "��", "��","��",
"��",  "��",
"��", "��",  "��", "��" ,"��","��", "��", "��", "��","ʿ",
"��", "��",
"��","��", "��", "ʿ", "��", "��", "˧", "��","��" };
IplImage* model[33] = { NULL };//�洢�ü���Сͼģ��
IplImage* model_rota[33] = { NULL };//�洢�ü���Сͼģ��_��ת
CvMemStorage* storage = NULL;
char* names[] = { "0.jpg", "1.jpg","2.jpg","3.jpg","4.jpg","5.jpg","6.jpg","7.jpg","8.jpg",0 };
struct chess
{
	int center[33][2] = { 0 };//Բ�����꣬X*Y
	int radius[33] = { 0 };//Բ�İ뾶
}Chess_inf;

struct chart
{
	char flag;			 //0��1  ��ʾ�л���
	char type;			  //0-31 ��ʾ0-15 ��ɫ����������   16-31 ��ɫ��������
	char color = 1;			  //��ɫ���ɫ
	int xy_axis[1][2];			//���̽�����������

}Chess_Chart[10][9];

void Cut_model(IplImage* yuantu, int center[32][2])
{
	char szName[56] = { 0 };
	IplImage* jiequ[32] = { NULL };
	IplImage* yuantu1 = NULL;
	yuantu1 = cvCloneImage(yuantu);
	for (int k = 0; k < 32; k++)
	{
		if (center[k][1] == 0)
			break;
		cvSetImageROI(yuantu1, cvRect(center[k][0] - 10, center[k][1] - 10, 20, 20));
		jiequ[k] = cvCloneImage(yuantu1);
		cvResetImageROI(yuantu1);
		//sprintf(szName, "cut%d-%d.jpg", pic_num, k);
	//  cvSaveImage(szName, jiequ[k]);
		model[k] = cvCloneImage(jiequ[k]);
	}
	cvReleaseImage(&yuantu1);
}

void HoughCircle(IplImage *imageP, struct chess *Chess_inf)
{
	IplImage* pImg8u = NULL;
	CvSeq * circles = NULL;
	pImg8u = cvCreateImage(cvGetSize(imageP), 8, 1);
	CvMemStorage* storage = cvCreateMemStorage(0);
	cvCvtColor(imageP, pImg8u, CV_BGR2GRAY);
	cvSmooth(pImg8u, pImg8u, CV_GAUSSIAN, 3, 3);//��cvSmoothһ�£��ٵ���cvHoughCircles  
	circles = cvHoughCircles(pImg8u, storage, CV_HOUGH_GRADIENT,
		1,   //��С�ֱ��ʣ�Ӧ��>=1  
		20,   //�ò��������㷨���������ֵ�������ͬԲ֮�����С����  20
		80,   //����Canny�ı�Ե��ֵ���ޣ����ޱ���Ϊ���޵�һ��  70
		10,    //�ۼ��� �ķ�ֵ  20
		9,  //��СԲ�뾶   
		11  //���Բ�뾶  
	);
	int k;
	for (k = 0; k<circles->total; k++)
	{
		float *p = (float*)cvGetSeqElem(circles, k);
		if ((cvRound(p[2]))>9 && (cvRound(p[2]))<14)
		{
			cvCircle(imageP, cvPoint(cvRound(p[0]), cvRound(p[1])), cvRound(p[2]), CV_RGB(0, 255, 0), 3, CV_AA, 0);
			Chess_inf->radius[k] = cvRound(p[2]);	
			Chess_inf->center[k][0] = cvRound(p[0]);
			Chess_inf->center[k][1] = cvRound(p[1]);
		}
	}
	cvShowImage("2", imageP);
	cvReleaseImage(&pImg8u);
	cvClearMemStorage(storage);
}
void Linedet(IplImage *imageP)
{
	vector<int>::iterator it1, it2;
	vector<int> Cx, Cy;
	IplImage* pImg8u = NULL;
	IplImage * DstImg111 = NULL;
	int paixu[5][50] = { 0 };
	int paixu1[2][10] = { 0 };
	int paixu_x[50] = { 0 }; int paixu_y[50] = { 0 };
	DstImg111 = cvCreateImage(cvGetSize(imageP), IPL_DEPTH_8U, 1);
	CvSeq *lines = 0;
	cvCanny(imageP, DstImg111, 140, 250, 3);
	CvMemStorage *storage = cvCreateMemStorage();
	lines = cvHoughLines2(DstImg111, storage, CV_HOUGH_PROBABILISTIC, 1, CV_PI / 180, 80, 100, 30);
	for (int i = 0; i < lines->total; i++)
	{
		CvPoint *line = (CvPoint *)cvGetSeqElem(lines, i);
		cvLine(imageP, line[0], line[1], CV_RGB(255, 0, 0), 3, 8);
		paixu[0][i] = line[0].x;
		paixu[1][i] = line[0].y;
		paixu[2][i] = line[1].x;
		paixu[3][i] = line[1].y; 
	}
	cvShowImage("3", imageP);
	int i = 0, j = 0, n = 0, m = 0, k = 0, a = 0;
	for (i = 1; i < 50; i++)
	{
		if ((paixu[0][i] == 0)&&(paixu[1][i] == 0))
			break;
		if (paixu[0][i] == paixu[2][i]) //����
		{
			paixu_x[j++] = paixu[0][i]; //
		}
	}
	for (i = 1; i < 50; i++)
	{
		if ((paixu[0][i] == 0) && (paixu[1][i] == 0))
			break;
		if (paixu[1][i] ==paixu[3][i]) //����
		{
			paixu_y[n++] = paixu[1][i];
		}
	}
	k = j;
	for (i = 0; i <k - 1; i++)
	{
		for (m = 0; m<k - 1 - i; m++)
		{
			if (paixu_x[m] > paixu_x[m + 1])
			{
				a = paixu_x[m]; paixu_x[m] = paixu_x[m + 1]; paixu_x[m + 1] = a;
			}
		}
	}
	k = n;
	for (i = 0; i <k - 1; i++)
	{
		for (m = 0; m<k - 1 - i; m++)
		{
			if (paixu_y[m] > paixu_y[m + 1])
			{
				a = paixu_y[m]; paixu_y[m] = paixu_y[m + 1]; paixu_y[m + 1] = a;
			}
		}
	}
	int b = 0;
	for (i = 1; i <j + 1; i++)
	{
		if ((paixu_x[i] - paixu_x[i - 1]>10) || (paixu_x[i] - paixu_x[i - 1]<-10)) {
			for (int q = 0; q < 10; q++) {
				Chess_Chart[q][b].xy_axis[0][0] = paixu_x[i - 1];
			}
			b += 1;
		}

	}
	b = 0;
	for (i = 1; i <n + 1; i++)
	{
		if ((paixu_y[i] - paixu_y[i - 1]>10) || (paixu_y[i] - paixu_y[i - 1]<-10)) {
			for (int q = 0; q < 9; q++) {
				Chess_Chart[b][q].xy_axis[0][1] = paixu_y[i - 1];
			}
			b += 1;
		}

	}
	if (b == 9)
	{
		for (int q = 0; q < 9; q++) {
			Chess_Chart[b][q].xy_axis[0][1] = Chess_Chart[b-1][q].xy_axis[0][1]+24;
		}
	}

}
double chess_match(cv::Mat image, cv::Mat tepl, cv::Point &point, int method)
{
	int result_cols = image.cols - tepl.cols + 1;
	int result_rows = image.rows - tepl.rows + 1;
	cv::Mat result = cv::Mat(result_cols, result_rows, CV_32FC1);
	cv::matchTemplate(image, tepl, result, method);

	double minVal, maxVal;
	cv::Point minLoc, maxLoc;
	cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

	switch (method)
	{
	case CV_TM_SQDIFF:
	case CV_TM_SQDIFF_NORMED:
		point = minLoc;
		return minVal;
		break;
	default:
		point = maxLoc;
		return maxVal;
		break;
	}
}

void imrotate(Mat& img, Mat& newIm, double angle) {
	int len = max(img.cols, img.rows);
	Point2f pt(len / 2., len / 2.);
	Mat r = getRotationMatrix2D(pt, angle, 1.0);
	warpAffine(img, newIm, r, Size(len, len));

}

void Chess_match1(IplImage *tu, int a, int b)
{
	char szName[56] = { 0 };
	cv::Point matchLoc;
	double matchVal;
	Mat tu1 = Mat(tu);
	double q = 0;
	int x = 0, y = 0;

	for (int n = 0; n < 180; n++) 
	{

		for (int i = 0; model[i] != NULL; i++)
		{
			Mat model1 = Mat(model[i]);
			Mat model2;
			imrotate(model1, model2, n);

			//matchVal = chess_match(tu1, model1, matchLoc, CV_TM_CCOEFF_NORMED);
			matchVal = chess_match(tu1, model2, matchLoc, CV_TM_CCOEFF_NORMED);
			model1.release();
			model2.release();
			if (matchVal >= q)
			{
				q = matchVal;
				x = i;
			}
		}n+2;
		
	}
	Chess_Chart[a][b].type = x+1; 
	
}
void Color_match(IplImage* tu1, int a, int b)//��ɫƥ����д��Chess_chart[][].color
{
	CvScalar pixel;
	IplImage* tu2 = NULL;
	int numr = 0, numb = 0;
	for (int i = 0; i < tu1->height; i++)
	{
		for (int j = 0; j < tu1->width; j++)
		{
			pixel = cvGet2D(tu1, i, j);
			if ((pixel.val[2] > 200)&& (pixel.val[1] < 100)&& (pixel.val[0] < 100))
			{
					++numr;//ͳ��hong�����
			}
			else
			{
				++numb;//ͳ��hei�����
			}
		}
	}
	if (numr > 30)
	{
		Chess_Chart[a][b].color =1;
	}
}
void Cut_match(IplImage* yuantu1, IplImage jiequ[32])//������������꽻��Ϊ���Ĳü�һƬ����ƥ����������
{
	IplImage* tu1 = cvCreateImage(cvSize(30, 30),8,3);
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 9; j++)
		{
			if (Chess_Chart[i][j].flag == 1)
			{
				cvSetImageROI(yuantu1, cvRect(Chess_Chart[i][j].xy_axis[0][0] - 15, Chess_Chart[i][j].xy_axis[0][1] - 15, 30, 34));
				cvResize(yuantu1, tu1);
				cvResetImageROI(yuantu1);
				Chess_match1(tu1, i, j);
				Color_match(tu1, i, j);
			}
		}
	}
	cvReleaseImage(&tu1);
}
void Flag_Set()
{
	int x = 0, y = 0;
	int num = 100000;
	int sum = 0;
	for (int i = 0; Chess_inf.center[i][0] != 0; i++)
	{

		for (int j = 0; j < 9; j++)
		{
			for (int k = 0; k< 10; k++)
			{
				sum = abs(Chess_inf.center[i][0] - Chess_Chart[k][j].xy_axis[0][0]) + abs(Chess_inf.center[i][1] - Chess_Chart[k][j].xy_axis[0][1]);
				if (num > sum)
				{
					num = sum;
					x = j;
					y = k;
				}
			}
		}
		Chess_Chart[y][x].flag = 1;
		num = 100000;
		sum = 0;
		x = 0;
		y = 0;
	}

}
void Vision_chess()
{
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 9; j++) {
			if ( Chess_Chart[i][j].flag == 1) { 
				//printf(" %d", Chess_Chart[i][j].type);
				printf(" %d", Chess_Chart[i][j].color);
				printf(" %s ", hanzi1[Chess_Chart[i][j].type]);
			}
			else { printf("  +   "); }
			if (j == 8)
			{
				printf("\n");
				printf("\n");
			}
		}
	}
}
int main(int argc, char** argv)
{
	int c;
	IplImage* img0 = NULL;
	IplImage* img1 = NULL;
	IplImage* img2 = NULL;
	IplImage* pImg8u = NULL;


	for (pic_num = 0; names[pic_num] != 0; pic_num++)
	{
		memset(&Chess_inf, 0, sizeof(struct chess));
		memset(Chess_Chart, 0, sizeof(struct chart) * 90);
		img0 = cvLoadImage(names[pic_num], 1);
		if (!img0)
		{
			cout << "��������" << names[pic_num] << "������һ��ͼƬ" << endl;
			continue;
		}
		img1 = cvCloneImage(img0);
		img2 = cvCloneImage(img0);
		DWORD begin = 0, end = 0;
		begin = GetTickCount();

		HoughCircle(img1, &Chess_inf);		//     ����Բ��⣬�ڽṹ��Chess_inf�б���Բ�ġ��뾶����Ϣ

		if(pic_num==0) 
             Cut_model(img0, Chess_inf.center);//��һ��ͼģ���⣬���沢���

		Linedet(img1);						//ֱ�߼�⣬�������̣�д��Chess_Chart[10][9].xy_axis[][]  ��������;

		Flag_Set();							//����Բ�����꣬ƥ�����������̵�λ�ã���Chess_Chart[10][9].flag д��1��0����ʾ��Ӧλ����������

		Cut_match(img0, model[32]);			//  ģ��ƥ�䣬��Chess_Chart[10][9].type д����Ϣ����ʾ��һ������

		Vision_chess();						 //���̿��ӻ�

		end = GetTickCount();
		cout << "The run time is:" << (end - begin) << "ms!" << endl;//�������ʱ��    
		cvShowImage("1", img0);

		c = cvWaitKey(0);
		if ((char)c == 27)
			break;
	}

	cvReleaseImage(&img0);
	cvReleaseImage(&img1);
	cvReleaseImage(&img2);
	cvReleaseImage(&pImg8u);
	return 0;
}
//	printf("depth: %d \n", cir.center[1][0]);		printf("depth: %d \n", img->depth);