# week_18
大津阈值分割算法
(1)实现思路：
1)计算0~255各灰阶对应的像素个数，保存至一个数组中，该数组下标是灰度值，保存内容是当前灰度值对应像素数
2)计算背景图像的平均灰度、背景图像像素数所占比例
3)计算前景图像的平均灰度、前景图像像素数所占比例
4)遍历0~255各灰阶，计算并寻找类间方差极大值
(2)关键代码：
int Otsu(Mat matSrc)
{
	if (CV_8UC1 != matSrc.type())
	{
		cout << "Please input Gray-image!" << endl;
		return 0;
	}
	int nCols = matSrc.cols;
	int nRows = matSrc.rows;
	int nPixelNum = nCols * nRows;    //图像像素总数  

	int pixelNum[256];
	double probability[256];
	for (int i = 0; i < 256; i++)     //图像像素初始化
	{
		pixelNum[i] = 0;
		probability[i] = 0.0;
	}
	// 统计像素数和频率
	for (int j = 0; j < nRows; j++)
	{
		for (int i = 0; i < nCols; i++)
		{
			pixelNum[matSrc.at<uchar>(j, i)]++;
		}
	}
	for (int i = 0; i < 256; i++)
	{
		probability[i] = (double)0.1*pixelNum[i] / nPixelNum;
	}

	int T = 0;          // 最佳阈值
	double dMaxDelta = 0;      // 最大类间方差
	double f = 0;        // 前景平均灰度
	double b = 0;        // 背景平均灰度
	double dDelta = 0;         // 类间方差
	double f_temp = 0;   // 前景均值中间值
	double b_temp = 0;   // 背景均值中间值
	double fProbability = 0;  // 前景频率值
	double bProbability = 0;   // 背景频率值

	for (int j = 0; j < 256; j++)
	{
		for (int i = 0; i < 256; i++)
		{
			if (i < j)// 前景部分
			{
				fProbability += probability[i];
				f_temp += i * probability[i];     //计算前景部分灰度中间值
			}
			else      // 背景部分
			{
				bProbability += probability[i];
				b_temp += i * probability[i];     //计算背景部分灰度中间值
			}
		}

		f = f_temp / fProbability;    //前景的灰度
		b = b_temp / bProbability;    //背景的灰度

		dDelta = (double)(fProbability * bProbability * pow((f - b), 2));   //当前类方差的计算

		if (dDelta > dMaxDelta)
		{
			dMaxDelta = dDelta;
			T = j;
		}

		fProbability = 0;
		bProbability = 0;
		f_temp = 0;
		b_temp = 0;
		f = 0;
		b = 0;
		dDelta = 0;

	}
	return T;
}
2.拉普拉斯变换 
（1）实验原理

（2）函数原型及参数说明
函数原型：CV_EXPORTS_W void Laplacian( InputArray src, OutputArray dst, int ddepth,
                                       int ksize = 1, double scale = 1, double delta = 0,
                                       int borderType = BORDER_DEFAULT );
参数说明：src：源图像。
dst：目标图像的大小和频道数与src相同。
ddepth：目标图像的期望深度。
ksize： 用于计算二阶导数滤波器的孔径大小。大小必须是正数和奇数。
scale ：可选的计算拉普拉斯值的比例因子。默认情况下，不应用缩放。
delta：可选值，最终结果的偏移值。
（3）实验代码：
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui_c.h>  
using namespace cv;

Mat src, dst, abs_dst_laplace;

int main()
{

	src = imread("D:\\opencv\\test4_1\\test4_1\\lena.jpg");
	namedWindow("原图", CV_WINDOW_NORMAL);
	imshow("原图", src);

	//拉普拉斯变换
	Laplacian(src, dst, CV_16S, 3);                     //拉普拉斯变换
	//                          深度      核大小
	convertScaleAbs(dst, abs_dst_laplace);              //将类型转化为CV_8UC1
	namedWindow("laplacian", CV_WINDOW_NORMAL);
	imshow("laplacian", abs_dst_laplace);

	waitKey(0);
}
3.Canny检测边缘算法
（1）Canny算法实现的步骤
1）对图像进行高斯模糊处理
2）求出图像的梯度图像及梯度方向矩阵
3）对梯度图像进行非极大化抑制，使宽边变为细边
4）对生成的非极大化抑制结果图进行滞后阈值法，用强边延伸弱边，解决伪边问题
（2）函数原型及参数说明
函数原型：void cv::Canny(InputArray  image,
                   OutputArray  edges,
	                 double  threshold1,
	                 double  threshold2,
                   int  apertureSize = 3,
	                 bool  L2gradient = false 
	                     )
参数说明：image：输入图像，必须是CV_8U的单通道或者三通道图像。
edges：输出图像，与输入图像具有相同尺寸的单通道图像，且数据类型为CV_8U。
threshold1：第一个滞后阈值。
threshold2：第二个滞后阈值。
apertureSize：Sobel算子的直径。
L2gradient：计算图像梯度幅值方法的标志。
（3）实验代码
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h> 
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
int main() {
	Mat srcImage, grayImage;
	srcImage = imread("D:\\opencv\\test4_1\\test4_1\\lena.jpg");
	namedWindow("srcImage", CV_WINDOW_NORMAL);
	imshow("srcImage", srcImage);
	Mat srcImage1 = srcImage.clone();
	cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
	namedWindow("grayImage", CV_WINDOW_NORMAL);
	imshow("grayImage", grayImage);
	Mat dstImage, edge;

	blur(grayImage, grayImage, Size(3, 3));
	Canny(grayImage, edge, 150, 100, 3);

	dstImage.create(srcImage1.size(), srcImage1.type());
	dstImage = Scalar::all(0);
	srcImage1.copyTo(dstImage, edge);
	namedWindow("canny", CV_WINDOW_NORMAL);
	imshow("canny", dstImage);
	//imwrite("canny.jpg", dstImage);

	waitKey(0);
}
4.区域生长法算法
（1）实验原理
区域生长算法的基本思想是将有相似性质的像素点合并到一起。对每一个区域要先指定一个种子点作为生长的起点，然后将种子点周围领域的像素点和种子点进行对比，将具有相似性质的点合并起来继续向外生长，直到没有满足条件的像素被包括进来为止。这样一个区域的生长就完成了。
（2）算法流程图
（3）关键代码
Mat RegionGrow(Mat src, Point2i pt, int th)
{
	Point2i ptGrowing;                        //待生长点位置
	int nGrowLable = 0;                                //标记是否生长过
	int nSrcValue = 0;                                //生长起点灰度值
	int nCurValue = 0;                                //当前生长点灰度值
	Mat matDst = Mat::zeros(src.size(), CV_8UC1);    //创建一个空白区域，填充为黑色
	//生长方向顺序数据
	int DIR[8][2] = { { -1, -1 }, { 0, -1 }, { 1, -1 }, { 1, 0 }, { 1, 1 }, { 0, 1 }, { -1, 1 }, { -1, 0 } };
	vector<Point2i> vcGrowPt;                        //生长点栈
	vcGrowPt.push_back(pt);                            //将生长点压入栈中
	matDst.at<uchar>(pt.y, pt.x) = 255;                //标记生长点
	nSrcValue = src.at<uchar>(pt.y, pt.x);            //记录生长点的灰度值

	while (!vcGrowPt.empty())                        //生长栈不为空则生长
	{
		pt = vcGrowPt.back();                        //取出一个生长点
		vcGrowPt.pop_back();

		//分别对八个方向上的点进行生长
		for (int i = 0; i < 8; ++i)
		{
			ptGrowing.x = pt.x + DIR[i][0];
			ptGrowing.y = pt.y + DIR[i][1];
			//检查是否是边缘点
			if (ptGrowing.x < 0 || ptGrowing.y < 0 || ptGrowing.x >(src.cols - 1) || (ptGrowing.y > src.rows - 1))
				continue;

			nGrowLable = matDst.at<uchar>(ptGrowing.y, ptGrowing.x);        //当前待生长点的灰度值

			if (nGrowLable == 0)                    //如果标记点还没有被生长
			{
				nCurValue = src.at<uchar>(ptGrowing.y, ptGrowing.x);
				if (abs(nSrcValue - nCurValue) < th)                    //在阈值范围内则生长
				{
					matDst.at<uchar>(ptGrowing.y, ptGrowing.x) = 255;        //标记为白色
					vcGrowPt.push_back(ptGrowing);                    //将下一个生长点压入栈中
				}
			}
		}
	}
	return matDst.clone();
}
5.霍夫变换直线检测算法
（1）实验原理

（2）函数原型及参数说明
函数原型：void cv::HoughLines(InputArray  image,
                                          OutputArray  lines,
                                          double  rho,
                                          double  theta,
                                          int  threshold,
                                          double  srn = 0,
                                          double  stn = 0,
                                          double  min_theta = 0,
                                          double  max_theta = CV_PI 
                                          )
参数说明：image：待检测直线的原图像，必须是CV_8U的单通道二值图像。
lines：霍夫变换检测到的直线输出量，每一条直线都由两个参数表示，分别表示直线距离坐标原点的距离 和坐标原点到直线的垂线与x轴的夹角 。
rho：以像素为单位的距离分辨率，即距离 离散化时的单位长度。
theta：以弧度为单位的角度分辨率，即夹角 离散化时的单位角度。
threshold：累加器的阈值，即参数空间中离散化后每个方格被通过的累计次数大于该阈值时将被识别为直线，否则不被识别为直线。
srn：对于多尺度霍夫变换算法中，该参数表示距离分辨率的除数，粗略的累加器距离分辨率是第三个参数rho，精确的累加器分辨率是rho/srn。这个参数必须是非负数，默认参数为0。
stn：对于多尺度霍夫变换算法中，该参数表示角度分辨率的除数，粗略的累加器距离分辨率是第四个参数rho，精确的累加器分辨率是rho/stn。这个参数必须是非负数，默认参数为0。当这个参数与第六个参数srn同时为0时，此函数表示的是标准霍夫变换。
min_theta：检测直线的最小角度，默认参数为0。
max_theta：检测直线的最大角度，默认参数为CV_PI，是OpenCV4中的默认数值具体为3.1415926535897932384626433832795。
（3）关键代码：
void drawLIne(Mat &img,//要标记的图像
	vector<Vec2f>lines,//检测的直线数据
	double rows,//原图像的行数
	double cols,//原图像的列数
	Scalar scalar,//绘制直线的颜色
	int n//绘制直线的线宽
)
{
	Point pt1, pt2;
	for (size_t i = 0; i < lines.size(); ++i) {
		float rho = lines[i][0];//直线距离坐标原点的距离
		float theta = lines[i][1];//直线过坐标原点垂线与x轴夹角
		double a = cos(theta);//夹角的余弦值
		double b = sin(theta);//夹角的正弦值
		double x0 = a * rho, y0 = b * rho;//直线与坐标原点垂线的交点
		double length = max(rows, cols);//图像高宽的最大值

		//计算直线上的一点
		pt1.x = cvRound(x0 + length * (-b));
		pt1.y = cvRound(y0 + length * (a));
		//计算直线上另一点
		pt2.x = cvRound(x0 - length * (-b));
		pt2.y = cvRound(y0 - length * (a));
		//两点绘制一条直线
		line(img, pt1, pt2, scalar, n);
	}
