
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/cvstd.hpp"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdarg.h>
#include <math.h>

using namespace std;
using namespace cv;

//Mat img, resizeImg(Size(9,9), CV_32F, Scalar::all(0));
char *itoa(int a);
char itoa_basic(int d);
void resize2(Size size, int a);
void threshold2(int thresh, int maxValue, int method);

int main( int argc, const char** argv )
{    
  /*
    int models[4][81] = {{1,1,1,1,0,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,1,1,1}, 
{1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,1,1,0,0,0,1,1,0,0,1,1,0,0,0,1,1,0,0,1,1,0,0,0,1,1,0,0,1,1,0,0,0,1,1,0,0,1,1,0,0,0,1,1,0,0,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1},
{1,1,1,1,1,1,1,1,1,1,0,0,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,0,0,0,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,0,1,0,1,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1},
{1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}};
 */
    int models[4][81] = {{1,1,1,1,0,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,1,1,1}, 
{1,1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,1,1,0,0,1,0,0,0,0,1,1,0,0,1,1,0,0,0,1,1,0,0,1,1,1,0,0,1,1,0,0,1,1,1,1,0,1,1,0,0,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1},
{1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,0,1,1,1,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,1,1,1},
{1,1,1,1,0,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0}};

    int threshs[] = {30, 30, 30, 30};
    int errors[] = {0, 0, 0, 0};
    double weights[] = {0.0, 0.0, 0.0, 0.0};
    double ratios[] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    int origJudge[] = {1,1,1,1,1,-1,-1,-1,-1,-1};
    int testResults[10][4] = {{1,1,1,1},{1,1,1,1},{1,1,1,1},{1,1,1,1},{1,1,1,1},{1,1,1,1},{1,1,1,1},{1,1,1,1},{1,1,1,1},{1,1,1,1}};
    int imgResults[10][81];
    int imgTestResults[10][81];

    Mat img[10];
    int round = 1;
    char content[81];
    int flag = 0;
    int dist = 0, minError, minIndex;
    double alpha;
    while(round<=4){
       for (int i=0; i<4; i++)
          errors[i] = 0;
       for (int i=0; i<10; i++){
          for (int j=0; j<4; j++)
             testResults[i][j] = 1;
       }
       
       //char name[40];      
       //char *fileName; char *fileName2;
       for (int i=0; i<10; i++){
          //name = new char[40];
          char name[40] = "train/redLightTrain\0";
          cout << "name = " << name << endl;
          char *fileName = new char[strlen(name) + 10];
          char *fileName2 = new char[strlen(name) + 40]; 
          if (i!=9){
             cout << "before strcat" << endl;
             char *a = new char[2];
             a = itoa(i+1);
             fileName = strcat(name, a);
             cout << "fileName = " << fileName << endl;  
          }else{
             char *a = itoa(11);
             fileName = strcat(name, a);
             cout << "fileName = " << fileName << endl; 
          }
          fileName2 = strcat(fileName, ".jpg");
          cout << "fileName2 = " << fileName2 << endl;
          if (round == 1){
             img[i] =  imread(fileName2);
             cout << "img[i].channels() = " << img[i].channels() << " and img[i].rows = " << img[i].rows << " and img[i].cols = " << img[i].cols << endl;
             float resizeImg[9][9];
             //resizeImg[i].create(9, 9, CV_32F);
             //cout << "resizeImg[i].rows =" << resizeImg[i].rows << " and resizeImg[i].cols =" << resizeImg[i].cols << endl;
             for(int i2=0; i2<9; i2++){
                for (int j2=0; j2<9; j2++){
                   //resizeImg[i].at<float>(i2,j2) = 0;
                   resizeImg[i2][j2] = 0;
                }
             } 
             
             cout << "so far is fine before img2" << endl;
             //first, we want to average img[i] by its channels vertically
             //img2[i].create(img[i].rows/3, img[i].cols, CV_8UC1);
             float img2[img[i].rows/3][img[i].cols];             
             for(int i2=0; i2<img[i].rows/3; i2++){
                for (int j2=0; j2<img[i].cols; j2++){ 
                   //img2[i].at<uchar>(i2,j2) = (img[i].at<uchar>(3*i2,j2) + img[i].at<uchar>(3*i2+1,j2) + img[i].at<uchar>(3*i2+2,j2))/3;
                   img2[i2][j2] = (img[i].at<uchar>(3*i2,j2) + img[i].at<uchar>(3*i2+1,j2) + img[i].at<uchar>(3*i2+2,j2))/((float)3);
                }
             }
             cout << "so far is fine 2 before resizeImg calcualtion" << endl; 
             //calculate the value of resizeImg
             int rowTimes = (img[i].rows/3) / 9;
             int colTimes = img[i].cols / 9;  
             for(int i2=0; i2<img[i].rows/3; i2++){
                for (int j2=0; j2<img[i].cols; j2++){
                   int newRow = i2 / rowTimes;
                   newRow = newRow>=9 ? (9-1) : newRow;
                   int newCol = j2 / colTimes;
                   newCol = newCol>=9 ? (9-1) : newCol;
                   resizeImg[newRow][newCol] += img2[i2][j2] / ((float)(rowTimes * colTimes));
                }
             }
             //set each value of resizeImge to 0 or 1
             double total = 0;
             for (int i2=0; i2<9; i2++){
                for (int j2=0; j2<9; j2++)
                   total += resizeImg[i2][j2];
             }
             double average = ((double)total) / (9*9);
             for (int i2=0; i2<9; i2++){
                for (int j2=0; j2<9; j2++){
                   if (resizeImg[i2][j2]>average)
                      resizeImg[i2][j2] = 1;
                   else
                      resizeImg[i2][j2] = 0;
                }
             }
             //resizeImg.create(9, 9, CV_32F);
             //resize2(Size(9,9), CV_32F);
             //threshold2(128, 1, THRESH_BINARY); 
             //resize(img, resizeImg, Size(9,9), CV_32FC1);
             //threshold(resizeImg, resizeImg, 128, 1, THRESH_BINARY); 
            
             flag = 0;
             //imgResults[i] = new char[81];
             for (int i2=0; i2<9; i2++){
                for (int j2=0; j2<9; j2++){
                   imgResults[i][flag] = (int)resizeImg[i2][j2];
                   //cout << "content[" << flag << "] = " << cout << content[flag] << "\t";
                   //printf("content[%d]=%d", flag, content[flag]);
                   flag++;
                }
             }
          }//end of (round==1)

          //we want to calculate the distance between the ith train/redLightTrain.jpg with the count_th model/redLightModel.jpg
          char *b1, *b2;
          for (int count=0; count<4; count++){
             dist = 0;
             for (int j=0; j<81; j++){
                //b1 = new char[2]; b2 = new char[2];
                //b1[0] = imgResults[i][j]; b1[1] = '\0'; b2[0] = models[count][j]; b2[1] = '\0';
                //if (atoi(b1) != atoi(b2))
                if (imgResults[i][j] != models[count][j])
                   dist ++;
             }
             if (dist>30){
                testResults[i][count] = -1;
                int judge = origJudge[i] * (-1);
                if (judge==1){
                   cout << "train/redLightTrain" << i << ".jpg is NOT redlight, and judged by weak classifier " << count << " as NOT with dist =" << dist << endl;
                }else{
                   cout << "train/redLightTrain" << i << ".jpg IS redlight, but judged by weak classifier " << count << " as NOT with dist=" << dist << endl;
                   errors[count] = errors[count] + 1;
                }
             }else{
                testResults[i][count] = 1;
                int judge = origJudge[i] * (1);
                if (judge==1){
                   cout << "train/redLightTrain" << i << ".jpg IS redlight, and judged by weak classifier " << count << " as IS with dist=" << dist << endl;
                }else{
                   cout << "train/redLightTrain" << i << ".jpg is NOT redlight, but judged by weak classifier " << count << " as IS with dist=" << dist << endl;
                   errors[count] = errors[count] + 1;                   
                }
             }//end of (dist>30)
          }//end of for(count=0:3)          
       }//end of for(i=0:9)

       minError = 20; minIndex = -1;
       //we want to find in this round the most efficient weak classifier
       for (int count=0; count<4; count++){
          if (errors[count] < minError){
             minError = errors[count];
             minIndex = count;
          }
       }
       double errorRatio = double(minError)/10;
       if (errorRatio != 0 ){
          alpha = 0.5 * log((1-errorRatio)/errorRatio);
          weights[minIndex] = alpha;
       }else{
          weights[minIndex] = 1;
          break;
       }
     
       //now we want to adjust the ratio for next rounds
       double totalRatio = 0.0;
       for (int i=0; i<9; i++){
          totalRatio += ratios[i] * exp(-alpha * origJudge[i] * testResults[i][minIndex]);
       }
       for (int i=0; i<9; i++){
          ratios[i] = ratios[i] * exp(-alpha * origJudge[i] * testResults[i][minIndex]) / totalRatio;
       } 
       cout << "in round=" << round << ", weak classifier " << minIndex << " performs best with alpha = " << alpha << endl;
       round ++;
    }//end of round<=4

    //now we are going to test against predict
    for (int i=0; i<10; i++){
       //name = new char[40];
       char name[40] = "predict/redLightTest\0";
       cout << "name = " << name << endl;
       char *fileName = new char[strlen(name) + 10];
       char *fileName2 = new char[strlen(name) + 40]; 
       char *a;
       if (i==9)
          a = itoa(11);
       else
          a = itoa(i+1);
       fileName = strcat(name, a);
       cout << "fileName = " << fileName << endl; 
       fileName2 = strcat(fileName, ".jpg");
       cout << "fileName2 = " << fileName2 << endl;

       Mat img =  imread(fileName2);
       cout << "img.channels() = " << img.channels() << " and img.rows = " << img.rows << " and img.cols = " << img.cols << endl;
       float resizeImg[9][9];
       for(int i2=0; i2<9; i2++){
          for (int j2=0; j2<9; j2++){
             //resizeImg[i].at<float>(i2,j2) = 0;
             resizeImg[i2][j2] = 0;
          }
       } 
       float img2[img.rows/3][img.cols];             
       for(int i2=0; i2<img.rows/3; i2++){
          for (int j2=0; j2<img.cols; j2++){ 
             img2[i2][j2] = (img.at<uchar>(3*i2,j2) + img.at<uchar>(3*i2+1,j2) + img.at<uchar>(3*i2+2,j2))/((float)3);
          }
       }
       int rowTimes = (img.rows/3) / 9;
       int colTimes = img.cols / 9;  
       for(int i2=0; i2<img.rows/3; i2++){
          for (int j2=0; j2<img.cols; j2++){
             int newRow = i2 / rowTimes;
             newRow = newRow>=9 ? (9-1) : newRow;
             int newCol = j2 / colTimes;
             newCol = newCol>=9 ? (9-1) : newCol;
             resizeImg[newRow][newCol] += img2[i2][j2] / ((float)(rowTimes * colTimes));
          }
       }
       //set each value of resizeImge to 0 or 1
       double total = 0;
       for (int i2=0; i2<9; i2++){
          for (int j2=0; j2<9; j2++)
             total += resizeImg[i2][j2];
       }
       double average = ((double)total) / (9*9);
       for (int i2=0; i2<9; i2++){
          for (int j2=0; j2<9; j2++){
             if (resizeImg[i2][j2]>average)
                resizeImg[i2][j2] = 1;
             else
                resizeImg[i2][j2] = 0;
          }
       }
       int flag = 0;             
       for (int i2=0; i2<9; i2++){
          for (int j2=0; j2<9; j2++){
             imgTestResults[i][flag] = (int)resizeImg[i2][j2];
             flag++;
          }
       }
       int dist = 0;
       int minIndex = 0;
       double passValue = 0;
       for (int i2=0; i2<4; i2++){
          dist = 0;
          for (int j2=0; j2<81; j2++){
             if (imgTestResults[i][j2] != models[i2][j2])
                dist ++;
          }
          cout << "for the " << i << " th test as true redlight, dist = " << dist << endl;
          if (dist<=41)
             passValue = passValue + weights[i2] * 1;
          else
             passValue = passValue - weights[i2] * 1;
       }
       if (passValue>0)
          cout << "for the " << i << " th test as true redlight, it is judged as (1 for true and -1 for false): " << 1 << endl;
       else
          cout << "for the " << i << " th test as true redlight, it is judged as (1 for true and -1 for false): " << -1 << endl;
    }
    return 0;
}


char *itoa(int a){
    char *ret;
    int n = 1;
    int na = a;
    while(na/10 >= 1){
       na /= 10;
       n++;
    } 
    ret = new char[n+1];
    int index;
    na = a;
    int d;
    for (int i=1; i<=n; i++){
       index = n-i;
       d = na % 10;
       ret[index] = itoa_basic(d);
       na /= 10;
    }
    ret[n] = '\0';
    return ret;    
}

char itoa_basic(int d){
   char c;
   switch (d){
       case 0:
          c = '0';
          break;
       case 1:
          c = '1';
          break;
       case 2:
          c = '2';
          break;
       case 3:
          c = '3';
          break;
       case 4:
          c = '4';
          break;
       case 5:
          c = '5';
          break;
       case 6:
          c = '6';
          break;
       case 7:
          c = '7';
          break;
       case 8:
          c = '8';
          break;
       case 9:
          c = '9';
          break;
       default:
          c = '0';
          break;
    }
    return c;
}

void resize2(Size size, int a){
   /*
   //int channel = img.channels(); 
   int rowTimes = img.rows / resizeImg.rows;
   int colTimes = img.cols / resizeImg.cols;  
   for(int i=0; i<img.rows; i++){
      for (int j=0; j<img.cols; j++){
         int newRow = i / rowTimes;
         newRow = newRow>=resizeImg.rows ? resizeImg.rows : newRow;
         int newCol = j / colTimes;
         newCol = newRow>=resizeImg.cols ? resizeImg.cols : newCol;
         resizeImg.at<uchar>(i/rowTimes, j/(colTimes)) += img.at<uchar>(i,j) / ((double)(rowTimes * colTimes));
      }
   }
   */
}

void threshold2(int thresh, int maxValue, int method){
   /*
   double total = 0;
   for (int i=0; i<resizeImg.rows; i++){
      for (int j=0; j<resizeImg.cols; j++)
         total += resizeImg.at<uchar>(i,j);
   }
   double average = ((double)total) / (resizeImg.rows * resizeImg.cols);
   for (int i=0; i<resizeImg.rows; i++){
      for (int j=0; j<resizeImg.cols; j++){
         if (resizeImg.at<uchar>(i,j)>average)
            resizeImg.at<uchar>(i,j) = 1;
         else
            resizeImg.at<uchar>(i,j) = 0;
      }
   }
   */
}
