//+------------------------------------------------------------------+
//|                                                   TestScript.mq5 |
//|                                     Copy_nasdaqright 2021, Omega Joctan |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
//hire me on your next big Machine Learning Project on this link > https://www.mql5.com/en/job/new?prefered=omegajoctan
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "Copy_nasdaqright 2021, Omega Joctan"
#property link      "https://www.mql5.com/en/users/omegajoctan"
#property version   "1.00"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
#include "LinearRegressionLib.mqh";  

//#include  "C:\Users\Omega Joctan\AppData\Roaming\MetaQuotes\Terminal\892B47EBC091D6EF95E3961284A76097\MQL5\Experts\DataScience\LogisticRegression\LogisticRegressionLib.mqh";
//CLogisticRegression Logreg;

#include "LinearRegressionLib.mqh";
CMatrixRegression *m_lr;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+  
void OnStart()
  {       
//--- 

    string file_name = "Apple Dataset.csv",delimiter =",";
  
//---
/* 
    m_lr = new CMatrixRegression;
    Print("Matrix simple regression");
    m_lr.Init(2,"1",file_name,delimiter);
    m_lr.MultipleMatLinearRegMain();
    delete m_lr;
*/
//---

      m_lr = new CMatrixRegression;
      
      Print("Matrix multiple regression");
      m_lr.LrInit(8,"2,3,4,5,6,7",file_name,",",0.7);
      
      m_lr.corrcoeff();
      m_lr.MultipleMatLinearRegMain();    
      delete m_lr;       
   
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+