//+------------------------------------------------------------------+
//|                                                   TestScript.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
#include "LogisticRegressionLib.mqh";
CLogisticRegression Logreg;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
    Logreg.Init("titanic.csv",",",false);
     
   
    string Sex[];
    int SexEncoded[];
    Logreg.GetDatatoArray(4,Sex);
    Logreg.LabelEncoder(Sex,SexEncoded,"male,female");
    
    //ArrayPrint(SexEncoded); 
    
//---

    double Age[];
    Logreg.GetDatatoArray(5,Age);
    Logreg.FixMissingValues(Age); 
    
    double y_survival[];
    int Predicted[];
    Logreg.GetDatatoArray(2,y_survival); 
    Logreg.LogisticRegression(Age,y_survival,Predicted);  
    
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

