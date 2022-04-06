//+------------------------------------------------------------------+
//|                                        LogisticRegressionLib.mqh |
//|                                    Copyright 2022, Omega Joctan. |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Omega Joctan."
#property link      "https://www.mql5.com/en/users/omegajoctan"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CLogisticRegression
  {
  private:
                        double e; //Euler's number
                        double m_xvalues[];
                        double m_yvalues[];
                        double m_slope; //store linear model slope
   protected:             
                        bool m_debug;
                        double y_mean;
                        double x_mean;
                        int m_handle;
                        string m_delimiter;
                        string m_filename;
                        int rows_total;
                        int columns_total;
                        
                        bool fileopen();
                        double mean(double &data[]);
                        void WriteToCSV(double& row1[],int& row2[],string file_name,string label1="Label01",string label02="label02");
   
   public:  
                        CLogisticRegression(void);
                       ~CLogisticRegression(void);
                       
                        void Init(string filename, string delimiter=",",bool debugmode=false); 
                        void GetDatatoArray(int from_column_number, string &toArr[]);
                        void GetDatatoArray(int from_column_number, double &toArr[]); 
                        void LabelEncoder(string& src[],int& EncodeTo[],string members="male,female"); //write all the members you want to encode
                        void FixMissingValues(double& Arr[]);
                        double LogisticRegression(double &x[], double& y[],int& Predicted[],double train_size = 0.7);
                        void ConfusionMatrix(double &y[], int &Predicted_y[], double &accuracy); 
                        double LogLoss(double& rawPredicted[]);
                        
                      //--- Linear regression stuff
                        double y_intercept();
                        double coefficient_of_X();
                        
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CLogisticRegression::CLogisticRegression(void)
 {
    e = 2.718281828; 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CLogisticRegression::~CLogisticRegression(void)
 {
    FileClose(m_handle); //Just to Avoid Err=5004
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CLogisticRegression::fileopen(void)
 { 
    m_handle  = FileOpen(m_filename,FILE_READ|FILE_CSV|FILE_ANSI,m_delimiter); 

    if (m_handle == INVALID_HANDLE)
      {
         return(false);
         Print(__FUNCTION__," Invalid csv handle err=",GetLastError());
      }
   return (true);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CLogisticRegression::Init(string filename,string delimiter=",",bool debugmode=false)
 {
    m_delimiter = delimiter;
    m_filename = filename;
    m_debug = debugmode;
    
//---
    if (fileopen())
     {
       int column = 0, rows=0;
       while (!FileIsEnding(m_handle))
         {
          string data = FileReadString(m_handle);
           if (rows==0) 
             {
               columns_total++; 
             }
            
           column++;
            
           if (FileIsLineEnding(m_handle))
             {
               rows++;
               column=0;
             }
         }
         rows_total = rows;
         columns_total = columns_total;
     }
//---
   FileClose(m_handle);
  
   if (m_debug) 
    Print("rows ",rows_total," columns ",columns_total);
 }
//+------------------------------------------------------------------+
//|                     Get string data                              |
//+------------------------------------------------------------------+
void CLogisticRegression::GetDatatoArray(int from_column_number, string &toArr[])
 {
    int counter=0;
    
    if (fileopen()) 
     {
       int column = 0, rows=0;
       while (!FileIsEnding(m_handle))
         {
           string data = FileReadString(m_handle);
           
           column++;
//---      
           if (column==from_column_number)
              {
                  if (rows>=1) //Avoid the first column which contains the column's header
                    {   
                        counter++;
                        ArrayResize(toArr,counter); 
                        toArr[counter-1]=data;
                    }   
                     
              }
//---
           if (FileIsLineEnding(m_handle))
             {                     
               rows++;
               column=0;
             }
         }
     }
   FileClose(m_handle);
 }
//+------------------------------------------------------------------+
//|                       Get double values to Array                 |
//+------------------------------------------------------------------+
void CLogisticRegression::GetDatatoArray(int from_column_number, double &toArr[])
 {
  
    int counter=0;
    
    if (fileopen())
     {
       int column = 0, rows=0;
       while (!FileIsEnding(m_handle))
         {
           string data = FileReadString(m_handle);
           
           column++;
//---      
           if (column==from_column_number)
              {
                  if (rows>=1) //Avoid the first column which contains the column's header
                    {   
                        counter++;
                        ArrayResize(toArr,counter); 
                        toArr[counter-1]=(double)data;
                    }   
                     
              }
//---
           if (FileIsLineEnding(m_handle))
             {                     
               rows++;
               column=0;
             }
         }
     }
   FileClose(m_handle);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+ 
void CLogisticRegression::LabelEncoder(string &src[],int &EncodeTo[],string members="male,female")
 {
   string MembersArray[];
   ushort separator = StringGetCharacter(m_delimiter,0);
   StringSplit(members,separator,MembersArray); //convert members list to an array
   ArrayResize(EncodeTo,ArraySize(src)); //make the EncodeTo array same size as the source array
   
      int binary=0;
      for(int i=0;i<ArraySize(MembersArray);i++) // loop the members array
        {
           string val = MembersArray[i];
           binary = i; //binary to assign to a member
           int label_counter = 0;
           
           for (int j=0; j<ArraySize(src); j++)
              {
                string source_val = src[j];
                 if (val == source_val)
                   {
                    EncodeTo[j] = binary;
                    label_counter++;
                   }
              } 
           Print(MembersArray[binary]," total =",label_counter," Encoded To = ",binary);
        } 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CLogisticRegression::FixMissingValues(double &Arr[])
 {
   int counter=0; double mean=0, total=0;
   for (int i=0; i<ArraySize(Arr); i++) //first step is to find the mean of the non zero values
       {
         if (Arr[i]!=0)
           {
             counter++;
             total += Arr[i];
           }
       }
       
     mean = total/counter; //all the values divided by their total number
     
     if (m_debug)
      {
        Print("mean ",MathRound(mean)," before Arr");
        ArrayPrint(Arr);
      }
      
     for (int i=0; i<ArraySize(Arr); i++)
       {
         if (Arr[i]==0)
           {
             Arr[i] = MathRound(mean); //replace zero values in array
           }
       }
    
    if (m_debug)
     {
        Print("After Arr");
        ArrayPrint(Arr); 
     }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CLogisticRegression::LogisticRegression(double &x[],double &y[],int& Predicted[],double train_size_split = 0.7)
 { 
 
   double accuracy =0; //Accuracy of our Train/Testmodel
   int arrsize = ArraySize(x); //the input array size
   double p_hat =0; //store the probability 
   
//---
  
   int train_size = (int)MathCeil(arrsize*train_size_split);
   int test_size = (int)MathFloor(arrsize*(1-train_size_split));
   
   ArrayCopy(m_xvalues,x,0,0,train_size); 
   ArrayCopy(m_yvalues,y,0,0,train_size); 
   
//---

   y_mean = mean(m_yvalues);
   x_mean = mean(m_xvalues);
   
//   Training our model in the background

   double c = y_intercept(), m = coefficient_of_X(); 


//--- Here comes the logistic regression model
        
      int TrainPredicted[];
      double RawPredicted[]; //the predicted values before round off
      double sigmoid = 0;
      
      ArrayResize(RawPredicted,train_size);
      ArrayResize(TrainPredicted,train_size); //resize the array to match the train size
      Print("Training starting..., train size=",train_size);
      
      for (int i=0; i<train_size; i++)
        { 
          double y_= (m*m_xvalues[i])+c;
          double z = log(y_)-log(1.0-y_); //log odds
          
          p_hat = 1.0/(MathPow(e,-z)+1.0);
          RawPredicted[i] =  p_hat;
               
          TrainPredicted[i] = (int) round(p_hat); //round the values to give us the actual 0 or 1  
          
          if (m_debug)
           PrintFormat("%d Age =%.2f survival_Predicted =%d ",i,m_xvalues[i],TrainPredicted[i]);
        }
      ConfusionMatrix(m_yvalues,TrainPredicted,accuracy); //be careful not to confuse the train predict values arrays
      printf("Train Model Accuracy =%.5f",accuracy);
      
//--- Log loss on the predicted values
 
    LogLoss(RawPredicted);

//--- Testing our model 

   if (train_size_split<1.0) //if there is room for testing
      {         
      
         ArrayRemove(m_xvalues,0,train_size); //clear our array
         ArrayRemove(m_yvalues,0,train_size); //clear our array from train data
   
         ArrayCopy(m_xvalues,x,0,train_size,test_size); //new values of x, starts from where the training ended
         ArrayCopy(m_yvalues,y,0,train_size,test_size);  //new values of y, starts from where the testing ended
         
         ArrayResize(RawPredicted,test_size);
         ArrayResize(Predicted,test_size); //resize the array to match the test size
         Print("start testing...., test size=",test_size);
          
         for (int i=0; i<test_size; i++)
           { 
             double y_= (m*m_xvalues[i])+c;
             double z = log(y_)-log(1-y_); //log loss
             
             p_hat = 1.0/(MathPow(e,-z)+1);
             RawPredicted[i] =  p_hat;
                          
             Predicted[i] = (int) round(p_hat); //round the values to give us the actual 0 or 1  
          
             if (m_debug)  
               PrintFormat("%d Age =%.2f survival_Predicted =%d , Original survival=%.1f ",i,m_xvalues[i],Predicted[i],m_yvalues[i]);  
           }
        ConfusionMatrix(m_yvalues,Predicted,accuracy);
        printf("Testing Model Accuracy =%.5f",accuracy);
        
//--- Log loss on the predicted values
 
        LogLoss(RawPredicted);
 
      } 
      
    return (accuracy); //Lastly, the testing Accuracy will be returned
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CLogisticRegression::ConfusionMatrix(double &y[], int &Predicted_y[], double& accuracy)
 {
    int TP=0, TN=0,  FP=0, FN=0; 
    
    for (int i=0; i<ArraySize(y); i++)
       {
         if ((int)y[i]==Predicted_y[i] && Predicted_y[i]==1)
            TP++;
         if ((int)y[i]==Predicted_y[i] && Predicted_y[i]==0)
            TN++;
         if (Predicted_y[i]==1 && (int)y[i]==0)
            FP++;
         if (Predicted_y[i]==0 && (int)y[i]==1)
            FN++;
       }
     Print("Confusion Matrix \n ","[ ",TN,"  ",FP," ]","\n","  [  ",FN,"  ",TP,"  ] ");
     accuracy = (double)(TN+TP) / (double)(TP+TN+FP+FN);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CLogisticRegression::LogLoss(double &rawPredicted[])
 { 
   double log_loss =0;
   double penalty = 0;
   for (int i=0; i<ArraySize(rawPredicted); i++)
      {
        penalty += -((m_yvalues[i]*log(rawPredicted[i])) + (1-m_yvalues[i]) * log(1-rawPredicted[i]));
         
         if (m_debug)
            printf(" penalty =%.5f",penalty); 
      }
    log_loss = penalty/ArraySize(rawPredicted);
    Print("Logloss =",log_loss);
    
    return(log_loss);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
//>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//+------------------------------------------------------------------+
//|             Here comes Linear Model                              |
//+------------------------------------------------------------------+
double CLogisticRegression::mean(double &data[])
 {
   double x_y__bar=0;
   
   for (int i=0; i<ArraySize(data); i++)
     {
      x_y__bar += data[i]; // all values summation
     }
           
    x_y__bar = x_y__bar/ArraySize(data); //total value after summation divided by total number of elements
   
   return(x_y__bar); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CLogisticRegression::y_intercept()
 {
   // c = y - mx
   return (y_mean-m_slope*x_mean);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CLogisticRegression::coefficient_of_X()
 { 
   double m=0;
//---      
      double x__x=0, y__y=0;
      double numerator=0, denominator=0; 
      
      for (int i=0; i<ArraySize(m_xvalues); i++)
       { 
            x__x = m_xvalues[i] - x_mean; //right side of the numerator (x-side)
            y__y = m_yvalues[i] - y_mean; //left side of the numerator  (y-side)
             
         numerator += x__x * y__y;  //summation of the product two sides of the numerator 
         denominator += MathPow(x__x,2); 
         
       } 
      
      m = numerator/denominator; 
      m_slope = m; //store the slope to a global slope of a linear model
      
   return (m);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CLogisticRegression::WriteToCSV(double& row1[],int& row2[],string file_name,string label1="Label01",string label02="label02")
 {
    FileDelete(file_name); 
    m_handle  = FileOpen(file_name,FILE_READ|FILE_WRITE|FILE_CSV|FILE_ANSI,m_delimiter); 

    if (m_handle == INVALID_HANDLE)
      {
         Print(__FUNCTION__," Invalid csv handle err=",GetLastError());
      }
   else
       { 
            FileSeek(m_handle,0,SEEK_SET);
            FileWrite(m_handle,label1,label02); 
            
            for (int i=0; i<ArraySize(row2); i++)               
                FileWrite(m_handle,DoubleToString(row1[i],0),IntegerToString(row2[i],0));
                 
      }
      
    FileClose(m_handle);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
