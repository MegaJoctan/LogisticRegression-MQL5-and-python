//+------------------------------------------------------------------+
//|                                        MultipleMatRegression.mqh |
//|                                    Copyright 2022, Omega Joctan. |
//|                           https://www.mql5.com/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Omega Joctan."
#property link      "https://www.mql5.com/users/omegajoctan"

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

#define DBL_MAX_MIN(val) if (val>DBL_MAX) Alert("Function ",__FUNCTION__,"\n Maximum Double value Allowed reached"); if (val<DBL_MIN && val>0) Alert("Function ",__FUNCTION__,"\n MInimum Double value Allowed reached") 

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CMatrixRegression
  {
      protected:             
                           double  TestXDataSet[];
                           double  TestYDataSet[];
                           double  m_train_split;
                           
                           double  m_allyCopy[]; //original values of y copied
                           double  m_allxCopy[]; //original values of x copied
  
                   
      
                           int     m_handle;
                           string  m_filename;
                           string  DataColumnNames[];    //store the column names from csv file
                           int     rows_total;
                           int     x_columns_chosen;     //Number of x columns chosen
                           
                           bool    m_debug;
                           double  m_yvalues[];     //y values or dependent values matrix
                           double  m_allxvalues[];  //All x values design matrix
                           double  xT[];            //store the transposed values
                           string  m_XColsArray[];  //store the x columns chosen on the Init 
                           string  m_delimiter;
                           
                           double  Betas[]; //Array for storing the coefficients 
  
      protected:
                           
                           bool     fileopen(); 
                           void     GetColumnDatatoArray(int from_column_number, double &toArr[]);
                           void     GetColumnDatatoArray(int from_column_number, string &toArr[]);
                           void     GetAllDataToArray(double& array[]);
                           int      MatrixtypeSquare(int sizearr);
                           void     MatrixInverse(double &Matrix[],double &output_mat[]);
                           void     TrainTestSplit(double &Arr[], double& outTrain[], double& outTest[], int each_rowsize, int colsinArray=1);
      public:
      
                                    CMatrixRegression(void);
                                   ~CMatrixRegression(void);
                          
                           void     LrInit(int y_column, string x_columns="", string filename = NULL, string delimiter = ",",double train_size_split = 0.7, bool debugmode=true);
                           void     MatrixDetectType(double &Matrix[],int rows,int &__r__,int &__c__);
                           void     MatrixMultiply(double &A[],double &B[],double &output_arr[],int row1,int col1,int row2,int col2);
                           void     MatrixUnTranspose(double &Matrix[],int torows, int tocolumns);
                           void     Gauss_JordanInverse(double &Matrix[],double &output_Mat[],int mat_order);
                           void     MatrixPrint(double &Matrix[], int rows, int cols,int digits=0);
                           
                           void     LinearRegMain();
                           
                           double   corrcoef(double &x[],double &y[]);
                           void     corrcoeff();
                           double   mean(double &data[]); 
                           double   r_squared(double &y[],double &y_predicted[]);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CMatrixRegression::CMatrixRegression(void)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CMatrixRegression::~CMatrixRegression(void)
 {
    ArrayFree(TestXDataSet);
    ArrayFree(TestYDataSet);
    ArrayFree(DataColumnNames);
    ArrayFree(m_yvalues);
    ArrayFree(m_allxvalues);
    ArrayFree(m_XColsArray);
    ArrayFree(Betas);
    ArrayFree(xT);
    ArrayFree(m_allxCopy);
    ArrayFree(m_allyCopy);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CMatrixRegression::LrInit(int y_column,string x_columns="",string filename=NULL,string delimiter=",",double train_size_split = 0.7, bool debugmode=true)
 {
//--- pass some inputs to the global inputs since they are reusable

   m_filename = filename;
   m_debug = debugmode;
   m_delimiter = delimiter;
   m_train_split = train_size_split;
   
//---

   ushort separator = StringGetCharacter(m_delimiter,0);
   StringSplit(x_columns,separator,m_XColsArray);
   x_columns_chosen = ArraySize(m_XColsArray);
   ArrayResize(DataColumnNames,x_columns_chosen); 

//---

   if (m_debug) 
    {
      Print("Init, number of X columns chosen =",x_columns_chosen);
      ArrayPrint(m_XColsArray);
    }
    
//--- collect the data and store into corresponding arrays
   
   GetColumnDatatoArray(y_column,m_yvalues);
   GetAllDataToArray(m_allxvalues);  
     
   
// check for variance in the data set by dividing the rows total size by the number of x columns selected, there shouldn't be a reminder
   
   if (rows_total % x_columns_chosen != 0)
     Alert("There are variance(s) in your dataset columns sizes, This may Lead to Incorrect calculations");
   else
     {
       int single_rowsize = rows_total/x_columns_chosen;
       
       ArrayCopy(m_allxCopy,m_allxvalues); //copy these values before they are split
       ArrayCopy(m_allyCopy,m_yvalues);  
       
       
//--- Before all of that Let's split data into training and testing 
       
       if (m_train_split >= 1) //if there is no room for testing, do not split the data the model
        {
           if (!m_debug)
             Alert("TrainTest split has to be less than one to leave the room for testing");
           else 
             Print("trainTest split is greater than or equal to one,\n this model will not be tested");
        }
      else //if there is room for testing then split the data into training and testing dataset
       {
          TrainTestSplit(m_allxvalues,m_allxvalues,TestXDataSet,single_rowsize,x_columns_chosen);     
          TrainTestSplit(m_yvalues,m_yvalues,TestYDataSet,single_rowsize);
       }
      
//---  modify the single rowsize to fit the trainsize number of rows
 
       single_rowsize = (int)MathCeil(single_rowsize*train_size_split);
       
/*
       Print("TrainDataSet x");
       int train_rows = int(rows_total*train_size_split), test_rows = rows_total-train_rows;
       MatrixPrint(m_allxvalues,train_rows,x_columns_chosen);
       
       Print("TrainDataSet y");
       MatrixPrint(m_allxvalues,train_rows,1);
       
       Print("TestDataSet x");
       MatrixPrint(TestXDataSet,test_rows,x_columns_chosen);
       
       Print("TestDataSet y");
       MatrixPrint(TestYDataSet,train_rows,1);
*/  
      
// --- end of data split now le'ts begin the matrix calculations

      //--- Refill the first row of a design matrix with the values of 1   
       double Temp_x[]; //Temporary x array
       
       ArrayResize(Temp_x,single_rowsize);
       ArrayFill(Temp_x,0,single_rowsize,1);
       ArrayCopy(Temp_x,m_allxvalues,single_rowsize,0,WHOLE_ARRAY); //after filling the values of one fill the remaining space with values of x
      
       //Print("Temp x arr size =",ArraySize(Temp_x));
       ArrayCopy(m_allxvalues,Temp_x);
       ArrayFree(Temp_x); //we no longer need this array
       
       int tr_cols = x_columns_chosen+1,
           tr_rows = single_rowsize;
       
       ArrayCopy(xT,m_allxvalues);  //store the transposed values to their global array before we untranspose them
       MatrixUnTranspose(m_allxvalues,tr_cols,tr_rows); //we add one to leave the space for the values of one

//---
       ArrayResize(Betas,tr_cols); //let's also not forget to resize our coefficients matrix
//---
       
       
       if (m_debug)
         {
           //Print("Design matrix");
           //MatrixPrint(m_allxvalues,tr_cols,tr_rows);
           
           //Print("Transposed Design Matrix");
           //MatrixPrint(m_allxvalues,tr_rows,tr_cols); 
         } 
     }

 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CMatrixRegression::LinearRegMain(void)
 {
   
    double xTx[]; //x transpose matrix times x   
    
    int data_rowsize = rows_total/x_columns_chosen; //single rowsize for each dataset
   
    int tr_rows = (int) MathCeil(m_train_split * data_rowsize), 
        tr_cols = x_columns_chosen+1; //we add one to leave the space for the values of one in our design matrix


    MatrixMultiply(xT,m_allxvalues,xTx,tr_cols,tr_rows,tr_rows,tr_cols);
    
    
    if (m_debug)
     {
      Print("xTx");
      MatrixPrint(xTx,tr_cols,tr_cols,2);
     }

//---
    
    double inverse_xTx[];
    
    if (x_columns_chosen > 1)
      Gauss_JordanInverse(xTx,inverse_xTx,tr_cols);
    else 
      MatrixInverse(xTx,inverse_xTx);   
        
//---

    if (m_debug)
      {
         Print("xtx Inverse");
         MatrixPrint(inverse_xTx,tr_cols,tr_cols,7);
      }
      
//---
     
     double xTy[];
     MatrixMultiply(xT,m_yvalues,xTy,tr_cols,tr_rows,tr_rows,1); //remember!! the value of 1 at the end is because we have only one dependent variable y   
     
     if (m_debug)
       {
          Print("xTy");
          MatrixPrint(xTy,tr_cols,1,5);
       }

//---

     MatrixMultiply(inverse_xTx,xTy,Betas,tr_cols,tr_cols,tr_cols,1); 
     
     if (m_debug)
       {
         Print("Coefficients Matrix");
         MatrixPrint(Betas,tr_cols,1,5);
       }

//---

      Print("==== TRAINED LINEAR REGRESSION MODEL COEFFICIENTS ====");
      MatrixPrint(Betas,tr_cols,1,5);    

//---

   /////////////////////////////////////////////////////////////////////////
   ///////// END OF TRAINING  OF OUR MODEL /////////////////////////////////
   ////////////  NOW IT'S TIME TO TEST  /////////////////////////////////////
   /// since this model was primarly created on the data that was filtered 
   // 70% of it only was being used in creating the model and training it at the
   //// same time now it's time to use the rest of the data /////////////////
   ////////////////////////////////////////////////////////////////////////////

//--- Testing the model
     
   
   int index = 0 , index_from=0;
   int each_row = ArraySize(TestYDataSet);
   
   double TestPredicted[];
   ArrayResize(TestPredicted,each_row);
   
//---

    
  if (m_train_split < 1)  // if there is room for testing, test the model
   { 
      Print("========= LINEAR REGRESSION MODEL TESTING STARTED =========");
      
      for ( int i =0; i < each_row; i++ ) // let's predict the values first using the model coefficient
        {
          //Print("Test Array index ",index);
           double sum=0; int start=0; 
           for ( int j=0, beta_index=1; j < x_columns_chosen; j++ , index++,  beta_index++)
               {
                 index_from = start + i;
                 
                 sum += TestXDataSet[index_from] * Betas[beta_index];
                 //Print(" from ", index_from,/*" data = ",TestXDataSet[index_from],*/" betas index =",beta_index," arr size =",ArraySize(TestXDataSet));
                 
                 start += each_row;
               }    
               
           TestPredicted[i] = Betas[0] + sum; //plus the summation output with the constant
        }   
         
//--- Checking the Accuracy of testing dataset 

         Print("Tested Linear Model R square is = ",r_squared(TestYDataSet,TestPredicted));        
    }                 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CMatrixRegression::fileopen(void)
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
void CMatrixRegression::GetAllDataToArray(double &toArr[])
 {
    int counter=0; 
    for (int i=0; i<ArraySize(m_XColsArray); i++)
      {                    
        if (fileopen())
         {  
          int column = 0, rows=0;
          while (!FileIsEnding(m_handle))
            {
              string data = FileReadString(m_handle);

              column++;
   //---      
              if (column==(int)m_XColsArray[i])
                 {                      
                     if (rows>=1) //Avoid the first column which contains the column's header
                       {   
                           counter++;
                           
                           ArrayResize(toArr,counter); //array size for all the columns 
                           toArr[counter-1]=(double)data;
                       }   
                     else 
                        DataColumnNames[i]=data;
                 }
   //---
              if (FileIsLineEnding(m_handle))
                {                     
                   rows++;
                   column=0;
                }
            } 
          rows_total += rows-1; //since we are avoiding the first row we have to also remove it's number on the list here
          //adding a plus equals sign to rows total ensures that we get the total number of rows for the entire dataset
          
        }
         FileClose(m_handle); 
     }
    
    if (m_debug)
     Print("All data Array Size ",ArraySize(toArr)," consuming ", sizeof(toArr)," bytes of memory");
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CMatrixRegression::GetColumnDatatoArray(int from_column_number, double &toArr[])
 {
    int counter=0;
    
    int column = 0, rows=0;
    
    fileopen(); 
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
    FileClose(m_handle);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CMatrixRegression::GetColumnDatatoArray(int from_column_number, string &toArr[])
 {
    int counter=0;
    
    int column = 0, rows=0;
    
    fileopen(); 
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
    FileClose(m_handle);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CMatrixRegression::MatrixUnTranspose(double &Matrix[],int torows, int tocolumns)
 {
    int rows, columns;
    
    double Temp_Mat[]; //temporary array
  
      rows = torows;
      columns = tocolumns;
       
//--- UnTransposing Array Starting

          ArrayResize(Temp_Mat,ArraySize(Matrix));
          
          int index=0; int start_incr = 0;
          
           for (int C=0; C<columns; C++)
              {
                 start_incr= C; //the columns are the ones resposible for shaping the new array
                 
                  for (int R=0; R<rows; R++, index++) 
                     {
                       //if (m_debug)
                       //Print("Old Array Access key = ",index," New Array Access Key = ",start_incr);
                       
                       Temp_Mat[index] = Matrix[start_incr];                       
                       
                       start_incr += columns;
                     }
                     
              }
       
       ArrayCopy(Matrix,Temp_Mat);
       ArrayFree(Temp_Mat);
      
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CMatrixRegression::MatrixMultiply(double &A[],double &B[],double &output_arr[],int row1,int col1,int row2,int col2)
 {
//---   
   double MultPl_Mat[]; //where the multiplications will be stored
   
   if (col1 != row2)
        Alert("Matrix Multiplication Error, \n The number of columns in the first matrix is not equal to the number of rows in second matrix");
 
   else 
    { 
        ArrayResize(MultPl_Mat,row1*col2);
        
        int mat1_index, mat2_index;
        
        if (col1==1)  //Multiplication for 1D Array
         {
            for (int i=0; i<row1; i++)
              for(int k=0; k<row1; k++)
                 {
                   int index = k + (i*row1);
                   MultPl_Mat[index] = A[i] * B[k];
                 }
           //Print("Matrix Multiplication output");
           //ArrayPrint(MultPl_Mat);
         }
        else 
         {
         //if the matrix has more than 2 dimensionals
         for (int i=0; i<row1; i++)
          for (int j=0; j<col2; j++)
            { 
               int index = j + (i*col2);
               MultPl_Mat[index] = 0;
                
               for (int k=0; k<col1; k++)
                 {
                     mat1_index = k + (i*row2);   //k + (i*row2)
                     mat2_index = j + (k*col2);   //j + (k*col2)
                     
                     //Print("index out ",index," index a ",mat1_index," index b ",mat2_index);
                     
                       MultPl_Mat[index] += A[mat1_index] * B[mat2_index];
                       DBL_MAX_MIN(MultPl_Mat[index]);
                 }
               //Print(index," ",MultPl_Mat[index]);
             }
           //Print("Matrix Multiplication output");
           //ArrayPrint(MultPl_Mat);
           ArrayCopy(output_arr,MultPl_Mat);
           ArrayFree(MultPl_Mat);
       }
    }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CMatrixRegression::MatrixPrint(double &Matrix[],int rows,int cols,int digits=0) 
 {
   Print("[ ");
   int start = 0; 
    //if (rows>=cols)
      for (int i=0; i<cols; i++)
        {
          ArrayPrint(Matrix,digits,NULL,start,rows);
          start += rows;     
        } 
   printf("] \ncolumns = %d rows = %d",rows,cols);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CMatrixRegression::MatrixDetectType(double &Matrix[],int rows,int &__r__,int &__c__)
{
  int size = ArraySize(Matrix);
     __c__ = size/rows;
     __r__ = size/__c__;
     
   //if (m_debug) 
   // printf("Matrix Type \n %dx%d Before Transpose/Original \n %dx%d After Transposed/Array Format",__r__,__c__,__c__,__r__);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CMatrixRegression::Gauss_JordanInverse(double &Matrix[],double &output_Mat[],int mat_order)
 {
 
    int rowsCols = mat_order;
    
//---  
       if (mat_order <= 2) 
          Alert("To find the Inverse of a matrix Using this method, it order has to be greater that 2 ie more than 2x2 matrix");
       else
         {
           int size =  (int)MathPow(mat_order,2); //since the array has to be a square
              
// Create a multiplicative identity matrix 

               int start = 0; 
               double Identity_Mat[];
               ArrayResize(Identity_Mat,size);
               
               for (int i=0; i<size; i++) 
                 {
                     if (i==start)
                       {
                        Identity_Mat[i] = 1;
                        start += rowsCols+1;
                       }
                     else 
                        Identity_Mat[i] = 0;
                        
                 }
                 
               //Print("Multiplicative Indentity Matrix");
               //ArrayPrint(Identity_Mat);
               
//---
      
              double MatnIdent[]; //original matrix sided with identity matrix
              
              start = 0;
              for (int i=0; i<rowsCols; i++) //operation to append Identical matrix to an original one
                {
                   
                   ArrayCopy(MatnIdent,Matrix,ArraySize(MatnIdent),start,rowsCols); //add the identity matrix to the end 
                   ArrayCopy(MatnIdent,Identity_Mat,ArraySize(MatnIdent),start,rowsCols);
                  
                  start += rowsCols;
                }
              
//---
   
               int diagonal_index = 0, index =0; start = 0;
               double ratio = 0; 
               for (int i=0; i<rowsCols; i++)
                  {  
                     if (MatnIdent[diagonal_index] == 0)
                        Print("Mathematical Error, Diagonal has zero value");
                     
                     for (int j=0; j<rowsCols; j++)
                       if (i != j) //if we are not on the diagonal
                         {
                           /* i stands for rows while j for columns, In finding the ratio we keep the rows constant while 
                              incrementing the columns that are not on the diagonal on the above if statement this helps us to 
                              Access array value based on both rows and columns   */
                            
                            int i__i = i + (i*rowsCols*2);
                            
                            diagonal_index = i__i;
                                                        
                            int mat_ind = (i)+(j*rowsCols*2); //row number + (column number) AKA i__j 
                            ratio = MatnIdent[mat_ind] / MatnIdent[diagonal_index];
                            DBL_MAX_MIN(MatnIdent[mat_ind]); DBL_MAX_MIN(MatnIdent[diagonal_index]);
                            //printf("Numerator = %.4f denominator =%.4f  ratio =%.4f ",MatnIdent[mat_ind],MatnIdent[diagonal_index],ratio);
                            
                             for (int k=0; k<rowsCols*2; k++)
                                {
                                   int j_k, i_k; //first element for column second for row
                                   
                                    j_k = k + (j*(rowsCols*2));
                                    
                                    i_k = k + (i*(rowsCols*2));
                                    
                                     //Print("val =",MatnIdent[j_k]," val = ",MatnIdent[i_k]);
                                     
							                //printf("\n jk val =%.4f, ratio = %.4f , ik val =%.4f ",MatnIdent[j_k], ratio, MatnIdent[i_k]);
							                
                                     MatnIdent[j_k] = MatnIdent[j_k] - ratio*MatnIdent[i_k];
                                     DBL_MAX_MIN(MatnIdent[j_k]); DBL_MAX_MIN(ratio*MatnIdent[i_k]);                                    
                                }
                                
                         }
                  }
                  
// Row Operation to make Principal diagonal to 1
             
/*back to our MatrixandIdentical Matrix Array then we'll perform 
operations to make its principal diagonal to 1 */
     
             
             ArrayResize(output_Mat,size);
             
             int counter=0;
             for (int i=0; i<rowsCols; i++)
               for (int j=rowsCols; j<2*rowsCols; j++)
                 {
                   int i_j, i_i;
                    
                    i_j = j + (i*(rowsCols*2));
                    i_i = i + (i*(rowsCols*2));
                    
                    //Print("i_j ",i_j," val = ",MatnIdent[i_j]," i_i =",i_i," val =",MatnIdent[i_i]);  
                    
                    MatnIdent[i_j] = MatnIdent[i_j] / MatnIdent[i_i];   
                    //printf("%d Mathematical operation =%.4f",i_j, MatnIdent[i_j]); 

                    output_Mat[counter]= MatnIdent[i_j];  //store the Inverse of Matrix in the output Array
                    
                    counter++;
                 }
                            
         }
//---
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CMatrixRegression::MatrixInverse(double &Matrix[],double &output_mat[])
 {
// According to Matrix Rules the Inverse of a matrix can only be found when the 
// Matrix is Identical Starting from a 2x2 matrix so this is our starting point
   
   int matrix_size = ArraySize(Matrix);

   if (matrix_size > 4)
     Print("Matrix allowed using this method is a 2x2 matrix Only");
  
  if (matrix_size==4)
     {
       MatrixtypeSquare(matrix_size);
       //first step is we swap the first and the last value of the matrix
       //so far we know that the last value is equal to arraysize minus one
       int last_mat = matrix_size-1;
       
       ArrayCopy(output_mat,Matrix);
       
       // first diagonal
       output_mat[0] = Matrix[last_mat]; //swap first array with last one
       output_mat[last_mat] = Matrix[0]; //swap the last array with the first one
       double first_diagonal = output_mat[0]*output_mat[last_mat];
       
       // second diagonal  //adiing negative signs >>>
       output_mat[1] = - Matrix[1]; 
       output_mat[2] = - Matrix[2];
       double second_diagonal = output_mat[1]*output_mat[2]; 
       
       if (m_debug)
        {
          Print("Diagonal already Swapped Matrix");
          MatrixPrint(output_mat,2,2);
        }
        
       //formula for inverse is 1/det(xTx) * (xtx)-1
       //determinant equals the product of the first diagonal minus the product of the second diagonal
       
       double det =  first_diagonal-second_diagonal;
       
       if (m_debug)
       Print("determinant =",det);
       
       
       for (int i=0; i<matrix_size; i++)
          { output_mat[i] = output_mat[i]*(1/det); DBL_MAX_MIN(output_mat[i]); }
       
     }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CMatrixRegression::MatrixtypeSquare(int sizearr)
 { 
//function for checking if the matrix is a square matrix or not
   
    int squarematrices[9] = {4,9,16,25,36,49,64,81,100}; //the squares of 2...10
    //int divident=0;
    int type=0;
    
     for (int i=0; i<9; i++)
       {
          if (sizearr % squarematrices[i] == 0)
             {
                //divident = sizearr/squarematrices[i];
                type = (int)sqrt(sizearr);
                printf("This is a %dx%d Matrix",type,type);
                break;
             }
             
          if (i==9 && sizearr % squarematrices[i] !=0 ) //if after 10 iterations the matrix size couldn't be found on the list then its not a square one
           Print("This is not a Square Matrix");
       }
    return (type);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CMatrixRegression::TrainTestSplit(double &Arr[], double& outTrain[], double& outTest[], int each_rowsize, int colsinArray=1)
 {
   double TestArr[], TrainArr[];
  
   int selected_x_cols = colsinArray;
   
   int start = 0, index=0;
   int train_index = 0;
   
   int total_size = ArraySize(Arr);  //size of the original array
       
   int train_size = (int)MathCeil(each_rowsize*m_train_split);
   int test_size = (int)MathFloor(each_rowsize*(1-m_train_split));

//--- 
   
   
   int start_train =0, start_test=0;
   for (int i=0; i<selected_x_cols; i++)
    {
         start_test += train_size;
         
         //printf("start train %d, start test %d",start_train,start_test);
         
         ArrayCopy(TrainArr, Arr , ArraySize(TrainArr) , start_train, train_size);
         ArrayCopy(TestArr, Arr ,  ArraySize(TestArr) , start_test, test_size);
                                            
         start_train += each_rowsize;
         
    }
    
//--- Get the output

   ArrayFree(outTrain); //free preexisting data
   ArrayFree(outTest);  
    
   ArrayCopy(outTrain,TrainArr);
   ArrayCopy(outTest,TestArr);
   
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CMatrixRegression::mean(double &data[])
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
double CMatrixRegression::corrcoef(double &x[],double &y[])
 {
   double r=0;
   
   double mean_x = mean(x);
   double mean_y = mean(y);
   
   double numerator =0, denominator =0;
   double x__x =0, y__y=0;
   
   for(int i=0; i<ArraySize(x); i++)
     {
         numerator += (x[i]-mean_x)*(y[i]-mean_y);
         x__x += MathPow((x[i]-mean_x),2);  //summation of x values minus it's mean squared 
         y__y += MathPow((y[i]-mean_y),2);  //summation of y values minus it's mean squared   
     }
     denominator = MathSqrt(x__x)*MathSqrt(y__y);  //left x side of the equation squared times right side of the equation squared
     r = numerator/denominator;   
    return(r);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CMatrixRegression::corrcoeff(void)
  {
    double TempXArr[]; //Temporary array
    double corrArr[];
    ArrayResize(corrArr,x_columns_chosen);
    
    int start_x = 0, single_colRows = rows_total/x_columns_chosen;
   
    Print("Correlation Coefficients"); 
    for (int i=0; i<x_columns_chosen; i++)
       {
          ArrayCopy(TempXArr,m_allxCopy,0,start_x, single_colRows);
          
          corrArr[i] = corrcoef(TempXArr,m_allyCopy);
          printf(" Independent Var Vs %s = %.3f",DataColumnNames[i],corrArr[i]);
          
          start_x += single_colRows;
       }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CMatrixRegression::r_squared(double &y[],double &y_predicted[])
 {
   double error=0;
   double numerator =0, denominator=0; 
   
   double mean_y = mean(y);
   
//--- 
  
  if (ArraySize(y_predicted)==0)
    Print("The Predicted values Array seems to have no values, Call the main Simple Linear Regression Funtion before any use of this function = ",__FUNCTION__);
  else
    {
      for (int i=0; i<ArraySize(y); i++)
        {
          numerator += MathPow((y[i]-y_predicted[i]),2);
          denominator += MathPow((y[i]-mean_y),2);
        }
      error = 1 - (numerator/denominator);
    }
   return(error);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
