# PCLSD
Implementation of the paper PCLSD: Line segment detection based on predictive correction mechanism, accepted on ICASSP 2024. PCLSD is a universal linear detector that starts from the prediction stage of generating line segment predictions based on Canny. In the subsequent correction stage, each predicted line segment undergoes refinement. Specifically, directed routing methods are used to extend and re fit line segments, improving their accuracy in direction, position, and integrity. Then verify the corrected line segments to ensure confidence. The experimental results show that the proposed PCLSD performs better than the most advanced existing methods.    

## Full install  
If you want to run PCLSD.  
Dependencies that need to be installed on your system:  
OpenCV  4.1.1  

 
