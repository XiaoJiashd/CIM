# CIM

An imputation model for congestion level data. It can complete the missing traffic congestion levels of road segments in the urban road network via capturing the characteristices of congestion levels.

python 3.6

maskTensor.py is used to generate the mask tensor of the congestion level tensor and training sets of different sizes (20%, 40%, 60%, 80%).  
The code of CIM model is wrote in cim. The input is a traffic congestion level tensor which contains some missing values and the output is a complete tensor. 

data:  
rushHour.csv can be organized into a congestion level tensor which has three modes, including segment, time slot and day. It has 25 sections separated by "# New slice", and each section has 317 rows, indicating the congestion levels of 317 sections on that day. It only considers the traffic congestion data from 6:00 to 21:00 and the size of time slot is 5 minutes. So each line has 180 elements.  
rushHour_10.csv is a congestion level tensor. The data source is the same as rushHour.csv, but the difference is that each line of it has 90 elements because the size of time slot is 10 minutes.  
rushHour_15.csv is a congestion level tensor with the size of 15 minutes. In this setting, each line has 60 elements.
