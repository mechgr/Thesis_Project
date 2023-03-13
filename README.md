# Thesis_Project



This project involves an Electric motor fault detection and diagnosis through the use of Deep Learning technologies and architectures with the use of raw data collected
from AWE

## AWE dataset

The AWE dataset contains 7 files, "Normal", "Fault-1", "Fault-2", "Fault-3", "Fault-4", "Fault-5" respectively represent six classes of health conditions, 
and "Label" represents the corresponding labels for each health condition. Each data file is composed of 1000×4000 sampling points of vibration signals, 
where 1000 denote the data acquisition duration (i.e., 1000 seconds) and 4000 denote the sampling frequency.

Notably, The data are collected from real industrial facilities via destructive experiments, and each fault condition are formed by artificial destruction. 
Here, "Normal" indicates that the equipment is operated without any faults. "Fault-M" indicates that M gaskets are added to the upper bearing to make a M×3mm concentricity 
deviation between the upper bearing and lower bearing such that make the fault condition manually. Detailed description of the destructive experiment and data collection process can be found in the original paper.
Initially, the raw signal-data collected were modified and partitioned by creating a 23436×1024 x_data table and a y_data table containing the label of each signal. The same tables are then divided into three sets, the training data set (Train set) which constitutes 70% of the total set, 
the validation data set (Validation set) which constitutes 20% and the test set (Test set) which constitutes 10% of the total. 

The data was then modified through scaling and each row of 1024 positions was converted into a gray-scale table-image of 32×32×1. 
Then the above data were fed into the convolutional neural network (CNN) and Attention architectures.
