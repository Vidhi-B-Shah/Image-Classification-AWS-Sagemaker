# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.


## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

![dataset](./screenshots/Screenshot%202024-08-02%20at%202.54.17%20PM.png)
### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 


## Hyperparameter Tuning

In this experiment, I used the ResNet50 model to perform image classification. I performed hyperparameter search, using the hpo.py script, to find the optimal values for the learning rate and batch size. The learning rate was searched over a continuous range from 0.001 to 0.1, while the batch size was searched over a categorical set of values including 16, 32, 64, 128, 256, and 512.
Finally, used the best values for the learning rate and batch size to train the model for 20 epochs.

![training_1](./screenshots/Screenshot%202024-08-02%20at%203.14.33%20PM.png)

![hypertuning](./screenshots/Screenshot%202024-08-02%20at%203.14.56%20PM.png)


![training](./screenshots/Screenshot%202024-08-02%20at%203.33.47%20PM.png)

![training_2](./screenshots/Screenshot%202024-08-02%20at%203.16.12%20PM.png)



## Debugging and Profiling

Model debugging in SageMaker is performed using the smdebug library, which is part of the SageMaker Python SDK. This library provides hooks to capture tensor values at various points during the training process and includes rules to identify common training issues.


Set up the Debugger Rules and Hook Parameters to track specific metrics in the notebook `train_and_deploy.ipynb`. If the debugging output indicated any anomalies, we would review the CloudWatch logs and adjust the code as needed to resolve the issues.

### Results

After profiling and debugging, some recommendations were considered:

- Identify any bottlenecks (CPU, I/O) related to the step outliers.
- Try a different distributed training strategy or framework.
- Consider increasing the number of data loaders or using data pre-fetching.
- Opt for a larger instance type with more memory if the current memory usage is near the maximum.




## Model Deployment

![endpoint](./screenshots/Screenshot%202024-08-02%20at%204.14.06%20PM.png)

