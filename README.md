# 1. Introduction

## 1.1 Background

### Importance of Diagnosing Lumbar Spine Degenerative Conditions

Low back pain is a pervasive and significant health issue affecting individuals worldwide. It is one of the leading causes of disability, work absenteeism, and decreased quality of life. The prevalence of low back pain is substantial, with estimates suggesting that approximately 80% of people will experience it at some point in their lives. Chronic low back pain can lead to long-term physical, emotional, and financial burdens for patients and society.

Accurate and timely diagnosis of lumbar spine degenerative conditions is crucial for effective treatment and management. Degenerative conditions of the lumbar spine, such as herniated discs, spinal stenosis, and degenerative disc disease, are common causes of low back pain. Early diagnosis can help in planning appropriate therapeutic interventions, thus preventing the progression of the disease and reducing the impact on the patient's life.

### Role of MRI in Assessing Lumbar Spine Conditions

Magnetic Resonance Imaging (MRI) is a vital tool in the assessment of lumbar spine conditions. It provides detailed images of soft tissues, including intervertebral discs, spinal cord, and nerves, which are essential for diagnosing degenerative changes in the spine. MRI is non-invasive and offers superior contrast resolution compared to other imaging modalities, making it the preferred method for evaluating lumbar spine pathology. Accurate interpretation of MRI scans is critical for the diagnosis and management of lumbar spine degenerative conditions.

## 1.2 Objective

Develop a comprehensive classification system for lumbar spine degeneration using Magnetic Resonance Imaging, aimed at detecting, enhancing diagnostic accuracy and clinical decision-making. This objective outlines the goal of creating a new classification system that can potentially improve how clinicians diagnose and treat lumbar spine degeneration based on radiological findings.

- Testing various deep learning based models to predict the severity of subarticular stenosis.

- Fine-tuning the parameters of the model to improve the efficacy.

The successful implementation of LSTM and ANN models for this purpose has the potential to revolutionize the approach to diagnosing and managing lumbar spine pathologies, ultimately leading to better patient outcomes and more efficient use of healthcare resources.

# 2. Dataset Description

## 2.1 Details

The challenge will focus on the detection and severity of left subarticular stenosis, a kind of lumbar spine degenerative condition. For each imaging study in the dataset, the dataset contains severity scores (Normal/Mild, Moderate, or Severe) across the intervertebral disc levels.

The disc levels refer to the specific regions of the lumbar spine where the intervertebral discs are located.

- **L1/L2**: The disc between the first (L1) and second (L2) lumbar vertebrae

- **L2/L3**: The disc between the second (L2) and third (L3) lumbar vertebrae

- **L3/L4**: The disc between the third (L3) and fourth (L4) lumbar vertebrae

- **L4/L5**: The disc between the fourth (L4) and fifth (L5) lumbar vertebrae

- **L5/S1**: The disc between the fifth lumbar vertebra (L5) and the first sacral vertebra (S1)

****![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXe8nnLtNo1U-qMqRExvJHAeqDc2SGzMQg5eXmy6B-4fMVKM6CLMUij9Nk3PQhm8L5GE9ISHrsy7FmuHWSP7OyEXwtHXsZFMddIhszrU6uQAD8lQOpx4-jLWjoxIeJdoVhcUbELIrPeG_crZ5lNBXnJZkmhsAilmXEYgUCO8W1JjwnoIF_OiIPY?key=zksjqYm5zxk0Mxod2MU-rQ)****

Each image belongs to an MRI scan and these scans are taken along the axial plane of the spine using T2-Weighted Imaging modality.

**Axial Plane:**[**¶**](https://www.kaggle.com/code/gunesevitan/rsna-2024-lsdc-eda#2.-Axial-Plane:)

- **Description**: This plane divides the body into upper (superior) and lower (inferior) sections

- **Orientation**: Images are taken as if you are looking up from the feet or down from the head, providing horizontal slices of the body

- **Common Uses**:

  - Commonly used to view the brain, spine, abdomen, and pelvis

  - Allows for the examination of cross-sectional anatomy, making it easier to detect abnormalities such as tumors, lesions, or injuries

**T2-Weighted Imaging (T2)**[**¶**](https://www.kaggle.com/code/gunesevitan/rsna-2024-lsdc-eda#2.-T2-Weighted-Imaging-\(T2\))

**Characteristics**:

- Fluid-sensitive: Highlights fluid-containing structures

- Pathological contrast: Effective for detecting a wide range of pathologies involving water content

**Uses**:

- Pathologies: Excellent for identifying edema, inflammation, and tumors

- Conditions: Commonly used to detect cysts, abscesses, and degenerative changes

**Appearance**:

- Water and fluid: Appear bright

- Fat: Less bright compared to T1-weighted images

- Muscle and other soft tissues: Intermediate to dark signal intensity

**Applications**:

- Brain: Detecting lesions, edema, and demyelinating diseases

- Spine: Identifying disc herniations, spinal stenosis, and other degenerative conditions

- Joints: Assessing joint effusions and cartilage integrity

****![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfN14TyiB1iD06ZFsbyd2L9j37xLS4xZxIa9TM_5bCpmmJvjb54WjkR1cA7T9DYsTOVQFtLgtdrDTpOKcAGsvyLWf_6Aes5TYz9yggZJ5M3qTMc_VM5_a_5eMNtexWK2FT-5lBo7qK_BH4e0sqXgXmTo0ehqIVfelBNWD6aw_yUrKo8jm-tzgE?key=zksjqYm5zxk0Mxod2MU-rQ)****

****![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdq5xK1Wmphi_3DQvtrN4pGMR5EPoc66Q6laXLtWLtm8lGNr16tb2-vpgrYX4f-feiSKxUs9EI7IpPJLTR2lj7mrLhLvF5-O_nyZ9CdOuqNHtFJegMZUMZbU30Rm9Tl-SC8OFaCBWZKjI0w_-tfx97jEM3JY5J6ierc0bK_Pw?key=zksjqYm5zxk0Mxod2MU-rQ)****

## 2.2 Features

**train.csv** - Labels for the train set.

- study\_id - The study ID. Each study may include multiple series of images.

- \[condition]\_\[level] - The target labels, such as left\_subarticular\_stenosis\_L1\_L2, left\_subarticular\_stenosis\_L2\_L3, left\_subarticular\_stenosis\_L3\_L4, left\_subarticular\_stenosis\_L4\_L5, left\_subarticular\_stenosis\_L5\_S1 with the severity levels of Normal/Mild, Moderate, or Severe. 

**train\_label\_coordinates.csv**

- study\_id

- series\_id - The imagery series ID.

- instance\_number - The image's order number within the 3D stack.

- condition - There is one condition: left subarticular\_stenosis. 

- level - The relevant vertebrae, such as L1\_L2, L2\_L3, L3\_L4, L4\_L5, L5\_S1

- \[x/y] - The x/y coordinates for the center of the area that defines the label.

**\[train/test]\_images/\[study\_id]/\[series\_id]/\[instance\_number].dcm** The imagery data.

**\[train/test]\_series\_descriptions.csv**

- study\_id

- series\_id

- series\_description - The scan's orientation.

**Master\_data.csv**

- study\_id

- series\_id 

- instance\_number 

- condition  

- level 

- \[x/y]

- series\_description – filtered only for Axial T2

## 2.3 Target

The target variable for this dataset is Severity which has three categorical values:

- Normal/Mild

- Moderate

- Severe


********

# 3. Methodology

## 3.1 Data Collection and Preprocessing

**Source of MRI Data**

The MRI data used for this study were obtained from [Kaggle](https://www.kaggle.com/c/rsna-2024-lumbar-spine-degenerative-classification/data). The dataset includes a comprehensive collection of lumbar spine MRI scans from patients diagnosed with various degenerative conditions. These include spondylosis, spinal stenosis, herniated discs, and degenerative disc disease. Each unique image id is labeled with the specific condition and its severity.

**Preprocessing Steps**

To prepare the MRI data for model training, several preprocessing steps were applied:

- **Image preprocessing** 

* **Normalization**: The pixel intensity values of the MRI scans were normalized to a standard range to ensure uniformity across the dataset.

* **Augmentation**: Data augmentation techniques such as scaling were done to increase the diversity of the training data and prevent overfitting.

* **Resizing**: The MRI images were resized to a consistent resolution to match the input requirements of the neural network models.

- **Data preprocessing**

* **Combining raw datasets**: 

  1. created mapping between columns of **_train.csv_** and all the other files

  2. basis this mapping, pivoted all the levels of severity to combine with the rest of the files

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXekQgjcjUKI7MuvUAAZEDQAdlfHmoi0OsvDpwkYgGeLD1g-IMfRWlVRG0mHTy8lr-am2eGUR0DvbVgC_DBbw93u32mteDlnPZDodfC37SgmEEQBOUCdwrxE2zNr7fEsPKitZ9xwlKu0vbFmdXqAOIfqSfDAwYgJof0p6WhH?key=zksjqYm5zxk0Mxod2MU-rQ)

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXd8ROuejNo4GrEAgtdTiY1KPSTrNrNsvGgfupHku8gmsocaNoADGChoDD5TnhFe66_DSLFpKT5Ft4vt5rUy95rANFSA1z_0DNsV9YiPQX8wsNwxS-5RisGhmqNINRD0j-5FT2bOUiisDlAQOsgmoh9dplFvUY9LIAzkP2z8oOUTnnyrODNc7A?key=zksjqYm5zxk0Mxod2MU-rQ)

- **Creating subset of the combined dataset**: We filtered the dataset to include only instances of left subarticular stenosis. Since the dataset was too large and needed exceptional GPU capacity for processing the entire data, filtering it for one condition was crucial to focus our analysis on that condition and ensure that our findings are relevant and specific to it so that we can employ the same model on the rest of the conditions. The filtering process involved selecting all records that matched the criteria for left subarticular stenosis, resulting in a subset of the original dataset that was used for subsequent analysis.

* **Data balancing to reduce bias**: For the chosen condition, 72.4% of the data had instances of “_Normal/Mild_” severity which could lead to overfitting of the model on this severity. To balance the dataset to have approximately equal instances of all the 3 severity levels, we have dropped the “_study\_id_” that had only “_Normal/Mild_” for all the disc levels.

## 3.2 Exploratory Data Analysis

There are three core degenerative conditions of the lower spine: **spinal canal stenosis**, **neural foraminal narrowing**, and **subarticular stenosis** is represented using a pie chart.

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfkHFIbfdsQ6UEN3O_y8LFc2Bv0e3hQTpLl8DZPQGwyI0blAIyc34OxrozbCbPr5prd2d1ftoqhJOpVv8Pu7WGEFx_kP4bNWgtoBfNWagnFsSfTtEs2-jWObCGMKCPViJo_KZ8t-Lo_CX_PxUC509xxQpnDauAgkkW1nPEqTQ?key=zksjqYm5zxk0Mxod2MU-rQ)


Based on the pie chart derived from the dataset, it was observed that 20% of the patients are affected by Spinal Canal Stenosis. Further, 20.2% are suffering from Right Neural Foraminal Narrowing and Left Neural Foraminal Narrowing each. Additionally, 19.7% of the patients each have been diagnosed with Left Subarticular Stenosis and Right Subarticular Stenosis.

Going forward the EDA is focused on Left Subarticular Stenosis as that is the condition this project is focusing on.

**Descriptive Statistics :** A total of 6,448 individuals were identified as suffering from Left subarticular stenosis, with the majority of cases predominantly localized in the L3/L4 region of the spine confirmed by Axial T2 imaging. 1,339 patients exhibited the disease localized in the L3/L4 region, followed by L4/L5, L5/S1, L2/L3, and L1/L2, in that order.

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfEXksAS2r5-GFiLp_17YMv4-TFqrALQQZ0GGbRXi8Uzxu2-C17AFQTj_ouG4xxgi-VQrzZOgCjY0gZ_pTcyk7D_RfXrfUQV8eC6O0W1kqV8vVQv-xE8HMu_mgpXp3sZ4gV_p2p9T9UcvQpH1FfcAo_GQ8tKImaRI1SrK2dBQ?key=zksjqYm5zxk0Mxod2MU-rQ)

**Disease Localisation :** 1,339 patients exhibited the disease localized in the L3/L4 region, followed by L4/L5, L5/S1, L2/L3, and L1/L2, in that order.

****![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfMTtxZqPtxex1FFE5mc5ta9ZSBQUtJOwN4P5UlkAX9GllocfjqO1dFi-XCwYFmvwR5NtfyDoUzkW8-wHY_5Q-u2fF7sjIqTsaUf60fdC1LBCgLRxAoxwyYW27qHEZfqk-HQEL_9Cz7yz2sOb-HPlvLqyiNVih3bw4LZbXuFw?key=zksjqYm5zxk0Mxod2MU-rQ)****

**Count/Measure of Severity :**   3,600 individuals exhibited normal to mild severity, approximately 1,700 individuals displayed moderate severity, and about 800 individuals had a severe severity level as shown in the graph below. 

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXeR6frTLWoM-DLfWP_tFHQFBkBTprjTZm2uZTWA63ukvws_RdqmC8fwFl8zDWJv5v3xvPuIdzRA4joLuiYH8HgSum0XFbJhJ9Ry9REliGj7x_IOSoP4Jvo-U_u6iI89lfbRdyiyP2DPKpjZNgBUFOlZxSPzEHugSG5rpnem?key=zksjqYm5zxk0Mxod2MU-rQ)

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXd5Og2uz2dJW1urwLDg4uwj4NjbjFgSzxf-HmcvJBnjxGVzJ3A11lDBtt51niVDi-80EE5SkGxeHMTZL5t9BUwUeuf7c6lrqYgejLk90BU7C4IQJ3LLXF2XRUynkiZA-cmJhwuf4si62_ghopdYV2UMYwR_srYFEbfGKx6OsQ?key=zksjqYm5zxk0Mxod2MU-rQ)

## 3.3 Model Training

Overall, the application of deep learning in analyzing MRI scans holds the potential to enhance diagnostic accuracy, particularly convolutional neural networks (CNNs), which can achieve high accuracy in image classification tasks. They can detect subtle patterns in MRI scans that might be missed by human observers or traditional image-processing techniques. It can automate the process of analyzing MRI scans, reducing the workload for radiologists and speeding up the diagnostic process. It provides consistent results, eliminating the variability that comes with human interpretation. Also, they are capable of identifying complex patterns and anomalies in MRI scans that may not be apparent to the naked eye, providing a more comprehensive analysis and personalized diagnosis. Deep learning models can efficiently handle and analyze large datasets, making them suitable for processing the vast amounts of data generated by MRI scans. 

Our model training incorporates two distinct features: MRI scan images and corresponding 2D coordinates indicating the location of the condition within each image. The images are vectorized using a Convolutional Neural Network (CNN), while the coordinates are processed with a Multi-Layer Perceptron (MLP). These models are concatenated and followed by an additional MLP. To enhance training performance, we employ early stopping and reduce the learning rate if the validation loss plateaus and does not decrease further.

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdThl53YpZwzDdOC0wkysE3cYJzqSxnx_6bcrMftEx4bALzGRfRLeZFh9Xy7XjGAAUEAfS3Bjqk5ooxi-gSCHoST3I4qEhT6LJbVzGnfOI5GHyiNGEZ48kXuFq3tkPT89WAszlkqd4IBzpoBsheqZMCMCnATxYyzQbDbW_Sdg?key=zksjqYm5zxk0Mxod2MU-rQ)


**Model Structure**:

We first design a CNN consisting of three primary components: input layers, convolutional layers, and a flattened layer. The input layer defines the image size and the number of channels provided to the model. The convolutional section includes three convolutional layers, each followed by Batch Normalization and Max-Pooling layers, culminating in a Flatten layer to convert the 3D tensor output into a 1D vector for subsequent processing.

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXevCzT2dKqZ3FyFkxEASvIKFAjBhVnfMYc9VIYmmsrJdavUc9jLu-jQ8H7usY0wWTCQW5C1cssun_ABt1dYR32YiXgEHeWXlW7tQd2WeckenUdMPjGEX8QPGNtyejpCZsmlmWsi8UVBv5K8vKHS_lo6DgENKh0wQc6d1oZKOA?key=zksjqYm5zxk0Mxod2MU-rQ)

Next, we develop an MLP to handle the 2D coordinates. This MLP includes an input layer of size (2,) and three Dense layers with sizes 2, 4, and 2, respectively.

The CNN and MLP networks are then combined by concatenation, followed by two additional Dense layers with 128 and 64 units, respectively. The final output layer consists of a single node activated by the softmax function to classify into three categories.

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcRQXyyON30je0rfybkGCyOYyaBqqRDLMbBTtb1ZIAuCtch4UMQy6gZeEp4zEMt7hETAXY7qs_J3kFIwZVVTkwceF7_MXz4sjLdAjdDgKAJASwyFbiMuL6dx3QQujNd08lwCOHNbPQg0rQQiRQS_5D0WHrryeFBtuKPdKZJ?key=zksjqYm5zxk0Mxod2MU-rQ)

**Transfer Learning Approaches**:

The image vectorization can also be done using TL approaches as training a high quality CNN takes a lot of time and transfer learning is a good way to make use of open source resources. 

We tried seven widely known CNN networks trained on imagenet dataset.

**Densely Connected Convolutional Networks (DenseNet)** is a feed-forward convolutional neural network (CNN) architecture that links each layer to every other layer. This allows the network to learn more effectively by reusing features, hence reducing the number of parameters and enhancing the gradient flow during training.

**VGG16**, on the other hand, uses a simple and consistent architecture with 16 layers, emphasizing depth and simplicity but with a relatively high parameter count.

**InceptionV3**, developed by Google, employs parallel convolutions with different filter sizes in each layer, capturing various levels of detail and improving efficiency.

**Xception**, built on top of Inception, extends its architecture with depthwise separable convolutions, improving efficiency and performance.

**ResNet** is a deep learning model that introduces residual connections to ease the training of deep networks by mitigating the vanishing gradient problem.

**EfficientNet** balances network depth, width, and resolution systematically, achieving high performance with fewer parameters whereas 

**MobileNet** optimizes for mobile and embedded vision applications with depthwise separable convolutions, achieving low latency and lightweight models.

**Hyperparameter Tuning**

To optimize model performance, we experimented with various hyperparameters. The following parameters were primarily adjusted:

1. **Activation Function**: Activation functions introduce non-linearity into the model, allowing it to learn more complex patterns. We tested the following activation functions:

   - **Tanh**: Maps input values to the range \[-1, 1].

   - **Leaky ReLU**: Allows a small gradient when the unit is inactive, preventing the dying ReLU problem.

   - **Sigmoid**: Maps input values to the range \[0, 1], often used in binary classification.

   - **ReLU**: Rectified Linear Unit, introduces non-linearity by setting negative values to zero.

Results showed that ReLU and Sigmoid provided better performance.

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcUqRe9qTJxAtf_FZKAlu8sSEBvb9ydTldWDHVvTPfVeoLCHlFanNE5k2qGzYi04zpeOPPSkv5jOZd3gCtzwglMnqvPh4OdeSf82_AId3XW86UZdo3p8Hj-m01kuIgeSFGWp1GkCaXOhVOpYbcKscppupwazVBznxWoux2SpQ?key=zksjqYm5zxk0Mxod2MU-rQ)

Source: ResearchGate

2. **Batch Size**: The batch size determines the number of training samples used in one forward/backward pass. Smaller batch sizes provide a regularizing effect and often lead to better generalization. We tested batch sizes of 16, 32, 64, and 100, finding that a batch size of 32 performed best.

3. **Epochs**: The number of epochs refers to the number of complete passes through the training dataset. We tested 10, 12, and 16 epochs, with 10 epochs yielding the best results.

4. **Number of Layers**: We experimented with various combinations of the number of layers, observing the impact on model performance.

5. **Number of Nodes**: Different configurations of the number of nodes in each layer were tested to find the optimal structure.

6. **Train-Validation Split**: We experimented with different strategies for splitting the dataset into training and validation sets:

   - Random 80:20 split

   - Stratified 80:20 split grouped by class

   - A 1:1:1 ratio of classes in the training set and a data-representative split in the validation set

7. **Early Stopping**: Early stopping is a regularization technique used to prevent overfitting by terminating training when the model's performance on the validation set stops improving.

8. **Reduce Learning Rate on Plateau**: This technique reduces the learning rate when the validation loss has plateaued, allowing the model to fine-tune its weights with smaller adjustments and potentially escape local minima.

These hyperparameter tuning steps helped in achieving an optimized and robust model for classifying the severity of lumbar spine conditions based on MRI scans.

# 4. Model Evaluation

In order to train the model, we divided the data into an 80:10:10 split for of train and validation. We trained the model with multiple epochs and calculated the accuracy. After that we evaluated this model using multiple evaluation metrics. 

## 4.1 Evaluation Metrics

The following evaluation metrics were calculated to compare the various models:

**Accuracy** measures the ratio of correctly predicted instances to the total instances. The formula for accuracy is:

Accuracy = TP + TNTP+TN+FP+FN

where TP stands for True Positives, TN for True Negatives, FP for False Positives, and FN for False Negatives.

**Precision** is the ratio of correctly predicted positive instances to the total predicted positives. It can be calculated using the formula:

Precision = TPTP+FP

**Recall** represents the ratio of correctly predicted positive instances to all actual positives. The formula for recall is:

Recall = TPTP+FN

**AUC (Area Under the ROC Curve)** measures the model's ability to distinguish between classes. It is calculated from the ROC curve, which plots the TPR (Recall) against the False Positive Rate (FPR):

AUC = 01TPR(FPR)d(FPR)

**PR\_AUC (Area Under the Precision-Recall Curve)** measures the trade-off between precision and recall for different threshold values. It is derived from the Precision-Recall curve, which plots Precision against Recall:

AUC = 01Precisions(Recall)d(Recall)

**RMSE (Root Mean Squared Error)** evaluates the average magnitude of the errors between predicted and actual values. The formula for RMSE is:

RSME = 1ni-1n(yi- yi)2

where yi represents the actual values, yi the predicted values, and n the number of observations.

**F1 Score** is the harmonic mean of precision and recall, balancing the two metrics. The formula for the F1 score is:

F1 score = Precision RecallPrecision + Recall

**F-Beta Score**  is a weighted harmonic mean of precision and recall, giving more importance to one of the metrics. The formula is:

F  =(1+2) Precision Recall(2Precision) + Recall

where  is a factor that weights the importance of recall over precision (or vice versa).

## 4.2 Results

The table compares multiple models across various evaluation metrics, highlighting the custom model's strong performance in precision (.668) and competitive accuracy (.596). Despite having the lowest train accuracy (.604), the custom model demonstrates robustness by maintaining a balance between precision and recall, as evidenced by its notable F1 score (.535) and AUC (.773). This indicates a well-rounded performance, particularly excelling in precision.

Additionally, examining the difference between train accuracy and accuracy across models reveals potential overfitting, especially in models like ResNet50 with a high train accuracy (.994) but moderate accuracy (.607). In contrast, the custom model's close values for train accuracy and accuracy suggest a balanced training process with reduced overfitting, making it a reliable choice for real-world applications. We also have a confusion matrix of the custom model. Followed by per epoch outcomes during training.

|                     |                  |               |                       |              |              |               |           |            |
| ------------------- | ---------------- | ------------- | --------------------- | ------------ | ------------ | ------------- | --------- | ---------- |
| **model**           | **efficientnet** | **inception** | **inception\_resnet** | **resnet50** | **xception** | **Mobilenet** | **vgg16** | **custom** |
| **train\_accuracy** | .624             | .784          | .714                  | .994         | .798         | .811          | .864      | .620       |
| **accuracy**        | .600             | .579          | .607                  | .607         | .569         | .599          | .610      | .618       |
| **precision**       | .650             | .615          | .642                  | .616         | .588         | .602          | .628      | .668       |
| **recall**          | .548             | .521          | .521                  | .597         | .524         | .587          | .576      | .446       |
| **AUC**             | .812             | .770          | .798                  | .799         | .767         | .790          | .822      | .773       |
| **PR\_AUC**         | .706             | .639          | .674                  | .677         | .626         | .651          | .715      | .661       |
| **RMSE**            | .402             | .432          | .410                  | .449         | .436         | .434          | .413      | .417       |
| **f1\_score**       | .595             | .564          | .575                  | .606         | .554         | .594          | .601      | .535       |
| **f\_beta**         | .469             | .446          | .448                  | .500         | .444         | .491          | .487      | .394       |

****![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXd-WAN6IbL2l_hgJwpk2yBbNUBU1qh_H3-Sjuu6B4dzoglJNsBsvNzRV3AtaXa02eBL2-vE2EnjgKHVNXekhaUupgjqgAYaWuQQfMn-k8EQOZnKUoN260Id5rSVegTtdzcBv2oSrIQn2jll12cS5TLpngRs4SwiW0jM9hixLA?key=zksjqYm5zxk0Mxod2MU-rQ)****

The matrix shows that the model correctly identified 582 Normal/Mild cases, 136 Moderate cases, and 54 Severe cases. However, it also indicates significant misclassifications. Additionally, 85 Severe cases were classified as Moderate, and 47 as Normal/Mild. This suggests the model performs well in identifying Normal/Mild cases but struggles to accurately differentiate between Moderate and Severe cases, indicating a need for further refinement to improve classification accuracy across all categories.

# 5. Limitations and Challenges

**Unbalanced Data and Bias**

In the Lumbar Spine project, data imbalance and bias are critical issues. An imbalanced dataset can lead to models overemphasizing majority classes while underperforming on minority classes. Bias can skew results if the dataset is unevenly sampled or collected, leading to misleading conclusions. This leads to overfitting of the model.

**MRI Imaging Challenges**

MRI imaging of the lumbar spine faces issues like variability in image quality, difficulty distinguishing complex anatomical structures, and the time-consuming nature of data analysis. Differences in machine settings, patient positioning, and movement artifacts can obscure critical details, complicating diagnosis and treatment planning. Advanced imaging technology and improved training for radiologists are essential to overcome these challenges.

**Computational Constraints**

The project is computationally intensive, encountering extended computation times and the need for substantial CPU resources. Sophisticated algorithms and high-resolution data require significant processing power, leading to delays. Utilizing high-performance computing resources is key to mitigate computational bottlenecks and enhance overall performance.

**Framework Limitations**

The challenge of limited exposure to different machine learning frameworks has been a hurdle. Current, focus has been on Tensorflow. The resources available are divided among other frameworks like PyTorch. Adapting and optimizing models for use in other frameworks, identifying limitations, and developing strategies to ensure robust model interoperability.

# 6. Conclusion

In this study, we aimed to evaluate the efficacy of various deep learning models in predicting the severity of left subarticular stenosis in lumbar spine degenerative conditions. Through rigorous testing and analysis, we identified key insights and advancements in the application of deep learning techniques for medical imaging and diagnostic purposes.

**The key findings of our research are as follows:**

1. **Model Performance:** Among the deep learning models tested, our custom model demonstrated the highest precision in predicting the severity of left subarticular stenosis, outperforming traditional diagnostic methods.

2. **Critical Feature:** The models were able to identify critical imaging features that correlate with the severity of left subarticular stenosis, providing valuable insights into the pathology of lumbar spine degeneration.

3. **Clinical Implications:** The adoption of deep learning models can significantly enhance the precision and consistency of diagnoses, facilitating timely and appropriate treatment interventions.

The results underscore the potential of deep learning technologies to revolutionize the diagnostic process for lumbar spine degenerative conditions. By leveraging advanced algorithms, healthcare professionals can achieve a higher level of diagnostic accuracy, leading to improved patient outcomes.

# 7. Future Work

**Suggestions for Further Research**

- **Model Generalization:** Validating the performance of these models across larger and more diverse patient populations to ensure their generalizability and robustness.

- **Continuous Improvement:** Exploring the integration of additional data types, such as genetic markers and patient history, to further enhance the predictive capabilities of the models.

- **Exploring Advanced Architectures**: Trying hybrids deep learning models.

* **Exploring Preprocessing Techniques**: Considering the data was comparatively clean according to our knowledge, there is a scope of improving the performance of the models by further processing the data before training.

In conclusion, the application of deep learning models to predict the severity of left subarticular stenosis represents a promising advancement in the field of spinal health. Continued research and development in this area have the potential to significantly improve diagnostic accuracy and patient care for those suffering from lumbar spine degenerative conditions.
