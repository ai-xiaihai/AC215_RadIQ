
## A Platform for Interactive Medical Image Interpretation

## Motivation

Imagine a generalist medical AI (GMAI) assistant — long been the dream of many patients and doctors. This assistant can interact with patients, help them understand their cases better, and give personalized suggestions. After a medical exam and before going into the doctor’s office, patients already have a basic understanding of their case, this assistant can improve hospital resources efficiency, enabling doctors to tackle more pressing challenges. In rural areas and third-world countries, where medical resources are scarce, this medical assistant can improve accessibility and equality, allowing everyone to reach medical specialists.

In this project, we would like to make a step towards GMAI through a grounded radiology report. More specifically, we aim to build an app that allows patients to better understand their chest X-ray diagnosis through an interactive web interface. By integrating chest X-rays with their associated radiology reports through multi-modal learning, users can highlight any phrases in the report by hovering over their mice, and the corresponding region in the medical image would light up, as shown in the following figure.

![](https://cdn-images-1.medium.com/max/3060/1*-qJbRyUAdqHSawmi9nCbWg.png)

## Dataset

### Dataset description

There are 2 relevant datasets for this project, summarized in the following figure.

![](https://cdn-images-1.medium.com/max/2594/1*t1gqD0q4TUBZVTDxGDilDw.png)

MIMIC-CXR (Medical Information Mart for Intensive Care — Chest X-rays) is a large, publicly available dataset of chest X-rays and associated radiology reports [4]. Developed by the MIT Laboratory for Computational Physiology and collaborators, MIMIC-CXR contains more than 220,000 radiographic studies from over 60,000 patients. Each X-ray image is linked to a de-identified radiology report, which includes free-text findings and impressions.

MS-CXR (Local Alignment Chest X-ray dataset) is a novel, publicly available dataset designed to facilitate the study of complex semantic modeling in biomedical vision-language processing [5]. Created by a team of researchers and board-certified radiologists, MS-CXR offers 1,153 image-sentence pairs of bounding boxes and corresponding phrases, encompassing eight different cardiopulmonary radiological findings with a roughly equal number of pairs for each finding. The dataset is derived from the public MIMIC-CXR dataset, and the annotations focus on locally-aligned phrase grounding.

Due to computational limitations in this project, we will not pre-train a multimodal model on a large dataset such as MIMIC-CXR. Instead, we will be adopting a foundation medical model, pre-trained on MIMIC-CXR, and using MS-CXR only for finetuning.

### Data preprocessing

In our project, the data processing procedures are containerized. This encapsulation encompasses tasks such as data downloading, resizing, normalization, and splitting. Both raw and processed data are securely stored on the Google Cloud Platform (GCP), ensuring easy retrieval and maintaining reproducibility.

Elevating our approach beyond the standard curriculum, we built a dataset class and dataloader to directly pull data from GCP bucket using google-cloud-storage package. This method effectively bypasses the need to first download data locally, offering two significant advantages:

 1. It addresses the common issue of data versioning among developers. This aspect is crucial in industrial projects where a large number of machine learning engineers (dozens or even hundreds) concurrently experiment with different models on the same database. With our approach, when the central database releases a new version, developers automatically access the most updated data during their training sessions. This ensures consistency and accuracy across various developmental stages.

 2. Our method is particularly beneficial for handling large-scale databases. In an era where machine learning increasingly focuses on big data, databases can scale up to terabytes. Downloading such massive volumes of data locally, or even on a remote server, is impractical. Our approach mitigates this challenge by enabling scalable access to data — fetching only what the model requires at a given time. This not only saves storage space but also enhances efficiency in data handling.

## Model development

### Related work

In this section, we outlined the state-of-the-art work in the related field. We believe our project is both timely and feasible, considering the recent papers and technologies in the field.

* **CheXzero**: This is a CLIP-based, self-supervised model designed for chest X-rays. Its primary functionality revolves around associating X-ray images with their corresponding reports, focusing mainly on classification rather than segmentation [1].

* **BioViL**: Building upon CheXzero’s foundation, BioViL introduces open-vocabulary segmentation capabilities. However, it achieves a modest mIoU of 0.22, primarily constrained by the limited size of the dataset it utilizes [2].

* **Biomed CLIP**: This recent publication boasts the compilation of the largest public biomedical dataset, extracting 15 million figure-caption pairs from research articles in PubMed Central. Despite its vast yet noisy dataset, the paper reported state-of-the-art performance on numerous biomedical vision-language tasks. Notably, there was no specific mention of its efficacy in open-vocabulary segmentation [3].

We aspire to utilize and expand upon the concepts and models mentioned above as the foundation for our project.

### Model architecture

At a high level, our model comprises 4 primary components: an image encoder, a text encoder, and a deconvolution decoder, as illustrated in the accompanying figure.

* **Image encoder**: An image is passed through this encoder to generate image embedding, represented as a tensor of shape $(\hat{H}, \hat{W}, d)$.

* **Text encoder**: A text prompt is passed through this encoder to generate text embedding, which is also a tensor but with shape $(1, d)$.

* **Cosine similarity and deconvolution**: A similarity map is generated by calculating the cosine similarity between the image and text embeddings, resulting in a tensor of shape $(\hat{H}, \hat{W})$. This similarity map is subsequently passed through a deconvolution layer to upscale to the original image dimensions, resulting in a tensor of shape $(H, W)$. Considering that cosine similarity values range from -1 to 1, we apply a sigmoid function to transform these into probability values between 0 and 1.

![](https://cdn-images-1.medium.com/max/3024/1*EYIrDEOfed_KnGiNhzcGiw.png)

Initially, our approach involved fine-tuning the image and text encoders without a decoder. In essence, we linearly projected the cosine similarity map back to the original image dimensions. However, this method did not yield significant performance improvements, suggesting that the encoders were already adept at capturing the necessary information. This insight led us to integrate a decoder to transform the embeddings into a segmentation mask.

Our initial choice for the loss function was binary cross-entropy (BCE). However, we encountered issues due to class imbalance (foreground vs background), which adversely affected the model’s performance. This challenge prompted us to transition to focal loss, a more advanced variant of BCE. In focal loss, the parameters $\alpha$ and $\gamma$ are critical hyperparameters that depend on the class sample ratio and the difficulty of classifying an object.

![](https://cdn-images-1.medium.com/max/2000/1*rO5TEPLZusgBGI1YW79TDQ.png)

![](https://cdn-images-1.medium.com/max/2106/1*gPb2KJj2Rs3gbx1COWOP6Q.png)

### Evaluation metric

Our evaluation process is distinct from the training procedure, necessitating the introduction of a specific threshold selection to assess model performance accurately. We convert the model-generated heatmaps and ground truth boxes into binary masks. The need for an appropriate threshold is crucial here, as it transforms our heatmap into a binary format, suitable for direct comparison with the binary mask of the ground truth box. The Dice score serves as our evaluation metric, offering a clear measure of the model’s prediction accuracy by comparing the similarity between the generated binary mask and the ground truth.

During inference, we can tune the threshold to optimize the dice value on the validation set, as shown in the figure below. Once the optimal threshold has been chosen, the performance can be evaluated on the test set.

![](https://cdn-images-1.medium.com/max/2000/1*RTxjU4z9fMIBemnAKvILgw.png)

## Deployment

### API

We created an InferenceEngine class to abstract away the details of the model and to handle the inference process. Specifically, the __init__ method loads the model and downloads the best checkpoint from wandb. The inference method takes in a text prompt and an image, preprocess the inputs, run the model, and returns the heatmap overlaid on top of the original image.

Furthermore, FastAPI is used to create an API endpoint that takes in a text prompt and an image, and returns the heatmap overlaid on top of the original image. The two main methods are startup_event and predict. On startup_event, the server will call the __init__ method of the InferenceEngine class. On predict, the server will call the inference method.

The frontend and API are integration are done through common I/O formats. The frontend sends a POST request to the API endpoint with the following JSON format:

    {
    "text": "text prompt",
    "image": "base64 encoded image"
    }

The API endpoint returns the heatmap in the following format:

    heatmap: "base64 encoded heatmap"

### Frontend
We created a simple React web app to allow patients easily interact with our AI model. The simple React app uses React Router for routing and React hooks for state management. We also leveraged state-of-the-art build tool Vite for better development expereince and performance.