## chat compAnIon
<br />
<div align="center">
  <a href="https://github.com/chatcompAnIon/chatcompAnIon">
    <img src="https://raw.githubusercontent.com/cleemazzulla/chatcompAnIon/main/Chat%20Companion%20Logo.png" alt="Logo" width="400" height="400">
  </a>

  <h3 align="center">Welcome to chat compAnIon's repo!</h3>

  <p align="center">
    <br />
    <a href="https://cleemazzulla.github.io/chatcompAnIon/"><strong>Visit our Webpage»</strong></a>
    <br />
    <br />
  </p>
</div>

<!-- ADD IN LATER TABLE OF CONTENTS -->

<!-- GETTING STARTED -->

## Model Overview

**compAnIonv1** is a transformer-based large language model (LLM) trained for child grooming text classification in gaming chat room environments. It is a lightweight model with only 110,479,516 total parameters designed to deliver classification decisions in milliseconds within chat room dialogues. 

Predicting child grooming is an incredibly complex NLP task, further complicated by the significant class imbalance due to the scarcity of publicly available grooming chat data and the prevalence of nongrooming chat data. Our dataset, therefore, consists of ~3% positive classes. Through a combination of up/downsampling, we managed to achieve a final training data mix consisting of 25% positive grooming class instances. To address the remaining class imbalance, the model was trained using the [Binary Focal Crossentropy](https://arxiv.org/abs/1708.02002v2) loss function with a customized gamma, aimed at penalizing the model for overfitting on the easier-to-predict class. 

The model was initially designed to be trained on the chat texts, but it also adapts to new features engineered from linguistic analysis within the field of child grooming. As child grooming (especially in digital chats) often follows a [lifecycle of stages](https://safechild.org/understanding-grooming/) such as (building trust, isolating the child, etc.) we were able to capture and extract such features through various techniques such as through homegrown Bag of Words type models. The chat texts concatenated with the newly extracted grooming stages features were fed into the bert-base-uncased encoder to build embedding representations. The pooler output of the BERT model with a dimension of 768 was extracted for each conversation. This embedding representation was fed into dense neural network layers to produce an ultimate binary classification output. 

However, after exhaustive ablation studies and model architecture experiments, we discovered that including 1D convolutional layers on top of our text embeddings was a much more effective and automated way to extract features. As such, **compAnIonv1** relies solely on the convolutional filters to act as feature extractors before feeding into the dense neural network layers.

### Technical Specs & Hardware
  
| **Training Specs**  | **compAnIonv1**    | 
| :---         |     :---:      | 
| Instance Size  | NVIDIA g5.4xlarge     | 
| GPU    | 1       | 
| GPU Memory (GiBs)    | 24     | 
| vCPUs    | 16       | 
| Memory (GiB)    | 64      | 
| Instance Storage (GB)    | 1 x 600 NVMe SSD       | 
| Network Bandwidth (Gbps)    | 25       | 
| EBS Bandwidth (Gbps)    | 8       | 

### Model Data
Our model was trained on non-grooming chat data from several sources including IRC Logs, Omegle, and the Chit Chats dataset. See detailed table below: 
 <table>
  <tr>
    <th scope="col">Dataset</th>
    <th scope="col">Sources</th>
    <th scope="col"># Grooming conversations</th>
    <th scope="col"># Non-grooming conversations</th>
    <th scope="col"># Total conversations</th>
  </tr>
  <tr>
    <th scope="row">PAN12 Train</th>
    <td>Perverted Justice (True positives), IRC logs (True negatives), Omegle (False positives)</td>
    <td style="text-align: center;">2,015</td>
    <td style="text-align: center;">65,992</td>
    <td style="text-align: center;">68,007</td>
  </tr>
  <tr>
    <th scope="row">PAN12 Test</th>
    <td>Perverted Justice (True positives), IRC logs (True negatives), Omegle (False positives)</td>
    <td style="text-align: center;">3,723</td>
    <td style="text-align: center;">153,262</td>
    <td style="text-align: center;">156,985</td>
  </tr>
  <tr>
    <th scope="row">PJZC</th>
    <td>Perverted Justice (True positives)</td>
    <td style="text-align: center;">1,104</td>
    <td style="text-align: center;">0</td>
    <td style="text-align: center;">1,104</td>
  </tr>
</table>

  <dl>
  <dt><strong>PAN12:</strong></dt> 
  <dd>Put together as part of a 2012 competition to analyze sexual predators and identify high risk text.</dd>
  <dt><strong>PJZC:</strong></dt>
  <dd>Milon-Flores and Cordeiro put together PJZC using the same method as PAN12, but with newer data. Because PAN12-train was already imbalanced, we decided to use just the grooming conversations for training.</dd>
<dt><strong>NOTE:</strong> There is no overlap between PAN12 and PJZC; PJZC conversations from the Perverted Justice are from 2013-2014.</dt>
</dl>

See our [Datasets](https://github.com/chatcompAnIon/chatcompAnIon/tree/main/Datasets) folder for our pre-processed data.


## Getting Started
To help combat what has been deemed an as *AN INDUSTRY WITHOUT AN ANSWER*, chat compAnIon is making the first model **compAnIonv1** publicly available.

### Prerequisites

In order to run compAnIon-v1.0, the following installs are required: 

  ```python
    ! pip install -q transformers==4.17
    !git lfs install
  ```

### Installation & Example Run

Below is an example of how you can clone our repo to access our trained model and quickly run predictions from a notebook environment:

1. Clone the repo
   ```python
   !git clone https://huggingface.co/chatcompanion/compAnIonv1
   ```
3. Change directories in order to access the trained weights file
   ```python
   %cd compAnIonv1
   ```
4. Import the **compAnIonv1** model
   ```python
   from compAnIonv1 import *
   ```
5. Run inference
   ```python
   texts = ["School is so boring, I want to be a race car driver when I grow up!",
         "I can pick you up from school tomorrow, but don't tell your parents, ok?"]

   run_inference_model(texts)
   ```
   
## Ethics and Safety
* The team did not conduct any new labeling of our dataset to avoid imputing our biases regarding what constitutes child grooming. All of our positive class grooming instances stem from grooming chat logs used as evidence in successful court convictions of sexual predators.
* This model is intended to be a tool for parents to help detect and mitigate digital child grooming. We acknowledge the real impact of misclassification, such as false positives potentially damaging parent-child relationships and unintended potential consequences for the false accused.
  
## Intended Usage 
* The model's intended use case is to predict child grooming in chat room conversations. It is intended to be a supportive tool for parents and is not to be considered a source of truth.
* However, there may be multiple other use cases, especially within the Trust & Safety space companies may explore. For instance, companies with chat environments may benefit from using such a model along with sentiment analysis to monitor their chat rooms. Additionally, we believe this model will lend itself well to niche detection use cases such as elderly scam prevention and cyberbullying. 
* This model paves the way for incorporating child grooming detection and linguistic analysis into AI models. However, to truly propel this field forward, we recognize the necessity for continued research, particularly in the area of feature extraction from text as it pertains to child grooming.

## Limitations
* **compAnIonv1** is primarily trained on English text; it will only generalize well to other languages with additional training. 
* Our model was trained to predict based on a token window size of 400. Conversations may vary in length, so the model's reliability might become constrained when running on extensively long conversations.
* Language is ever-changing, especially among children. The model may perform poorly if there are shifts in grooming stages and their representation in linguistic syntax. 

<!-- CONTACT -->
## Contact

* [Courtney Mazzulla](https://www.linkedin.com/in/courtney-l-mazzulla/) - cleemazzulla@berkeley.edu
* [Julian Rippert](https://www.linkedin.com/in/julianrippert/) - jrippert@berkeley.edu
* [Sunny Shin](https://www.linkedin.com/in/sunnyshin1/) - sunnyshin@berkeley.edu
* [Raymond Tang](https://www.linkedin.com/in/raymond-tang-0807aa1/) - raymond.tang@berkeley.edu
* [Leon Gutierrez](https://www.linkedin.com/in/leongutierrez29/) - leonrafael29@berkeley.edu
* [Karsyn Lee](https://www.linkedin.com/in/karsynlee/) - karsyn@berkeley.edu

[Project Website](https://cleemazzulla.github.io/chatcompAnIon/)


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

This project was developed as a part of UC Berkeley's Master of Information and Data Science Capstone. We thank our Capstone advisors, Joyce Shen and Korin Reid, for their extensive guidance and continued support. We also invite you to visit our cohort's projects as well: [MIDS Capstone Projects: Spring 2024](https://www.ischool.berkeley.edu/programs/mids/capstone/2024a-spring)
