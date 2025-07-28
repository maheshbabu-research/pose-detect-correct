# pose-detect-correct
Human Sitting and Yoga Pose Detection Correction and Pose Generation

This project is a part of the AAI-521 course in the Applied Artificial Intelligence Program at the University of San Diego (USD).

**Project Status:** Completed

**Project Objective:**

The main objective of this project is to develop a computer vision based real-time automated system to solve some of the problems of the new fast-paced digital age sedentary lifestyle specifically focusing on
personal health and fitness (Human sitting pose and yoga pose detection and correction).This project aims to develop an advanced system for Human Sitting and Yoga Pose Detection, Correction, and Pose Generation
using computer vision and deep learning techniques. The goal is to build accurate models for detecting sitting and yoga poses, provide real-time corrective feedback, and generate new poses for users.
The project aims to explore the effectiveness of pose detection models like MediaPipe and custom algorithms for posture correction and pose generation, making it useful for health, wellness, and fitness applications.  

**Partner(s)/Contributor(s):**

  - Mahesh Babu
  - Keerthana
  - Paritosh Umeshan

**Project Description:**

This project focuses on building a system for detecting, correcting, and generating sitting and yoga poses. The system leverages pose estimation techniques to detect human poses in real-time, compares these with
ideal reference poses, and provides corrective feedback. Additionally, it includes a feature for generating new poses based on detected postures using Stable Diffusion and offers personalized exercises and routines.
The pose detection is powered by the MediaPipe framework, which allows for tracking key human joints and creating a skeleton-like representation. The corrective algorithm compares deviations from the reference pose
and provides feedback for better posture. Pose generation is achieved through machine learning techniques, manipulating joint positions to suggest new poses.

**Challenges:**

  - Handling variations in posture due to different body types and environmental conditions.
  - Providing real-time feedback with low latency.
  - Ensuring that generated poses are realistic and safe for users.
  - Optimizing models to work in dynamic environments.


**Methods Used:**

  - Computer Vision
  - Pose Detection and Correction
  - Image Generation
  - Data Visualization
  - Data Manipulation
  - Python

**Technologies:**
  - MediaPipe for real-time pose estimation and correction
  - Custom Algorithms for pose correction
  - Stable Diffusion for Pose Generation
  - Python Libraries: OpenCV, TensorFlow, PyTorch, NumPy, Matplotlib, Scikit-learn

  **Frameworks:** PyTorch, TensorFlow, Mediapipe (Lugaresi et al., 2019), and OpenCV.

  **Libraries:** / Model: MobileNetV2, Stable Diffusion.

  **Tools:** NVIDIA CUDA for GPU acceleration, TensorBoard for model monitoring.

  **IDE:** Google Colab, VS Code

  **Hardware:** GPUs (e.g., NVIDIA RTX 30xx series, NVIDIA A100, NVIDIA L4, Tesla T4) for real-time processing and training large models.


**Dataset:**
    
The Yoga-82 dataset is a large-scale collection of images designed to aid in the development of machine learning models for recognizing and analyzing various yoga poses.
The dataset consists of images where 82 unique yoga poses are captured, each with annotations for pose class labels. These poses are useful for training deep learning models
in real-time yoga pose detection and pose correction applications.

**Dataset Statistics**
  
  **Selected Dataset:** Yoga82 Dataset - https://sites.google.com/view/yoga-82/home
  
  **Number of Images:** 14,000+ (labeled) images
  
  **Categories of Images:** 82 Distinct Yoga Poses
  
  **Pose Variations:** Includes images taken from multiple angles (side, front, back), ensuring diversity in the dataset
  
  **Annotations:** classifies images into 82 yoga poses
  
  **Keypoint Annotations:** Dataset doesnot provide keypoint annotations, but can be easily mapped standard pose estimations with models like OpenPose or HRNet for further keypoint detection.
  
  **Image Dimensions:** Resolution 1024x768 (HD Resolution)

  **Size of the Dataset:** 4 to 5 GB
  
  **Types of Poses:** Static, Balanced, Stretching, Strengthening, Breathing and Relaxation

  This dataset is perfect for training AI systems focused on yoga pose detection, pose classification, and correction, especially when integrated with keypoint detection models for real-time feedback
  in applications like mobile yoga assistants or posture correction systems.    


**Installation and Execution**

This project is primarily run using google colab because of the requirement of good GPU and also google drive to retrieve and store images and videos of yoga poses and human sitting poses.

To set up this project

1. **Clone the Repository:**
   Open a terminal and run the following command to clone the repository to your local machine:
   ```bash
   git clone https://github.com/maheshbabu-usd/aai-521-team07-pose-detect-correct
   ```

2. **Navigate to the Project Directory:**
   Change into the project directory:
   ```bash
   cd aai-521-team07-pose-detect-correct
   ```

3. **Setup google colab enviornment**
   Open https://colab.research.google.com using your respective gmail account
   Upload the notebooks into google colab
   Ensure you are using appropriate GPU and High RAM for faster execution
   Setup the Input and Output directories for images and videos in google drive
   
5. **Install Dependencies:**
   Install the required Python packages using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

6. **Run the project notebooks**
   Run in google colab


**Acknowledgments:**

We would like to thank Prof. Haisav Chokshi for his invaluable guidance and mentorship throughout the course. Special thanks to the developers of MediaPipe for providing an excellent pose estimation framework,
and to the open-source community for their contributions to machine learning and computer vision.
Additionally, we would like to acknowledge the authors of the Yoga-82 dataset for their contribution to pose classification in yoga poses:

**References:**

[1] Verma, Manisha, Sudhakar Kumawat, Yuta Nakashima, and Shanmuganathan Raman. "Yoga-82: A New Dataset for Fine-grained Classification of Human Poses." arXiv preprint arXiv:2003.00117 (2020). https://arxiv.org/abs/2004.10362

[2] Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
    4510–4520. https://doi.org/10.1109/CVPR.2018.00474

[3] Kar, A., Rai, S., & Gupta, P. (2021). Yoga-82: A new dataset for fine-grained classification of yoga postures. PeerJ Computer Science, 7, e734. https://doi.org/10.7717/peerj-cs.734

[4] Lugaresi, C., Tang, J., Nash, H., Majumdar, A., Ramasamy, D., Fuller, D., ... & Grundmann, M. (2019). MediaPipe: A framework for building perception pipelines. arXiv preprint arXiv:1906.08172. http://arxiv.org/abs/1906.08172

[5] Ravi, S. (2023, December 7). Building a body posture analysis system using MediaPipe. LearnOpenCV
https://learnopencv.com/building-a-body-posture-analysis-system-using-mediapipe/

[6] Siva, L. (2020, September 15). An easy guide for pose estimation with Google's MediaPipe. Medium. https://logessiva.medium.com/an-easy-guide-for-pose-estimation-with-googles-mediapipe-a7962de0e944
