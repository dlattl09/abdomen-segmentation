# abdomen-segmentation
<br> 2019년 대한의용생체공학회 추계학술대회
## Deep learning 기반의 CT 영상 multi class bone segmentation
<br><br>
프로젝트 소개 <br>
<img width="975" alt="스크린샷 2022-02-11 오후 6 30 39" src="https://user-images.githubusercontent.com/54707924/153567852-75cef9b6-4849-4752-ad91-58119ca61792.png">
<img width="969" alt="스크린샷 2022-02-11 오후 6 30 58" src="https://user-images.githubusercontent.com/54707924/153568442-9953edcb-e260-443c-b210-edca56fc520b.png">


Bone segmentation of CT images plays an important role in clinical diagnosis. Although deep learning algorithms have been applied for many segmentation problems for human organs, manual segmentations are still inevitable for constructing the ground truth data. Here, we present an automatic bone segmentation approach which minimizes users’ manual interventions. We obtained the ground truth data from femur images by using thresholding and convolutional neural network. U-net was used for training dataset and was tested in four settings: (1) randomly initialized multi-class segmentation, (2) randomly initialized single-class segmentation, (3)pretrained multi-class segmentation, and (4)pretrained single-class segmentation. The dice coefficient was calculated as 0.932, 0.929, 0.924, 0.917, respectively  in each of the 4 cases.
