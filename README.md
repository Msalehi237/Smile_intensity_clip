-----------------------------------------------------------------------------------------------------------------------------------------------
**Smile Intensity with CLIP**

This Project measures smile intensity. Smile intensity is a usually considered a subjective measure of the degree to which a person is smiling. It can vary from a slight, subtle smile to a wide, tooth-baring grin, and everything in between.
![smile](https://user-images.githubusercontent.com/71953974/235794735-52c273e0-51ad-4e1e-9660-57ac127d48bf.png)

This project demonstrates how to use the CLIP (Contrastive Language-Image Pre-Training) model to classify the intensity of smiles in human faces. The CLIP model is a neural network that has been pre-trained on a large dataset of images and their associated textual descriptions, allowing it to learn representations that capture both visual and semantic information.

 We tried several way to measure the intensity of a smile including: 
- OpenCV Haar cascades for faces and smiles
- A Pretrained CNN based-network fine_tuned on ifwcrop dataset with smile/no-smile annotations
- Facenet Package
- CLIP Model

Among these various models and approaches, CLIP model showed the best results. While seconed approach was showing excellent results for the same subject:
![Unknown-2](https://user-images.githubusercontent.com/71953974/235799186-a961849b-a2bb-4bfd-aeee-8a3741d3767c.png)


The results for different subjects was not comparable:
![Unknown-3](https://user-images.githubusercontent.com/71953974/235799199-95046e09-0d43-499b-8c75-70ebc545e213.png)


-----------------------------------------------------------------------------------------------------------------------------------------------
**CLIP Model**
-----------------------------------------------------------------------------------------------------------------------------------------------

The code tokenizes four text descriptions of different smile intensities using clip.tokenize and encodes the image features using model.encode_image. It then concatenates all image features and encodes the text features using model.encode_text.

Finally, the code runs a loop over the images and calculates the smile intensity of each image using the probabilities obtained from the CLIP model. It prints the probabilities and the smile intensities obtained from the model. The output of the model shows sattisfing results for between-subject comparision situation:
![smile](https://user-images.githubusercontent.com/71953974/235794735-52c273e0-51ad-4e1e-9660-57ac127d48bf.png)

