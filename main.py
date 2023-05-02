import torch
import clip
import matplotlib.pyplot as plt
import numpy as np
from data_loader import ImageDataSet

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

data = ImageDataSet('data_smile_2/', transform=preprocess)
loader = torch.utils.data.DataLoader(data, batch_size=10, shuffle=False, drop_last=False)


text = clip.tokenize(["an image of a human with No smile",
                      "an image of a human with Subtle smile",
                      "an image of a human with teeth-baring smile",
                      "an image of a human with laughing smile"]).to(device)

with torch.no_grad():
    all_image_features = []
    for images, idx in loader:
        image_features = model.encode_image(images.to(device))
        all_image_features.append(image_features)
    all_image_features = torch.cat(all_image_features).cpu().numpy()
    text_features = model.encode_text(text)

    all_probs = []
    smile_intensity = []
    for images, idx in loader:
        print(idx)
        logits_per_image, logits_per_text = model(images, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        all_probs.append(probs)
        for i in range(len(probs)):
            smile_intensity.append(np.sum(np.array([1,2,3,4]) * np.array(probs[i])))

    # Get the indices that would sort the list of smile intensities
    sorted_indices = np.argsort(smile_intensity)
    print(sorted_indices)
    # Reorder the lists using the sorted indices
    all_probs = [all_probs[0][i] for i in sorted_indices]
    images = [images[i] for i in sorted_indices]
    smile_intensity = [smile_intensity[i] for i in sorted_indices]

    # Create a grid of subplots with one row and the same number of columns as images
    fig, axs = plt.subplots(1, len(images), figsize=(20, 5))

    # Plot each image in its corresponding subplot
    for i, img in enumerate(images):
        # Convert the image data to a numpy array and transpose its dimensions
        img_np = img.permute(1, 2, 0).numpy()

        # Normalize the image data to [0, 1]
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        # Plot the image in the corresponding subplot
        axs[i].imshow(img_np)
        axs[i].axis('off')
        axs[i].set_title(f"{smile_intensity[i]:.2f}")

    # Save the figure to a PNG file
    plt.savefig('figure.png')

    plt.show()

'''print("Label probs:", all_probs)
print("Smile Intensity:", smile_intensity)'''