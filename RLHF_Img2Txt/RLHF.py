import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import hpsv2

import matplotlib.pyplot as plt
import clip
from PIL import Image

# Set the device
device = "cpu"
model_id = "stabilityai/stable-diffusion-2-1"

# Load the Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, low_cpu_mem_usage=True, torch_dtype=torch.float32
).to(device)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.safety_checker = None  # Disable safety filter for simplicity

pipe = pipe.to(device)

# Set up HPSv1
model, preprocess = clip.load("ViT-L/14", device=device)
params = torch.load("hpc.pt", map_location=device)['state_dict']
model.load_state_dict(params)

# Step 1: Generate the original image as a PyTorch tensor
prompt = "A tall red tree"
num_inference_steps = 13  # Use fewer steps for faster generation

with torch.no_grad():  # No gradient needed during generation
    generated = pipe(prompt, num_inference_steps=num_inference_steps,low_cpu_mem_usage=True, output_type="pt")
    images = generated.images.to(device)  # Ensure it's on the correct device

# Enable gradient tracking
images.requires_grad_(True)

# Used for control test change back when done
hps_prompt = prompt

# Save and display the original image
original_image = (images[0].detach().permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
Image.fromarray(original_image).save("original_image.png")
original_hpsv1_image = preprocess(Image.open("original_image.png")).unsqueeze(0).to(device)
hpsv1_images = torch.cat([original_hpsv1_image], dim=0)
hpsv1_text = clip.tokenize([hps_prompt]).to(device)


# Score the original image
original_score = hpsv2.score("original_image.png", hps_prompt, hps_version="v2.1")[0]
with torch.no_grad():
    image_features = model.encode_image(hpsv1_images)
    text_features = model.encode_text(hpsv1_text)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    original_hps = image_features @ text_features.T

#original_score = hpsv2.score("original_image.png", prompt, hps_version="v2.1")[0]
print(f"Hpsv2 Original Score: {original_score:.2f}")
print(f"Hpsv1 Original Score: {original_hps.item():.2f}")

# Step 2: Define FGSM attack
def fgsm_attack(image, epsilon, data_grad):
    """Apply FGSM attack by adding adversarial perturbation."""
    perturbed_image = image + epsilon * data_grad.sign()
    return torch.clamp(perturbed_image, 0, 1)  # Keep pixel values valid

# Step 3: Compute loss and generate adversarial example
loss = images.sum()  # Example: Sum of pixel values as a placeholder loss
loss.backward()  # Backpropagate to compute gradients

# Step 4: Apply the FGSM attack
epsilon = 0.0
data_grad = images.grad.data
adv_image_tensor = fgsm_attack(images, epsilon, data_grad)


# Save the adversarial image
adv_image = (adv_image_tensor[0].detach().permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
Image.fromarray(adv_image).save("adv_image.png")
adv_hpsv1_image = preprocess(Image.open("adv_image.png")).unsqueeze(0).to(device)
hpsv1_adv_images = torch.cat([adv_hpsv1_image], dim=0)

# Score the adversarial image
adv_score = hpsv2.score("adv_image.png", hps_prompt, hps_version="v2.1")[0]
with torch.no_grad():
    image_features = model.encode_image(hpsv1_adv_images)
    text_features = model.encode_text(hpsv1_text)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    adv_hps = image_features @ text_features.T

print(f"Hpsv2 Adversarial Score: {adv_score:.2f}")
print(f"Hpsv1 Adversarial Score: {adv_hps.item():.2f}")

# Step 5: Display the original and adversarial images with their scores
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(Image.open("original_image.png"))
#axes[0].set_title(f"Original Score: {original_score:.2f}", fontsize=14)
axes[0].axis("off")

axes[1].imshow(Image.open("adv_image.png"))
#axes[1].set_title(f"Adversarial Score: {adv_score:.2f}", fontsize=14)
axes[1].axis("off")

plt.show()
