import torch
import hydra
from exceptiongroup import print_exception
from omegaconf import OmegaConf

from mineclip import MineCLIP
from PIL import Image
import torchvision.transforms as T
import numpy as np
from pathlib import Path
import gc
import json

# torch.manual_seed(42)
# np.random.seed(42)

# 1 TODO: hook output attention activations
# 2 TODO: create C matrix
# 3 TODO: implement textspan

def extract_prompts(json_data):
    prompts = []

    # Extract from metadata
    if 'metadata' in json_data:
        if 'title' in json_data['metadata']:
            prompts.append(json_data['metadata']['title'])

    # Extract from tables
    if 'tables' in json_data:
        for table in json_data['tables']:
            # Get header texts
            if 'headers' in table and 'text' in table['headers']:
                prompts.extend(table['headers']['text'])

            # Get cell texts
            if 'cells' in table and 'text' in table['cells']:
                prompts.extend(table['cells']['text'])

    # Extract from images
    if 'images' in json_data:
        for image in json_data['images']:
            if 'alt_text' in image and image['alt_text']:
                prompts.append(image['alt_text'])

    # Clean prompts
    cleaned_prompts = []
    for prompt in prompts:
        if isinstance(prompt, str):  # Make sure it's a string
            # Remove empty strings and clean whitespace
            cleaned = prompt.strip()
            if cleaned and cleaned not in cleaned_prompts:
                cleaned_prompts.append(cleaned)

    return cleaned_prompts


def compute_contribution_matrix(recorded_blocks_list, projection_matrix, layer_idx, head_idx):
    """
    Compute contribution matrix for a specific attention head and layer across multiple images
    """
    # Get dimensions from first batch
    batch_size, num_heads, seq_len, _ = recorded_blocks_list[0][layer_idx]['attention_patterns'].shape
    embed_dim = projection_matrix.shape[1]
    total_batches = len(recorded_blocks_list)
    device = projection_matrix.device

    # Initialize contribution tensor for this specific head
    c = torch.zeros((total_batches * seq_len, embed_dim), device=device)

    for batch_idx, recorded_blocks in enumerate(recorded_blocks_list):
        # Move tensors to GPU and get patterns/output for specific layer
        attn_patterns = recorded_blocks[layer_idx]['attention_patterns'].to(device)
        layer_output = recorded_blocks[layer_idx]['output'].to(device).permute(1, 0, 2)

        # Get attention for specific head
        head_attention = attn_patterns[:, head_idx, :, :]

        for i in range(seq_len):
            pos_attention = head_attention[:, :, i].unsqueeze(-1)
            weighted_output = layer_output * pos_attention
            avg_contribution = weighted_output.mean(dim=0)

            # Store in the appropriate position
            idx = batch_idx * seq_len + i
            c[idx] = (avg_contribution[i] @ projection_matrix)

        # Clear GPU memory after each batch
        del attn_patterns
        del layer_output
        torch.cuda.empty_cache()

    return c


# And in the VisionTransformer class:
def forward_batch(self, image_list):
    """
    Process a list of images and accumulate their recorded blocks
    """
    recorded_blocks_list = []
    features_list = []

    for images in image_list:
        # Clear previous recorded blocks
        self.recorded_blocks = {}

        # Forward pass
        features = self.forward(images)

        # Store recorded blocks
        recorded_blocks_list.append(self.recorded_blocks.copy())
        features_list.append(features)

    return torch.cat(features_list, dim=0), recorded_blocks_list


def textspan(head_contributions, text_embeddings, m=5):
    """
    Algorithm 1: TextSpan

    Args:
        head_contributions: C matrix for a specific head [K x d']
        text_embeddings: R matrix of text embeddings [M x d']
        m: Number of components to find

    Returns:
        selected_texts: Indices of selected text descriptions
        projected_contributions: The projected representations C'
    """
    K, d = head_contributions.shape
    M, _ = text_embeddings.shape

    # Initialize
    C = head_contributions.clone()
    R = text_embeddings.clone()
    C_proj = torch.zeros_like(head_contributions)
    selected_texts = []

    for i in range(m):
        # Compute D = RC^T
        D = R @ C.T  # [M x K]

        # Find text with highest variance
        variances = torch.var(D, dim=1)  # [M]
        j_star = torch.argmax(variances)
        selected_texts.append(j_star)

        # Get the selected text direction
        r_star = R[j_star]  # [d']

        # Project contributions onto this direction
        proj_scalar = (C @ r_star) / (r_star @ r_star)  # [K]
        projection = torch.outer(proj_scalar, r_star)  # [K x d']

        # Update C' and remove projection from C
        C_proj += projection
        C = C - projection

        # Remove projection from remaining text embeddings
        R_proj_scalar = (R @ r_star) / (r_star @ r_star)  # [M]
        R = R - torch.outer(R_proj_scalar, r_star)

    return selected_texts, C_proj

def move_blocks_to_cpu(recorded_blocks):
    cpu_blocks = {}
    for key, value in recorded_blocks.items():
        if torch.is_tensor(value):
            cpu_blocks[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_blocks[key] = {
                sub_key: sub_value.cpu() if torch.is_tensor(sub_value) else sub_value
                for sub_key, sub_value in value.items()
            }
        else:
            cpu_blocks[key] = value
    return cpu_blocks

def print_system_memory():
    import psutil
    print(f"RAM Usage: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")


@torch.no_grad()
@hydra.main(config_name="conf", config_path=".", version_base="1.1")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OmegaConf.set_struct(cfg, False)
    cfg.pop("ckpt")
    OmegaConf.set_struct(cfg, True)

    model = MineCLIP(**cfg).to(device)
    model.load_ckpt('/mnt/c/Users/georg/OneDrive/Desktop/Coding/coding/MineCLIP_forked/checkpoints/attn.pth', strict=True)
    print(device)

    recorded_blocks_list = []

    transform = T.Compose([
        T.Resize((160, 256)),  # Match your dimensions
        T.ToTensor(),  # Convert to tensor [3, 160, 256]
        T.Lambda(lambda x: x * 255)  # Scale to 0-255 range like your random input
    ])

    image_files = [f for f in Path('/mnt/c/Users/georg/OneDrive/Desktop/Coding/coding/MineCLIP_forked/data/wiki_samples/Diamond/images').glob('*') if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]

    for img_path in image_files:
        print_system_memory()
        try:
            # Load and process image
            image = Image.open(img_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
                print(f"Converted {img_path.name} to RGB")
            single_frame = transform(image)

            # Create video tensor
            video_frames = single_frame.unsqueeze(0).repeat(16, 1, 1, 1)
            video = video_frames.unsqueeze(0).repeat(6, 1, 1, 1, 1)
            video = video.to(device)

            # Clear previous recorded blocks
            model.image_encoder.recorded_blocks = {}

            # Forward pass
            image_feats, recorded_blocks = model.forward_image_features(video)

            recorded_blocks = move_blocks_to_cpu(recorded_blocks)

            # Clear GPU memory
            del video
            del video_frames
            del single_frame
            del image_feats
            torch.cuda.empty_cache()

            # Store CPU version in list
            recorded_blocks_list.append(recorded_blocks)

            # Store recorded blocks
            # recorded_blocks_cpu = {k: v.cpu().detach() for k, v in recorded_blocks.items()}
            # recorded_blocks_list.append(recorded_blocks_cpu)

            print(f"Processed {img_path.name}")

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

        # video = torch.randint(0, 255, (6, 16, 3, 160, 256), device=device)
        prompts = [
            "a minecraft villager",
            "a creaper in minecraft",
            "Feel free to also checkout MineDojo at",
            "Minecraft is a sandbox video game developed by Mojang Studios",
            "the minecraft inventory menu",
            'the hockey player james swan',
            'the minecraft villager',
            'villager',
            'diamond pickaxe',
        ]
        # VIDEO_BATCH, TEXT_BATCH = video.size(0), len(prompts)

        # print(model.eval())
        # model.eval()

        # image_feats, recorded_blocks = model.forward_image_features(video)
        # recorded_blocks_list.append(recorded_blocks)
        # video_feats = model.forward_video_features(image_feats)
        # assert video_feats.shape == (VIDEO_BATCH, 512)
        # print('encoding video features')
        # video_feats_2, contribution_matrix = model.encode_video(video)
        # print('encoded video features')
        # quit()
        # encode_video is equivalent to forward_video_features(forward_image_features(video))
        # torch.testing.assert_allclose(video_feats, video_feats_2)


    # print(recorded_blocks_list[0][11]['attention_patterns'].shape)
    # print(len(recorded_blocks_list))

        # encode batch of prompts
    # assert text_feats_batch.shape == (TEXT_BATCH, 512)

    # print(video_feats.shape)
    # print(text_feats_batch.shape)
    # quit()

    # extracting text
    prompts_pre_tokenized = []

    wsl_path = "/mnt/c/Users/georg/OneDrive/Desktop/Coding/coding/MineCLIP_forked/data/wiki_samples/Diamond/data.json"

    try:
        with open(wsl_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        prompts = extract_prompts(data)

        # print("Extracted Prompts:")
        for i, prompt in enumerate(prompts, 1):
            # print(f"{i}. {prompt}")
            prompts_pre_tokenized.append(prompt)

    except Exception as e:
        print(f"Error processing JSON: {e}")

    text_feats_batch = model.encode_text(prompts)


    layer = 11
    head = 8



    contribution_matrix = compute_contribution_matrix(recorded_blocks_list, model.image_encoder.projection, layer, head)

    print(contribution_matrix.shape)

    texts, C_proj = textspan(contribution_matrix, text_feats_batch)
    print(texts)
    print(C_proj.shape)
    print([prompts[idx] for idx in [118, 218, 72, 241, 38]])

        # compute reward from features
        # logits_per_video, logits_per_text = model.forward_reward_head(
        #     video_feats, text_tokens=text_feats_batch
        # )
        # assert logits_per_video.shape == (VIDEO_BATCH, TEXT_BATCH)
        # assert logits_per_text.shape == (TEXT_BATCH, VIDEO_BATCH)
        # # directly pass in strings. This invokes the tokenizer under the hood
        # reward_scores_2, _ = model.forward_reward_head(video_feats, text_tokens=prompts)
        # # pass in cached, encoded text features
        # reward_scores_3, _ = model(
        #     video_feats, text_tokens=text_feats_batch, is_video_features=True
        # )
        # reward_scores_4, _ = model(
        #     video, text_tokens=text_feats_batch, is_video_features=False
        # )
        # # all above are equivalent, just starting from features or raw values
        # torch.testing.assert_allclose(logits_per_video, reward_scores_2)
        # torch.testing.assert_allclose(logits_per_video, reward_scores_3)
        # torch.testing.assert_allclose(logits_per_video, reward_scores_4)

    print("Inference successful")


if __name__ == "__main__":
    main()
