import torch
import hydra
from omegaconf import OmegaConf

from mineclip import MineCLIP
from PIL import Image
import torchvision.transforms as T
import numpy as np

# torch.manual_seed(42)
# np.random.seed(42)

# 1 TODO: hook output attention activations
# 2 TODO: create C matrix
# 3 TODO: implement textspan


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


@torch.no_grad()
@hydra.main(config_name="conf", config_path=".", version_base="1.1")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OmegaConf.set_struct(cfg, False)
    cfg.pop("ckpt")
    OmegaConf.set_struct(cfg, True)

    model = MineCLIP(**cfg).to(device)
    print(device)

    # Load image
    image = Image.open("/mnt/c/Users/georg/Desktop/coding/MineCLIP_fork/diamond_pickaxe.jpg")

    # Create transform pipeline
    transform = T.Compose([
        T.Resize((160, 256)),  # Match your dimensions
        T.ToTensor(),  # Convert to tensor [3, 160, 256]
        T.Lambda(lambda x: x * 255)  # Scale to 0-255 range like your random input
    ])

    # Process image
    single_frame = transform(image)

    # Expand dimensions to match [6, 16, 3, 160, 256]
    # First duplicate for 16 frames
    video_frames = single_frame.unsqueeze(0).repeat(16, 1, 1, 1)  # [16, 3, 160, 256]
    # Then duplicate for 6 batch items
    video = video_frames.unsqueeze(0).repeat(6, 1, 1, 1, 1)  # [6, 16, 3, 160, 256]

    # Move to correct device
    video = video.to(device)

    # video = torch.randint(0, 255, (6, 16, 3, 160, 256), device=device)
    prompts = [
        "a minecraft villager",
        "a creaper in minecraft",
        "Feel free to also checkout MineDojo at",
        "Minecraft is a sandbox video game developed by Mojang Studios",
        "the minecraft inventory menu",
        "a minecraft player holding a diamond sword",
        "crafting table",
        'crafting table in minecraft',
        'diamond pickaxe',
        'a minecraft diamond pickaxe',
        'a minecraft diamond pickaxe on a crafting table',
        'a minecraft pickaxe',
    ]
    VIDEO_BATCH, TEXT_BATCH = video.size(0), len(prompts)

    # print(model.eval())
    model.eval()

    image_feats= model.forward_image_features(video)
    video_feats = model.forward_video_features(image_feats)
    assert video_feats.shape == (VIDEO_BATCH, 512)
    print('encoding video features')
    # video_feats_2, contribution_matrix = model.encode_video(video)
    print('encoded video features')
    # quit()
    # encode_video is equivalent to forward_video_features(forward_image_features(video))
    # torch.testing.assert_allclose(video_feats, video_feats_2)

    # encode batch of prompts
    text_feats_batch = model.encode_text(prompts)
    assert text_feats_batch.shape == (TEXT_BATCH, 512)

    # print(video_feats.shape)
    # print(text_feats_batch.shape)
    # quit()

    layer = 10
    head = 8

    # texts, C_proj = textspan(contribution_matrix[layer, head], text_feats_batch)
    # print(texts)
    # print(C_proj.shape)

    # compute reward from features
    logits_per_video, logits_per_text = model.forward_reward_head(
        video_feats, text_tokens=text_feats_batch
    )
    assert logits_per_video.shape == (VIDEO_BATCH, TEXT_BATCH)
    assert logits_per_text.shape == (TEXT_BATCH, VIDEO_BATCH)
    # directly pass in strings. This invokes the tokenizer under the hood
    reward_scores_2, _ = model.forward_reward_head(video_feats, text_tokens=prompts)
    # pass in cached, encoded text features
    reward_scores_3, _ = model(
        video_feats, text_tokens=text_feats_batch, is_video_features=True
    )
    reward_scores_4, _ = model(
        video, text_tokens=text_feats_batch, is_video_features=False
    )
    # all above are equivalent, just starting from features or raw values
    torch.testing.assert_allclose(logits_per_video, reward_scores_2)
    torch.testing.assert_allclose(logits_per_video, reward_scores_3)
    torch.testing.assert_allclose(logits_per_video, reward_scores_4)

    print("Inference successful")


if __name__ == "__main__":
    main()
