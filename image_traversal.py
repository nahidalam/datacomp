'''
Image traversal: https://github.com/facebookresearch/meru/blob/main/scripts/image_traversals.py
Assets: https://github.com/facebookresearch/meru/tree/main/assets 
'''

import argparse
import json
import warnings
from pathlib import Path

import torch
import open_clip
from open_clip import METRICS
from PIL import Image
from torchvision import transforms as T

warnings.filterwarnings("ignore", message="Length of IterableDataset")


def euclidean_entailment(x, y, _, K):
    # https://arxiv.org/abs/1804.01882
    y_x = y - x
    x_norm = x.norm(dim=-1)
    ext = torch.acos((y_x * x).sum(-1) / (y_x.norm(dim=-1) * x_norm))
    if not K:
        return ext.mean()
    aper = torch.asin(torch.clamp(K / x_norm, max=1.))
    return torch.clamp(ext - aper, min=0.)

def _exponential_map(x, curvature):
    c_sqrt_norm = torch.sqrt(curvature) * x.norm(dim=-1, keepdim=True)
    x_space = torch.sinh(c_sqrt_norm) / c_sqrt_norm * x
    x_time = torch.sqrt(curvature.reciprocal() + (x_space ** 2).sum(-1))
    return x_space, x_time

def hyperbolic_entailment(x, y, curvature, K):
    # FP32 for exponential map and losses for numerical stability,
    # per https://arxiv.org/abs/2304.09172
    x, y, curvature = x.double(), y.double(), curvature.double()
    x_space, x_time = _exponential_map(x, curvature)
    x_space_norm = x_space.norm(dim=-1)
    y_space, y_time = _exponential_map(y, curvature)
    l = (x_space * y_space).sum(-1) - x_time * y_time
    c_l = curvature * l
    ext = torch.acos((y_time + x_time * c_l) / (x_space_norm * torch.sqrt(c_l ** 2 - 1.)))
    if not K:
        return ext.mean()
    aper = torch.asin(torch.clamp(2 * K / (torch.sqrt(curvature) * x_space_norm), max=1.))
    return torch.clamp(ext - aper, min=0.)

_ENTAILMENT = {
    'euclidean': euclidean_entailment,
    'hyperbolic': hyperbolic_entailment,
}

def create_model(model_arch, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    model_path = str(model_path)
    # Note: This assumes the default open_clip train_output_dir structure, including the
    # existence of the train_output_dir/params.txt file.
    train_output_dir = model_path.split('/')[:-2]
    params_file = '/'.join(train_output_dir + ["params.txt"])
    model_kwargs = {}
    with open(params_file) as f:
        for line in f:
            line = line.strip()
            if line == 'siglip: True':
                # Just use a placeholder value to make sure the state_dict matches.
                # This will get overwritten by the checkpoint and has no effect on eval at all.
                model_kwargs['init_logit_bias'] = 1.0
            else:
                l = line.split(': ')
                if l[0] == 'geometry':
                    model_kwargs['geometry'] = l[1]
    model, _, transform = open_clip.create_model_and_transforms(
        model_arch, pretrained=model_path, **model_kwargs
    )
    model.eval()
    # model.half()
    model = model.to(device)

    return model, transform, device

@torch.inference_mode()
def get_text_feats(model, model_arch) -> tuple[list[str], torch.Tensor]:
    # Get all captions, nouns, and ajectives collected from pexels.com website
    pexels_text = json.load(open("assets/pexels_text.json"))

    # Use very simple prompts for noun and adjective tags.
    tokenizer = open_clip.get_tokenizer(model_arch)
    NOUN_PROMPT = "a photo of a {}."
    ADJ_PROMPT = "this photo is {}."

    all_text_feats = []

    # Tokenize and encode captions.
    caption_tokens = tokenizer(pexels_text["captions"]).to(device)
    #all_text_feats.append(model.encode_text(caption_tokens, project=True))
    model_out = model(text = caption_tokens)
    caption_tokens_features = model_out[1]
    all_text_feats.append(caption_tokens_features)

    # Tokenize and encode prompts filled with tags.
    # Extract features of all captions and tags.
    noun_prompt_tokens = tokenizer(
        [NOUN_PROMPT.format(tag) for tag in pexels_text["nouns"]]
    ).to(device)
    #all_text_feats.append(model.encode_text(noun_prompt_tokens, project=True))
    model_out = model(text = noun_prompt_tokens)
    noun_prompt_tokens_features = model_out[1]
    all_text_feats.append(noun_prompt_tokens_features)

    adj_prompt_tokens = tokenizer(
        [ADJ_PROMPT.format(tag) for tag in pexels_text["adjectives"]]
    ).to(device)
    #all_text_feats.append(model.encode_text(adj_prompt_tokens, project=True))
    model_out = model(text = adj_prompt_tokens)
    adj_prompt_tokens_features = model_out[1]
    all_text_feats.append(adj_prompt_tokens_features)

    all_text_feats = torch.cat(all_text_feats, dim=0)
    all_pexels_text = [
        *pexels_text["captions"],
        *pexels_text["nouns"],
        *pexels_text["adjectives"],
    ]
    return all_pexels_text, all_text_feats

def interpolate(model, feats: torch.Tensor, root_feat: torch.Tensor, steps: int):
    """
    Interpolate between given feature vector and `[ROOT]` depending on model type 
    aka. generating intermediate embeddings that lie between the two given embeddings.
    """

    # Linear interpolation between root and image features. 

    interp_feats = [
        torch.lerp(root_feat, feats, weight.item())
        for weight in torch.linspace(0.0, 1.0, steps=steps)
    ]
    interp_feats = torch.stack(interp_feats)

    # Reverse the traversal order: (image first, root last)
    return interp_feats.flip(0)   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_path",
        type=str,
        required=True, 
        help="Path to an image (.jpg) for perfoming traversal.",
    )
    parser.add_argument("--steps", default=50, type=int, help="Number of traversal steps.")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--model_arch", type=str, help="Model Arch eg. ViT-B-32-no-final-ln.")
    parser.add_argument(
        "--model_path",
        type=str,
        help='Model checkpoint path',
        default=None,
    )
    parser.add_argument(
        "--min_radius",
        type=float,
        default=0.0,
        help="Radius of the epsilon-ball within which aperture is undefined."
    )

    args = parser.parse_args()
          
    # Create model
    model, transform, device = create_model(args.model_arch, args.model_path)

    ## TODO: compute root_feat - not sure how :(
    root_feat = torch.zeros(model.visual.output_dim, device=device)
    
    # If no external text features are provided, use captions/tags from pexels.
    text_pool, text_feats_pool = get_text_feats(model, args.model_arch)

    # Add [ROOT] to the pool of text feats.
    text_pool.append("[ROOT]")
    text_feats_pool = torch.cat([text_feats_pool, root_feat[None, ...]])


    # ------------------------------------------------------------------------
    print(f"\nPerforming image traversals...")
    # ------------------------------------------------------------------------
    image = Image.open(args.image_path)
    image = transform(image).to(device)

    ## get image and text feature https://github.com/EIFY/open_clip/blob/2b8bd6fa6377e56b7ce1a700cfa571b51746533c/src/open_clip/model.py#L332-L340       
    image_feats, _, _, _, curvature = model(image = image.unsqueeze(0))
    image_feats = image_feats[0]

    interp_feats = interpolate(model, image_feats, root_feat, args.steps)
    #nn1_scores = calc_scores(model, interp_feats, text_feats_pool, curvature, has_root=True)
    metric = METRICS[model.geometry]
    nn1_scores = metric(interp_feats, text_feats_pool, curvature)
    key = model.geometry.split('-')[0]
    if args.min_radius:
        entailment_energy = _ENTAILMENT[key](text_feats_pool[None, :, :], image_feats[:, None, :], curvature, args.min_radius)
        entailment_energy[..., -1] = 0
        nn1_scores[entailment_energy > 0] = -1e12

    nn1_scores, _nn1_idxs = nn1_scores.max(dim=-1)
    nn1_texts = [text_pool[_idx.item()] for _idx in _nn1_idxs]

    # De-duplicate retrieved texts (multiple points may have same NN) and print.
    print(f"Texts retrieved from [IMAGE] -> [ROOT] traversal:")
    unique_nn1_texts = []
    for _text in nn1_texts:
        if _text not in unique_nn1_texts:
            unique_nn1_texts.append(_text)
            print(f"  - {_text}")
    

    
