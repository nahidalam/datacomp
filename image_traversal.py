'''
Image traversal: https://github.com/facebookresearch/meru/blob/main/scripts/image_traversals.py
Assets: https://github.com/facebookresearch/meru/tree/main/assets 
'''

import argparse
import copy
import json
import os
import pickle
import re
import shutil
import time
import warnings
from pathlib import Path
from time import gmtime, strftime

import numpy as np
import requests
import yaml
from cloudpathlib import CloudPath
from huggingface_hub import (
    CommitOperationAdd,
    HfApi,
    Repository,
    dataset_info,
    delete_folder,
    upload_file,
)
from requests.structures import CaseInsensitiveDict

from eval_utils.main import evaluate_model
from scale_configs import available_scales, get_scale_config

import torch
import open_clip
from open_clip import METRICS
from PIL import Image
from torchvision import transforms as T

warnings.filterwarnings("ignore", message="Length of IterableDataset")




def path_or_cloudpath(s):
    if re.match(r"^\w+://", s):
        return CloudPath(s)
    return Path(s)



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
    caption_tokens = tokenizer(pexels_text["captions"])
    all_text_feats.append(model.encode_text(caption_tokens, project=True))

    # Tokenize and encode prompts filled with tags.
    # Extract features of all captions and tags.
    noun_prompt_tokens = tokenizer(
        [NOUN_PROMPT.format(tag) for tag in pexels_text["nouns"]]
    )
    all_text_feats.append(model.encode_text(noun_prompt_tokens, project=True))

    adj_prompt_tokens = tokenizer(
        [ADJ_PROMPT.format(tag) for tag in pexels_text["adjectives"]]
    )
    all_text_feats.append(model.encode_text(adj_prompt_tokens, project=True))

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


def calc_scores(
    model, image_feats: torch.Tensor, text_feats: torch.Tensor, has_root: bool
):
    """
    Calculate similarity scores between the given image and text features depending
    on model type.

    Args:
        has_root: Flag to indicate whether the last text embedding (at dim=0)
            is the `[ROOT]` embedding.
    """

    geometry = model.geometry.split('-')[0]
    metric = METRICS[geometry]
    model_out = model(image_feats, text_feats)
    curvature = model_out["curvature"]
    return metric(image_feats, text_feats, curvature)
    
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_path",
        type=str,
        required=True, 
        help="Path to an image (.jpg) for perfoming traversal.",
    )
    parser.add_argument(
        "--train_output_dir",
        required=True,
        help="Path to output directory from training.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Path to output directory to use for evaluation. If nothing is passed, use the training output dir.",
    )
    parser.add_argument("--steps", default=50, type=int, help="Number of traversal steps.")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--model_arch", default=64, type=int, help="Model Arch eg. ViT-B-32-no-final-ln.")
    parser.add_argument(
        "--model_path",
        type=str,
        help='Model checkpoint path',
        default=None,
    )

    args = parser.parse_args()
          
    # Create model
    model, transform, device = create_model(args.model_arch, args.model_path)

    ## TODO: compute root_feat - not sure how :(
    root_feat = torch.zeros(model.visual.output_dim, device=device)
    
    # If no external text features are provided, use captions/tags from pexels.
    text_pool, text_feats_pool = get_text_feats(args.model_path, args.model_arch)

    # Add [ROOT] to the pool of text feats.
    text_pool.append("[ROOT]")
    text_feats_pool = torch.cat([text_feats_pool, root_feat[None, ...]])


    # ------------------------------------------------------------------------
    print(f"\nPerforming image traversals...")
    # ------------------------------------------------------------------------
    image = Image.open(args.image_path).convert("RGB")

    image_transform = T.Compose(
        [T.Resize(224, T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor()]
    )
    image = image_transform(image).to(device)

    ## get image and text feature https://github.com/EIFY/open_clip/blob/2b8bd6fa6377e56b7ce1a700cfa571b51746533c/src/open_clip/model.py#L332-L340
    image_feats, text_features, logit_scale, logit_bias, curvature = model(image = image)

    interp_feats = interpolate(model, image_feats, root_feat, args.steps)
    nn1_scores = calc_scores(model, interp_feats, text_feats_pool, has_root=True)

    nn1_scores, _nn1_idxs = nn1_scores.max(dim=-1)
    nn1_texts = [text_pool[_idx.item()] for _idx in _nn1_idxs]

    # De-duplicate retrieved texts (multiple points may have same NN) and print.
    print(f"Texts retrieved from [IMAGE] -> [ROOT] traversal:")
    unique_nn1_texts = []
    for _text in nn1_texts:
        if _text not in unique_nn1_texts:
            unique_nn1_texts.append(_text)
            print(f"  - {_text}")
    

    
