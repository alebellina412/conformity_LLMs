import torch
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModelForVision2Seq, AutoModelForImageTextToText, BitsAndBytesConfig, pipeline
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from huggingface_hub import login
from qwen_vl_utils import process_vision_info
import numpy as np
import pickle
import random
import time
from collections import Counter
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import warnings
warnings.filterwarnings("ignore")
from transformers.utils import logging
logging.set_verbosity_error()

def get_confident_responses(answer, N):

    confident_responses = [
        f"The answer is definitely {answer}.",
        f"I am sure the answer is {answer}.",
        f"I'm confident it's {answer}.",
        f"The correct answer is {answer}.",
        f"It’s clearly {answer}.",
        f"Without a doubt, it's {answer}.",
        f"{answer}, no question.",
        f"{answer} is absolutely correct.",
        f"I strongly believe it's {answer}.",
        f"I'm certain the answer is {answer}.",
    ]

    responses = random.choices(confident_responses, k=N)
    return responses

def get_neutral_responses(answer, N):

    neutral_responses = [
        f"I think the answer is {answer}.",
        f"My answer is {answer}.",
        f"I'm going with {answer}.",
        f"It could be {answer}.",
        f"I believe the answer is {answer}.",
        f"I'll say {answer}.",
        f"{answer} is my choice.",
        f"{answer}, I think.",
        f"{answer} seems right.",
        f"I'm leaning toward {answer}.",
    ]

    responses = random.choices(neutral_responses, k=N)
    return responses

def get_uncertain_responses(answer, N):

    uncertain_responses = [
        f"I'm not sure, but maybe it's {answer}.",
        f"{answer}? I guess.",
        f"It might be {answer}.",
        f"I think it could be {answer}, but I'm unsure.",
        f"I'm not certain, but I'll say {answer}.",
        f"{answer}? Not really sure.",
        f"{answer}? That’s just a guess.",
        f"I'm unsure, maybe {answer}.",
        f"{answer}? Possibly.",
        f"{answer}? I'm not confident.",
    ]

    responses = random.choices(uncertain_responses, k=N)
    return responses




def import_qwen_7B():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    
    return model, processor



def import_qwen_3B():

    # Check if GPU is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype="auto",
        device_map="auto"
    )
    
    # Load processor (tokenizer + image preprocessor)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", use_fast=True)
    
    # (Opzionale ma utile per debug) Verifica vocab info
    tokenizer = processor.tokenizer
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    return model, processor

def import_qwen_32B():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "Qwen/Qwen2.5-VL-32B-Instruct"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    
    return model, processor


def import_qwen2_2B():

    # Check if GPU is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype="auto",
        device_map="auto"
    )
    
    # Load processor (tokenizer + image preprocessor)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", use_fast=True)
    
    # # (Opzionale ma utile per debug) Verifica vocab info
    # tokenizer = processor.tokenizer
    # print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    return model, processor


def import_qwen2_7B():

    # Check if GPU is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype="auto",
        device_map="auto"
    )
    
    # Load processor (tokenizer + image preprocessor)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", use_fast=True)
    
    # # (Opzionale ma utile per debug) Verifica vocab info
    # tokenizer = processor.tokenizer
    # print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    return model, processor

def import_qwen_72B_quantized():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Quantization config (4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
        bnb_8bit_use_double_quant=True,
        llm_int8_threshold=6.0
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-72B-Instruct",
        quantization_config=bnb_config,
        device_map={"": 0},  # Force on GPU 0
        trust_remote_code=True
        # attn_implementation="flash_attention_2"  # Facoltativo ma consigliato
    )

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-72B-Instruct",
        trust_remote_code=True,
        use_fast=False
    )

    return model, processor


def import_qwen_32B_quantized():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Quantization config (4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_threshold=6.0
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-32B-Instruct",
        quantization_config=bnb_config,
        device_map={"": 0},  # Force on GPU 0
        trust_remote_code=True
        # attn_implementation="flash_attention_2"  # Facoltativo ma consigliato
    )

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-32B-Instruct",
        trust_remote_code=True,
        use_fast=False
    )

    return model, processor

def import_mistral_24B():

    model_checkpoint = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    processor = AutoProcessor.from_pretrained(model_checkpoint)

    model = AutoModelForVision2Seq.from_pretrained(
        model_checkpoint,
        device_map="cuda",
        torch_dtype=torch.bfloat16
    )

    return model, processor

# def import_gemma_27B():

#     login(token="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")

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model_id = "google/gemma-3-27b-it" # (1b, 4b, 12b, 27b)

#     # Carica manualmente il processore con use_fast=True
#     processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

#     pipe = pipeline(
#         "image-text-to-text",
#         model=model_id,
#         processor=processor,
#         device=device,
#         torch_dtype=torch.bfloat16
#     )
#     return pipe, processor


def create_chat_template(text, image):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": text},
            ],
        }
    ]
    return messages

def clean_reply(text, labels):
    #print(text)
    if ((labels[0] in text) and (labels[1] in text))  or ((labels[0] in text) and (labels[2] in text)) or ((labels[1] in text) and (labels[2] in text)):
        return "None"
    if labels[0] in text:
        return labels[0]
    elif labels[1] in text:
        return labels[1]
    elif labels[2] in text:
        return labels[2]
    else:
        return "None"


def single_query_qwen(prompt, image, labels, model, processor, generation_kwargs, print_flag):
    
    messages = create_chat_template(prompt, image)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(**inputs, **generation_kwargs)
    
    # Calcolo delle probabilità per le label specificate
    label_token_ids = {label: processor.tokenizer.convert_tokens_to_ids(label) for label in labels}
    step_logits = outputs.scores[0]  # Logits del primo passo di generazione
    step_probs = F.softmax(step_logits, dim=-1)

    output_scores = {label: step_logits[0, token_id].item() for label, token_id in label_token_ids.items()}
    prob_labels = {label: step_probs[0, token_id].item() for label, token_id in label_token_ids.items()}
    # print(prob_labels)
    if print_flag :
        for label, prob in prob_labels.items():
            print(f"Token '{label}': {prob:.10f}")
        for label, score in output_scores.items():
            print(f"Token '{label}': {score:.10f}")

    return prob_labels, output_scores

def batch_query_qwen(prompts, images, labels, model, processor, generation_kwargs, print_flag=False):
    assert len(prompts) == len(images), "prompts and images must have the same length"

    # Crea batch di messaggi e testi
    messages_batch = [create_chat_template(p, i) for p, i in zip(prompts, images)]
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_batch]
    image_inputs_list, video_inputs_list = zip(*[process_vision_info(m) for m in messages_batch])
    
    # Flat map: processor vuole liste di immagini e video
    # (se sono None, passiamo None)
    image_inputs = [img for sublist in image_inputs_list for img in sublist] if image_inputs_list[0] is not None else None
    video_inputs = [vid for sublist in video_inputs_list for vid in sublist] if video_inputs_list[0] is not None else None

    # Tokenizza tutto il batch
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(**inputs, **generation_kwargs)

    # Probabilità del primo token generato per ogni elemento del batch
    label_token_ids = {label: processor.tokenizer.convert_tokens_to_ids(label) for label in labels}
    step_logits = outputs.scores[0]  # Shape: [batch_size, vocab_size]
    step_probs = F.softmax(step_logits, dim=-1)  # Shape: [batch_size, vocab_size]

    # Calcola le probabilità per ogni label per ogni esempio del batch
    batch_probs = []
    batch_scores = []
    for i in range(step_probs.size(0)):
        scores = {label: step_logits[i, token_id].item() for label, token_id in label_token_ids.items()}
        probs = {label: step_probs[i, token_id].item() for label, token_id in label_token_ids.items()}
        if print_flag:
            # print(f"\nEsempio {i}:")
            for label, prob in probs.items():
                print(f"Token '{label}': {prob:.10f}")

        if print_flag:
            # print(f"\nEsempio {i}:")
            for label, score in scores.items():
                print(f"Token '{label}': {score:.10f}")

        batch_scores.append(scores)
        batch_probs.append(probs)

    return batch_probs, batch_scores


from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM
import torch
# va importato in un altro modo 1b
def import_gemma_1B():

    # === CONFIG ===
    model_id = "google/gemma-3-1b-it"
    
    # === MODELLO + PROCESSOR ===
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    model = Gemma3ForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16
    ).eval()

    return model, processor

def import_gemma_4B():

    # === CONFIG ===
    model_id = "google/gemma-3-4b-it"
    
    # === MODELLO + PROCESSOR ===
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16
    ).eval()

    return model, processor

def import_gemma_12B():

    # === CONFIG ===
    model_id = "google/gemma-3-12b-it"
    
    # === MODELLO + PROCESSOR ===
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16
    ).eval()

    return model, processor


def import_gemma_27B():

    # === CONFIG ===
    model_id = "google/gemma-3-27b-it"
    
    # === MODELLO + PROCESSOR ===
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16
    ).eval()

    return model, processor


def get_probs_for_labels_gemma(output, labels, tokenizer, print_flag=True):
    # Ottieni token ID per ogni label
    label_token_ids = {label: tokenizer.convert_tokens_to_ids(label) for label in labels}
    
    # Logits del primo step
    logits = output.scores[0][0]  # shape: [vocab_size]
    probs = F.softmax(logits.float(), dim=-1)

    # Estrai la probabilità per ogni label
    prob_labels = {label: probs[token_id].item() for label, token_id in label_token_ids.items()}
    output_scores = {label: logits[token_id].item() for label, token_id in label_token_ids.items()}
    
    # Stampa (opzionale)
    if print_flag:
        for label, prob in prob_labels.items():
            print(f"Token '{label}': {prob:.10f}")

    return prob_labels, output_scores

def clear_gemma_cache(model):
    """Pulisce la cache per evitare rallentamenti nel modello Gemma"""
    if hasattr(model, 'past_key_values'):
        model.past_key_values = None

    for module in model.modules():
        if hasattr(module, 'past_key_values'):
            module.past_key_values = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def single_query_gemma(prompt, image, labels, model, processor, generation_kwargs, print_flag) :
    # Pulizia cache PRIMA di ogni generazione
    clear_gemma_cache(model)

    # Costruzione prompt e input
    messages = create_chat_template(prompt, image)

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            **generation_kwargs
        )

    generated = output.sequences[0][input_len:]
    first_token_id = generated[0].item()
    first_token_str = processor.tokenizer.decode(first_token_id)

    if print_flag:
        print(f"\n👉 First generated token: '{first_token_str}' (ID: {first_token_id})\n")

    prob_labels, output_scores = get_probs_for_labels_gemma(
        output, labels, processor.tokenizer, print_flag
    )

    return prob_labels, output_scores



def prepare_inputs(inputs, device, model_dtype=torch.bfloat16):
    """
    Sposta gli input su device e converte le immagini al tipo corretto (bfloat16 per compatibilità con modello).
    """
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)
    return inputs

def single_query_mistral(prompt, image, labels, model, processor, generation_kwargs, print_flag):
    
    messages = create_chat_template(prompt, image)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    inputs = prepare_inputs(inputs, device="cuda", model_dtype=torch.bfloat16)

    outputs = model.generate(**inputs, **generation_kwargs)

    # Calcolo delle probabilità per le label specificate
    label_token_ids = {label: processor.tokenizer.convert_tokens_to_ids(label) for label in labels}
    step_logits = outputs.scores[0]  # Logits del primo passo di generazione
    step_probs = F.softmax(step_logits, dim=-1)

    prob_labels = {label: step_probs[0, token_id].item() for label, token_id in label_token_ids.items()}
    output_scores = {label: step_logits[0, token_id].item() for label, token_id in label_token_ids.items()}

    if print_flag :
        for label, prob in prob_labels.items():
            print(f"Token '{label}': {prob:.10f}")

    return prob_labels, output_scores


import torch
from PIL import Image
from transformers import AutoModelForCausalLM
import matplotlib.pyplot as plt
import io
import torch.nn.functional as F
import os
import re
import random

def import_ovis_34B():

    # load model
    model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Ovis2-34B",
                                                 torch_dtype=torch.bfloat16,
                                                 multimodal_max_length=32768,
                                                 trust_remote_code=True,
                                                 llm_attn_implementation="eager").cuda()
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()

    return model, text_tokenizer, visual_tokenizer


def import_ovis_16B():

    # load model
    model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Ovis2-16B",
                                                 torch_dtype=torch.bfloat16,
                                                 multimodal_max_length=32768,
                                                 trust_remote_code=True,
                                                 llm_attn_implementation="eager").cuda()
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()

    return model, text_tokenizer, visual_tokenizer


def import_ovis_8B():

    # load model
    model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Ovis2-8B",
                                                 torch_dtype=torch.bfloat16,
                                                 multimodal_max_length=32768,
                                                 trust_remote_code=True,
                                                 llm_attn_implementation="eager").cuda()
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()

    return model, text_tokenizer, visual_tokenizer

def import_ovis_4B():

    # load model
    model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Ovis2-4B",
                                                 torch_dtype=torch.bfloat16,
                                                 multimodal_max_length=32768,
                                                 trust_remote_code=True,
                                                 llm_attn_implementation="eager").cuda()
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()

    return model, text_tokenizer, visual_tokenizer

def import_ovis_2B():

    # load model
    model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Ovis2-2B",
                                                 torch_dtype=torch.bfloat16,
                                                 multimodal_max_length=32768,
                                                 trust_remote_code=True,
                                                 llm_attn_implementation="eager").cuda()
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()

    return model, text_tokenizer, visual_tokenizer

def import_ovis_1B():

    # load model
    model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Ovis2-1B",
                                                 torch_dtype=torch.bfloat16,
                                                 multimodal_max_length=32768,
                                                 trust_remote_code=True,
                                                 llm_attn_implementation="eager").cuda()
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()

    return model, text_tokenizer, visual_tokenizer

def import_ovis_gemma_9B():

    # load model
    
    model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Ovis1.6-Gemma2-9B",
                                                 torch_dtype=torch.bfloat16,
                                                 multimodal_max_length=32768,
                                                 trust_remote_code=True,
                                                 llm_attn_implementation="eager",
                                                 use_fast=True).cuda()
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()

    return model, text_tokenizer, visual_tokenizer






def single_query_ovis(model, text_tokenizer, visual_tokenizer, prompt, image_path, labels, print_flag):

    text = f'<image>\n{prompt}'
    images = [Image.open(image_path)]
    
    # === Preprocessing ===
    prompt_text, input_ids, pixel_values = model.preprocess_inputs(text, images, max_partition=9)
    input_ids = input_ids.to(model.device).unsqueeze(0)
    attention_mask = (input_ids != text_tokenizer.pad_token_id).long().to(model.device)
    
    if pixel_values is not None:
        pixel_values = pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)
    pixel_values = [pixel_values]
    
    # === Forward per ottenere logits del primo token generato ===
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=None,
            return_dict=True
        )
        logits = outputs.logits  # [1, seq_len, vocab_size]
        next_token_logits = logits[:, -1, :]  # solo l'ultimo token
        probs = F.softmax(next_token_logits, dim=-1)
    
    # === Estrazione token A/B ===
    label_token_ids = {label: text_tokenizer.convert_tokens_to_ids(label) for label in labels}
    logit_values = {label: next_token_logits[0, token_id].item() for label, token_id in label_token_ids.items()}
    prob_values = {label: probs[0, token_id].item() for label, token_id in label_token_ids.items()}
    
    # === Output: stampa logits e probabilità ===
    if print_flag == True:
        print("\n=== Risultati ===")
        for label in labels:
            print(f"Token '{label}': logit = {logit_values[label]:.4f}, prob = {prob_values[label]:.6f}")
    
    # generate complete output
        
    # # === Output generato completo ===
    # gen_kwargs = dict(
    #     max_new_tokens=64,
    #     do_sample=False,
    #     top_p=None,
    #     top_k=None,
    #     temperature=None,
    #     eos_token_id=model.generation_config.eos_token_id,
    #     pad_token_id=text_tokenizer.pad_token_id,
    #     use_cache=True,
    # )
    # with torch.inference_mode():
    #     output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
    #     output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
    #     print(f'Output:\n{output}')
    
    return prob_values, logit_values






import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import re

def create_image_color(reference_color, other_color, position,
                       directory_salvataggio, nome_file,
                       square_size=200, spacing=100, dpi=150):
    """
    Genera un'immagine con 3 quadrati (A, REFERENCE COLOR, B) ben spaziati e quadrati perfetti.
    """

    def parse_color(col):
        if isinstance(col, str):
            if re.match(r'^\(\d{1,3},\d{1,3},\d{1,3}\)$', col.strip()):
                return tuple(map(int, col.strip('()').split(',')))
            return col
        return col

    reference_color = parse_color(reference_color)
    other_color = parse_color(other_color)

    color_a = reference_color if position == 0 else other_color
    color_b = other_color if position == 0 else reference_color
    colors = [color_a, reference_color, color_b]
    labels = ["A", "REFERENCE\nCOLOR", "B"]

    fig_w = (square_size * 3 + spacing * 2 + 40) / dpi
    fig_h = (square_size + 100) / dpi  # spazio in basso per label

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_facecolor("white")
    ax.set_aspect('equal')
    ax.axis("off")

    x_start = 20
    y_start = 60

    for i in range(3):
        x0 = x_start + i * (square_size + spacing)
        rect = plt.Rectangle((x0, y_start), square_size, square_size,
                             facecolor=[c/255 if isinstance(c, int) else c for c in colors[i]],
                             edgecolor='black', linewidth=2)
        ax.add_patch(rect)

        ax.text(x0 + square_size / 2, y_start - 25, labels[i],
                ha="center", va="top", fontsize=14, fontweight="bold")

    ax.set_xlim(0, x_start + 3 * (square_size + spacing))
    ax.set_ylim(0, y_start + square_size + 40)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    os.makedirs(directory_salvataggio, exist_ok=True)
    save_path = os.path.join(directory_salvataggio, nome_file)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    buf.seek(0)
    Image.open(buf).save(save_path)
    # print(f"Immagine salvata in: {save_path}")

import matplotlib.pyplot as plt
from PIL import Image
import io

def generate_asch_image_two_lines(lengths, position,
                        base_width, base_height, spacing, scale,
                        save_path, dpi, labels):
    """
    Genera un'immagine tipo Asch con una linea di riferimento al centro
    e due linee (A, B) simmetricamente ai lati.
    """
    # Dimensioni in pixel
    image_width, image_height = int(base_width * scale), int(base_height * scale)
    fig_w = image_width / dpi
    fig_h = image_height / dpi

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_facecolor("white")
    ax.axis("off")

    # Coordinate base
    center_x = base_width // 2
    # spacing = 80  # distanza orizzontale tra centro e linee A/B
    y_base = 20

    # Reference line (al centro)
    reference_length = lengths[position]
    ax.plot([center_x, center_x], [y_base, y_base + reference_length], color="black", linewidth=4)
    ax.text(center_x, y_base - 20, "REFERENCE\nLINE", ha="center", va="top", fontsize=14, fontweight="bold")

    # Linea A a sinistra, linea B a destra
    x_positions = [center_x - spacing, center_x + spacing]
    for x, h, label in zip(x_positions, lengths, labels):
        ax.plot([x, x], [y_base, y_base + h], color="black", linewidth=4)
        ax.text(x, y_base - 10, label, ha="center", va="top", fontsize=14, fontweight="bold")

    ax.set_xlim(0, base_width)
    ax.set_ylim(0, base_height)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Salva immagine
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)

    img = Image.open(buf).convert("L")  # grayscale
    img.save(save_path, optimize=True)

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import os

def create_image_dots(n_reference, n_other, position,
                      directory_salvataggio, nome_file,
                      square_size=200, spacing=100, dpi=150, dot_radius=5):
    """
    Genera un'immagine con 3 quadrati contenenti puntini (dots): A, REFERENCE BOX, B.
    """

    def generate_dots(n, square_size, margin=10):
        # Coordinate casuali per i dots dentro il quadrato
        x = np.random.uniform(margin, square_size - margin, size=n)
        y = np.random.uniform(margin, square_size - margin, size=n)
        return x, y

    # Determina i conteggi per A, REF, B
    count_a = n_reference if position == 0 else n_other
    count_b = n_other if position == 0 else n_reference
    dot_counts = [count_a, n_reference, count_b]
    labels = ["A", "REFERENCE\nBOX", "B"]

    fig_w = (square_size * 3 + spacing * 2 + 40) / dpi
    fig_h = (square_size + 100) / dpi

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_facecolor("white")
    ax.set_aspect('equal')
    ax.axis("off")

    x_start = 20
    y_start = 60

    for i in range(3):
        x0 = x_start + i * (square_size + spacing)

        # Disegna rettangolo
        rect = plt.Rectangle((x0, y_start), square_size, square_size,
                             facecolor='white', edgecolor='black', linewidth=2)
        ax.add_patch(rect)

        # Genera e disegna i puntini
        x_dots, y_dots = generate_dots(dot_counts[i], square_size)
        ax.scatter(x0 + x_dots, y_start + y_dots, s=dot_radius**2, color='black')

        # Label sotto
        ax.text(x0 + square_size / 2, y_start - 25, labels[i],
                ha="center", va="top", fontsize=14, fontweight="bold")

    ax.set_xlim(0, x_start + 3 * (square_size + spacing))
    ax.set_ylim(0, y_start + square_size + 40)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    os.makedirs(directory_salvataggio, exist_ok=True)
    save_path = os.path.join(directory_salvataggio, nome_file)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    buf.seek(0)
    Image.open(buf).save(save_path)

# create_image_dots(
#     n_reference=30,
#     n_other=15,
#     position=0,  # 0 → A è uguale alla reference
#     directory_salvataggio=f'images_dots_{model_label}',
#     nome_file="dots_example.png"
# )
