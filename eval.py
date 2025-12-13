import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = "Qwen/Qwen3-1.7B-Base"
adapter_path = "./qwen_sft_vr_run2/final"
scenes_path = "./samples/scenes.json"
output_path = "completions.json"
max_tokens = 512

with open(scenes_path, "r") as f:
    scenes = json.load(f)

# Load model & tokenizer.
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(model, adapter_path)
model.to("cuda")
model.eval()

# Generate completions.
def generate_output(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove prompt.
    text = text[len(prompt):]

    # Hard stop at HTML_END.
    if "HTML_END" in text:
        text = text.split("HTML_END")[0] + "HTML_END"

    return text.strip()

# HTML processing.
def extract_html_block(text):
    match = re.search(
        r'<!DOCTYPE html>.*?</html>',
        text,
        flags=re.DOTALL | re.IGNORECASE
    )
    return match.group(0) if match else ""

def normalize_html(html_string):
    html_string = re.sub(r'HTML_START|HTML_END', '', html_string, flags=re.IGNORECASE)
    html_string = "\n".join(line.strip() for line in html_string.splitlines())
    html_string = re.sub(r'\s+', ' ', html_string)
    return html_string.strip().lower()

# Evaluation loop.
completions = []
correct = 0

for idx, scene in enumerate(scenes, start=1):
    print(f"Processing scene {scene['scene_id']} ({idx}/{len(scenes)})")

    prompt = scene["prompt"].strip() + "\nHTML_START\n"
    solution = scene["html"]

    prediction = generate_output(prompt)

    pred_html = extract_html_block(prediction)
    sol_html = extract_html_block(solution)

    is_match = normalize_html(pred_html) == normalize_html(sol_html)

    if is_match:
        correct += 1

    completions.append({
        "scene_id": scene["scene_id"],
        "prompt": scene["prompt"],
        "solution": solution,
        "completion": prediction,
        "match": is_match
    })

    # Save after every iteration.
    with open(output_path, "w") as f:
        json.dump(completions, f, indent=2)

    running_pass_1 = correct / idx
    print(f"Running Pass@1: {correct}/{idx} = {running_pass_1:.3f}\n")


final_pass_1 = correct / len(scenes)
print("Final Pass@1:", final_pass_1)