import os
import gradio as gr
import torch
from transformers import LlamaForCausalLM
from peft import PeftModel
import sentencepiece as spm

# Offline-Modus
os.environ["HF_HUB_OFFLINE"] = "1"

# Pfade
base_model_path = "/Users/fmorena/PycharmProjects/medical_chatbot/tinyllama_checkpoint/checkpoint-50"
adapter_path = "./medical-bot-final"
tokenizer_model_path = "/Users/fmorena/PycharmProjects/medical_chatbot/tinyllama_checkpoint/checkpoint-50/tokenizer.model"

# Lade SentencePiece Tokenizer
print("Lade SentencePiece Tokenizer...")
sp = spm.SentencePieceProcessor()
sp.load(tokenizer_model_path)

# Lade das Basis-Modell
print("Lade Basis-Modell...")
model = LlamaForCausalLM.from_pretrained(
    base_model_path,
    local_files_only=True,
    device_map="auto",
    torch_dtype=torch.float16
)

# Lade den trainierten Adapter
print("Lade medizinischen Adapter...")
model = PeftModel.from_pretrained(model, adapter_path)

def generate_response(user_input):
    # Prompt formatieren
    prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"

    # Mit SentencePiece tokenisieren
    input_ids = sp.encode(prompt, add_bos=True)
    input_tensor = torch.tensor([input_ids]).to(model.device)

    # Antwort generieren
    with torch.no_grad():
        outputs = model.generate(
            input_tensor,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            eos_token_id=sp.piece_to_id("</s>")
        )

    # Antwort dekodieren
    full_output = sp.decode(outputs[0].tolist())
    if "<|assistant|>" in full_output:
        response = full_output.split("<|assistant|>")[-1].strip()
    else:
        response = full_output

    return response

# Gradio Interface
interface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=2, placeholder="Stelle deine medizinische Frage..."),
    outputs="text",
    title="⚕️ Medizinischer Chatbot",
    description="Stelle medizinische Fragen (keine Diagnose!)",
    examples=[
        ["Was hilft bei Kopfschmerzen?"],
        ["Welche Impfungen sind wichtig für Erwachsene?"],
        ["Wie erkennt man eine Erkältung?"]
    ]
)

interface.launch()

