from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from rag_tool import rag_search

MODEL_PATH = "C:/Users/翟君泽/Desktop/SCSE_26_GEX3-main/models/qwen3-0.6b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)


def answer(question):
    context = rag_search(question)

    if context == "NO_RELEVANT_INFO":
        return "I don't know. I do not have that information."

    prompt = f"""You are a helpful assistant.
Use ONLY the context below to answer. If the answer is not in the context, say you don't know.
Do not invent phone numbers, emails, or any information not present in the context.

Context:
{context}

Question: {question}
Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 只取 Answer: 之后的内容
    if "Answer:" in text:
        text = text.split("Answer:")[-1].strip()
    return text


if __name__ == "__main__":
    q = "What exact phone number should I call for the University IT Service Desk?"
    print(answer(q))