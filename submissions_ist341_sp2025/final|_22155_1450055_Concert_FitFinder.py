


import os
import openai
import gradio as gr

openai.api_key = "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"
client = openai.OpenAI(api_key=openai.api_key)



def parse_outfit_text(text):
    import re
    sections = {"Top": "", "Bottoms": "", "Shoes": "", "Accessories": ""}
    matches = re.findall(r"(Top|Bottoms|Shoes|Accessories):\s*(.*)", text, re.IGNORECASE)
    for key, value in matches:
        sections[key.capitalize()] = value.strip()
    return sections



def generate_outfit_components(vibe, style, gender, no_items, top, bottoms, shoes, accessories):
    style_clause = f"Style preference: {style}" if style != "Any" else ""
    prompt = f"""
You are a fashion assistant helping finalize an outfit for a concert.

Concert vibe: "{vibe}"
{style_clause}
Gender identity: {gender}

The user already has:
Top: {top or 'None'}
Bottoms: {bottoms or 'None'}
Shoes: {shoes or 'None'}
Accessories: {accessories or 'None'}

The user explicitly does not want:
{no_items}

Strictly follow these instructions:
- Do NOT include any forbidden items from the list above.
- Do NOT modify any fields that are already filled in by the user.
- Only generate items that are empty and must still be suggested.
- Generate items that complement the fixed ones and match the vibe and style.

Respond ONLY in this format:
Top: ...
Bottoms: ...
Shoes: ...
Accessories: ...
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        parsed = parse_outfit_text(response.choices[0].message.content)
        return top or parsed["Top"], bottoms or parsed["Bottoms"], shoes or parsed["Shoes"], accessories or parsed["Accessories"]
    except Exception as e:
        return f" Error: {e}", "", "", ""



with gr.Blocks() as app:
    gr.Markdown("## 🎤 Concert Outfit Generator")

    with gr.Row():
        with gr.Column():
            vibe = gr.Textbox(label="Concert Vibe")
            style = gr.Dropdown(["Any", "Trendy", "Bold", "Casual", "Eco-Friendly", "Classic"], label="Style")
            gender = gr.Dropdown(["Feminine", "Masculine", "Gender-Neutral"], label="Gender Identity")
            no_items = gr.Textbox(label="Things to Avoid")

            top = gr.Textbox(label="Top (optional)", placeholder="Add a top you want to keep — or leave blank")
            bottoms = gr.Textbox(label="Bottoms (optional)", placeholder="Add bottoms you want to keep — or leave blank")
            shoes = gr.Textbox(label="Shoes (optional)", placeholder="Add shoes you want to keep — or leave blank")
            accessories = gr.Textbox(label="Accessories (optional)", placeholder="Add accessories you want to keep — or leave blank")

            suggest_btn = gr.Button("🔄 Suggest Missing Pieces")

        with gr.Column():
            output_top = gr.Textbox(label="Top")
            output_bottoms = gr.Textbox(label="Bottoms")
            output_shoes = gr.Textbox(label="Shoes")
            output_accessories = gr.Textbox(label="Accessories")

    suggest_btn.click(generate_outfit_components,
                      inputs=[vibe, style, gender, no_items, top, bottoms, shoes, accessories],
                      outputs=[output_top, output_bottoms, output_shoes, output_accessories])

app.launch(share=True)



