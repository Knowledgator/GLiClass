from typing import Dict, Union
import gradio as gr
import torch
from transformers import AutoTokenizer

from gliclass import GLiClassModel, ZeroShotClassificationPipeline

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_path = "models/gliclass/deberta_new_base/checkpoint-9000"
model = GLiClassModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)


pipeline = ZeroShotClassificationPipeline(model, tokenizer, classification_type='single-label', device=device)

text1 = """
"I recently purchased the Sony WH-1000XM4 Wireless Noise-Canceling Headphones from Amazon and I must say, I'm thoroughly impressed. The package arrived in New York within 2 days, thanks to Amazon Prime's expedited shipping.

The headphones themselves are remarkable. The noise-canceling feature works like a charm in the bustling city environment, and the 30-hour battery life means I don't have to charge them every day. Connecting them to my Samsung Galaxy S21 was a breeze, and the sound quality is second to none.

I also appreciated the customer service from Amazon when I had a question about the warranty. They responded within an hour and provided all the information I needed.

However, the headphones did not come with a hard case, which was listed in the product description. I contacted Amazon, and they offered a 10% discount on my next purchase as an apology.

Overall, I'd give these headphones a 4.5/5 rating and highly recommend them to anyone looking for top-notch quality in both product and service."""

text2 = """
Apple Inc. is an American multinational technology company headquartered in Cupertino, California. Apple is the world's largest technology company by revenue, with US$394.3 billion in 2022 revenue. As of March 2023, Apple is the world's biggest company by market capitalization. As of June 2022, Apple is the fourth-largest personal computer vendor by unit sales and the second-largest mobile phone manufacturer in the world. It is considered one of the Big Five American information technology companies, alongside Alphabet (parent company of Google), Amazon, Meta Platforms, and Microsoft. 
Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975 to develop and sell BASIC interpreters for the Altair 8800. During his career at Microsoft, Gates held the positions of chairman, chief executive officer, president and chief software architect, while also being the largest individual shareholder until May 2014.
Apple was founded as Apple Computer Company on April 1, 1976, by Steve Wozniak, Steve Jobs (1955–2011) and Ronald Wayne to develop and sell Wozniak's Apple I personal computer. It was incorporated by Jobs and Wozniak as Apple Computer, Inc. in 1977. The company's second computer, the Apple II, became a best seller and one of the first mass-produced microcomputers. Apple went public in 1980 to instant financial success. The company developed computers featuring innovative graphical user interfaces, including the 1984 original Macintosh, announced that year in a critically acclaimed advertisement called "1984". By 1985, the high cost of its products, and power struggles between executives, caused problems. Wozniak stepped back from Apple and pursued other ventures, while Jobs resigned and founded NeXT, taking some Apple employees with him. 
"""

text3 = """
Several studies have reported its pharmacological activities, including anti-inflammatory, antimicrobial, and antitumoral effects. 
The effect of E-anethole was studied in the osteosarcoma MG-63 cell line, and the antiproliferative activity was evaluated by an MTT assay. 
It showed a GI50 value of 60.25 μM with apoptosis induction through the mitochondrial-mediated pathway. Additionally, it induced cell cycle arrest at the G0/G1 phase, up-regulated the expression of p53, caspase-3, and caspase-9, and down-regulated Bcl-xL expression. 
Moreover, the antitumoral activity of anethole was assessed against oral tumor Ca9-22 cells, and the cytotoxic effects were evaluated by MTT and LDH assays. 
It demonstrated a LD50 value of 8 μM, and cellular proliferation was 42.7% and 5.2% at anethole concentrations of 3 μM and 30 μM, respectively. 
It was reported that it could selectively and in a dose-dependent manner decrease cell proliferation and induce apoptosis, as well as induce autophagy, decrease ROS production, and increase glutathione activity. The cytotoxic effect was mediated through NF-kB, MAP kinases, Wnt, caspase-3 and -9, and PARP1 pathways. Additionally, treatment with anethole inhibited cyclin D1 oncogene expression, increased cyclin-dependent kinase inhibitor p21WAF1, up-regulated p53 expression, and inhibited the EMT markers.
"""
examples = [
    [   
        text1,
        "product review, sport, competition, electronics, positive feadback, negative feadback",
        0.5,
        False
    ],
    [
        text2,
        "business, computers, sport, politics, science",
        0.5,
        False
    ],
    [
        text3,
        "business, biology, science, politics, positive review",
        0.5,
        False
    ],    
]

def classification(
    text, labels: str, threshold: float, multi_label: bool = False
) -> str:
    labels = labels.split(",")
    if multi_label:
        pipeline.classification_type = 'multi-label'
    else:
        pipeline.classification_type = 'single-label'

    results = pipeline(text, labels, threshold=threshold)[0] #because we have one text
    
    predicts = {result['label']:float(result['score']) for result in results}
    # predicts = '\n'.join([f"{result['label']} => {result['score']}" for result in results])
    return predicts


with gr.Blocks(title="GLiClass-small-v1.0") as demo:
    with gr.Accordion("How to run this model locally", open=False):
        gr.Markdown(
            """
            ## Installation
            To use this model, you must install the GLiClass Python library:
            ```
            !pip install gliclass
            ```
         
            ## Usage
            Once you've downloaded the GLiClass library, you can import the GLiClassModel and ZeroShotClassificationPipeline classes.
            """
        )
        gr.Code(
            '''
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

model = GLiClassModel.from_pretrained("knowledgator/gliclass-small-v1")
tokenizer = AutoTokenizer.from_pretrained("knowledgator/gliclass-small-v1")

pipeline = ZeroShotClassificationPipeline(model, tokenizer, classification_type='multi-label', device='cuda:0')

text = "One day I will see the world!"
labels = ["travel", "dreams", "sport", "science", "politics"]
results = pipeline(text, labels, threshold=0.5)[0] #because we have one text

for result in results:
    print(result["label"], "=>", result["score"])
            ''',
            language="python",
        )

    input_text = gr.Textbox(
        value=examples[0][0], label="Text input", placeholder="Enter your text here"
    )
    with gr.Row() as row:
        labels = gr.Textbox(
            value=examples[0][1],
            label="Labels",
            placeholder="Enter your labels here (comma separated)",
            scale=2,
        )
        threshold = gr.Slider(
            0,
            1,
            value=0.3,
            step=0.01,
            label="Threshold",
            info="Lower the threshold to increase how many entities get predicted.",
            scale=1,
        )
        multi_label = gr.Checkbox(
            value=examples[0][2],
            label="Multi-label classification",
            info="Allow for multi-label classification?",
            scale=0,
        )
    output = gr.Label(label="Output", color="#4b5563")
    submit_btn = gr.Button("Submit")
    examples = gr.Examples(
        examples,
        fn=classification,
        inputs=[input_text, labels, threshold, multi_label],
        outputs=output,
        cache_examples=True,
    )

    # Submitting
    input_text.submit(
        fn=classification, inputs=[input_text, labels, threshold, multi_label], outputs=output
    )
    labels.submit(
        fn=classification, inputs=[input_text, labels, threshold, multi_label], outputs=output
    )
    threshold.release(
        fn=classification, inputs=[input_text, labels, threshold, multi_label], outputs=output
    )
    submit_btn.click(
        fn=classification, inputs=[input_text, labels, threshold, multi_label], outputs=output
    )
    multi_label.change(
        fn=classification, inputs=[input_text, labels, threshold, multi_label], outputs=output
    )

demo.queue()
demo.launch(debug=True, share=True)