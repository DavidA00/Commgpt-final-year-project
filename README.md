# CommGPT — A Teaching-Assistant LLM for Advanced Communication Systems

[**→ Model weights on Hugging Face: huggingface.co/dabboud/Commgpt-3B**](https://huggingface.co/dabboud/Commgpt-3B)
---

## 1 · What is CommGPT?

*CommGPT* is a conversational language model built on **Qwen 2.5-3B-Instruct** and fine-tuned exclusively for the syllabus of **EECE 442 / Advanced Communication Systems** at the American University of Beirut.  Paired with a retrieval-augmented generation (RAG) backend during development, the model learned to answer domain-specific questions, walk through derivations, and offer study-friendly explanations in plain English.  While the full deployment stack involves indexing and GUI components, this repository contains only the **final weights** and the **comprehensive implementation report** that documents every design decision from data wrangling to fine-tuning.

---

## 2 · Executive Summary

The CommGPT project attempts to create an AI teaching assistant that supports students in EECE 442, Communications Systems. Its motivation originates from the need of specialized chatbots that can provide guidance within technical fields like Communication Systems, where detailed, domain-specific knowledge is essential. Existing solutions lack the specificity and interactivity required for higher-level technical education, which is what drives this project. The design aims to provide responses that guide students towards solutions, mirroring the role of a human teaching assistant.

Several constraints impact the project, including computational limitations due to reliance on limited resources and technicalities, data security, sufficient accuracy, clarity and coherence of responses, and educational responsibility. These constraints guide the design and implementation choices to ensure that the system meets technical, qualitative, ethical, and educational standards.

The proposed solution involves a multistage system. It integrates retrieval-augmented generation (RAG) fusing the retrieval of open source documents based on cosine similarity between the query and specially created summaries of them with retrieved proprietary documents form another database. The documents are then reranked using a crossencoder, and
passed to an LLM that was fine-tuned successively on easy, medium, and hard Q\&A with our generated and curated domain-specific data. We developed a graphical user interface (GUI) using Gradio, integrating conversation history and a customizable system prompt. 

Additionally, we constructed a dataset of 450 multiple-choice questions tailored specifically to Communication Systems for evaluating large language models (LLMs). 
Our final RAG and Fine tuning setup with Qwen 2.5 3B model successfully achieved 61.33 \% score compared with a score of 57.11 \% with the base model. As for mathsral 7B, our RAG setup achieved a score of 70.22 \% vs a score of 58.22 \% with the base model. 


## 3 · Using the weights

Below is a minimal snippet that loads the model for pure text generation (without RAG context).  It still answers many EECE 442 questions from its internal knowledge learned during fine-tuning.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "dabboud/Commgpt-3B"

tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")

chat = pipeline("text-generation", model=model, tokenizer=tok)
print(chat("Explain why the matched filter maximizes SNR in the presence of AWGN.",
           max_new_tokens=160)[0]["generated_text"])

```

If you wish to recreate the full RAG assistant, consult the pipeline diagrams and parameter tables in Chapters 3 and 4 of the report, then plug in your own domain documents.

## 4 · Acknowledgements
CommGPT exists thanks to:

- Prof. Ibrahim Abou Faycal and Prof. Jihad Fahs for granting access to proprietary textbook drafts and providing expert feedback.
- Alibaba for releasing Qwen 2.5 under an open license.
- Marker developers for their PDF→ Markdown tool.
- The authors of Stella-400 M and MS-MARCO Cross-Encoder for open-sourcing strong embedding models.

See the report for more. 

## 5 · License
The CommGPT weights and this README are released under the MIT License.
All third-party documents referenced during training retain their original licenses and are not redistributed here.

## 6 · Suggested citation

```latex
@misc{CommGPT2025,
  title   = {CommGPT: A Domain-Tuned Qwen 2.5-3B Model for Advanced Communication Systems},
  author  = {
    Abboud, David and  
    Eid, Alex and  
    Menassa, Alexander and  
    Abou Faycal, Ibrahim and  
    Fahs, Jihad and  
    Zaraket, Fadi and  
    Chokr, Sally
  },
  note    = {Model and report available at \url{https://huggingface.co/dabboud/Commgpt-3B}},
  year    = {2025}
}
```

